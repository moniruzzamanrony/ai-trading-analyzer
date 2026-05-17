"""
Multi-symbol, multi-horizon, multi-quantile XGBoost regression trainer.

Pipeline:
  1. Fetch lookback_days of 15m candles per symbol from Binance.
  2. Compute past-only features (15m + resampled 1h/4h + microstructure +
     cyclic + MACD-cycle position) once per symbol.
  3. For each training horizon H in `horizons`, label every candle with
     one of:
        - "cycle_capped" (default): max forward reach until next MACD red-start,
          capped at H — matches what the bot can actually realize given it
          exits on MACD red.
        - "atr": legacy fixed-window max forward reach.
     Both labels are ATR-normalized so the same target value means the same
     statistical "size of move" across volatility regimes / symbols.
  4. Append per-symbol one-hot columns (replaces ordinal LabelEncoder).
  5. Time-ordered concat across symbols → walk-forward TimeSeriesSplit
     (5 folds) with embargo = horizon between train/test and fit/val to
     prevent label-window leakage.
  6. For each (horizon, alpha) pair train an XGBRegressor with quantile
     loss + early stopping on a held-out tail. Run a realized-PnL
     backtest on the held-out tail that simulates the bot's actual
     exit policy (target hit OR MACD-red-start OR horizon expires) so
     the trainer can pick the *primary* alpha by realized profit rather
     than by raw hit-rate. Persist the nested bundle
        models[horizon][alpha]
     so the predictor can snap to the nearest available horizon at inference.
"""

import json
import logging
import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from app.core.config import settings
from app.services.data_fetcher import fetch_klines_history
from app.services.regression_features import (
    BASE_FEATURE_COLS,
    compute_regression_features,
    symbol_one_hot_cols,
)

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 730
_DEFAULT_LABEL_FORM = "cycle_capped"
_BUNDLE_VERSION = 3  # bumped when bundle schema changes
_CV_SPLITS = 5
_EARLY_STOPPING_ROUNDS = 50

_MODEL_FILENAME = "regression_xgb_model.pkl"
_METADATA_FILENAME = "regression_metadata.json"

_lock = threading.Lock()
_cache: dict = {}


def _model_path() -> Path:
    return settings.models_dir / _MODEL_FILENAME


def _metadata_path() -> Path:
    return settings.models_dir / _METADATA_FILENAME


# ── Label creation ─────────────────────────────────────────────────────────────

def _macd_red_start_mask(macd_hist: np.ndarray) -> np.ndarray:
    """Boolean array marking bars where macd_hist transitions from >0 to <=0."""
    n = macd_hist.shape[0]
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    out[1:] = (macd_hist[:-1] > 0) & (macd_hist[1:] <= 0)
    return out


def _dist_to_next_red(red_start: np.ndarray) -> np.ndarray:
    """For each i, distance to the next True in `red_start` (i.e. j-i where
    j is the smallest j >= i with red_start[j] = True). Returns len(arr) if
    no such j exists, so callers can clip with `min(dist, horizon)`."""
    n = red_start.shape[0]
    dist = np.full(n, n, dtype=np.int64)
    last = n
    for k in range(n - 1, -1, -1):
        if red_start[k]:
            last = k
        dist[k] = last - k
    return dist


def _create_labels_atr(df: pd.DataFrame, horizon: int) -> pd.Series:
    """target[i] = (max(high[i+1 : i+1+horizon]) - close[i]) / atr[i]"""
    high = df["high"]
    # Rolling forward-max via reverse trick (vectorized).
    fwd_max_inclusive = high[::-1].rolling(horizon, min_periods=horizon).max()[::-1]
    fwd_max = fwd_max_inclusive.shift(-1)  # max over [i+1 .. i+horizon]
    target = (fwd_max - df["close"]) / df["atr"]
    target = target.where((df["atr"] > 0) & (df["close"] > 0))
    return target


def _create_labels_cycle_capped(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    target[i] = (max(high[i+1 .. min(j-1, i+H)]) - close[i]) / atr[i]

    where j is the first MACD-red-start bar with j > i (the next bar where
    macd_hist transitions from > 0 to <= 0). If no red-start is hit within
    H bars, the cap is i+H — equivalent to `_create_labels_atr` in that case.

    This is the upside the bot can actually realize given it exits on MACD red,
    so it's a tighter, more honest training target than the unconstrained max.
    """
    n = len(df)
    if n < 2:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)

    high = df["high"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    atr = df["atr"].values.astype(np.float64)
    macd_hist = df["macd_hist"].values.astype(np.float64)

    red_start = _macd_red_start_mask(macd_hist)
    # dist_to_next_red[k] = distance from k to the next red-start at or after k.
    dist_next = _dist_to_next_red(red_start)

    target = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - 1):
        if not (atr[i] > 0 and close[i] > 0):
            continue
        # First red-start at or after i+1.
        d = dist_next[i + 1] if i + 1 < n else n
        if d == 0:
            # red-start at i+1 itself: no room to capture upside.
            continue
        # We can hold up to bar i+d (exclusive of the red-start bar at i+1+d).
        # j_last is the last index whose `high` counts toward upside.
        j_last = min(i + horizon, i + d)
        j_last = min(j_last, n - 1)
        if j_last < i + 1:
            continue
        max_h = high[i + 1 : j_last + 1].max()
        target[i] = (max_h - close[i]) / atr[i]

    return pd.Series(target, index=df.index)


def _create_labels(df: pd.DataFrame, horizon: int, label_form: str) -> pd.Series:
    if label_form == "atr":
        return _create_labels_atr(df, horizon)
    if label_form == "cycle_capped":
        return _create_labels_cycle_capped(df, horizon)
    raise ValueError(f"Unknown label_form: {label_form!r}")


# ── Per-symbol feature construction (shared across horizons) ───────────────────

def _build_per_symbol(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted({s.upper() for s in symbols}):
        logger.info("Fetching %s (lookback=%dd) …", symbol, lookback_days)
        try:
            df = fetch_klines_history(symbol, days=lookback_days)
            if df.empty:
                logger.warning("No candles for %s – skipping", symbol)
                continue
            df = compute_regression_features(df)
            df.dropna(
                subset=["ema9", "ema21", "rsi_14", "macd_hist", "atr"],
                inplace=True,
            )
            if df.empty:
                logger.warning("No rows after feature warm-up for %s – skipping", symbol)
                continue
            out[symbol] = df
        except Exception as exc:
            logger.error("Failed to fetch/feature %s: %s", symbol, exc)
    if not out:
        raise ValueError("No usable price data for any requested symbol")
    return out


def _build_horizon_dataset(
    per_symbol: dict[str, pd.DataFrame],
    horizon: int,
    sym_cols: list[str],
    label_form: str,
) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    for symbol, df in per_symbol.items():
        sub = df.copy()
        sub["target"] = _create_labels(sub, horizon, label_form)
        sub.dropna(subset=["target"], inplace=True)
        if sub.empty:
            continue
        for col in sym_cols:
            sub[col] = 1.0 if col == f"is_{symbol}" else 0.0
        parts.append(sub)
    if not parts:
        return pd.DataFrame(), []
    combined = pd.concat(parts).sort_index()
    feature_cols = [c for c in BASE_FEATURE_COLS if c in combined.columns] + sym_cols
    return combined, feature_cols


# ── Metrics ────────────────────────────────────────────────────────────────────

def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    delta = y_true - y_pred
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> dict:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    hit_rate = float(np.mean(y_true >= y_pred)) if len(y_true) else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "hit_rate": hit_rate,
        "pinball": _pinball_loss(y_true, y_pred, alpha),
    }


# ── Sample weighting & conformal calibration ───────────────────────────────────

def _time_decay_weights(n: int, half_life_frac: float = 0.5) -> np.ndarray:
    """Exponential time-decay weights – newest row = 1.0, half-weight at
    `half_life_frac * n` rows back. Reduces the influence of stale regimes
    on the final fit without discarding history outright."""
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if half_life_frac <= 0:
        return np.ones(n, dtype=np.float64)
    lam = np.log(2.0) / max(1.0, half_life_frac * n)
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    return np.exp(-lam * ages)


def _backtest_realized_pnl(
    *,
    pred_units: np.ndarray,        # one prediction per row, in ATR units
    close: np.ndarray,
    high: np.ndarray,
    atr: np.ndarray,
    macd_hist: np.ndarray,
    horizon: int,
    fee_pct: float,
    require_buy_signal: bool = True,
) -> dict:
    """
    Simulate the bot's actual exit policy bar-by-bar and report realized PnL.

    For each candidate entry bar i:
      - skip if pred profit (post-fee) <= 0
      - skip if `require_buy_signal` and bar i isn't a MACD green-start
        (approximates the bot's MACD buy gate; EMA/RSI gates are intentionally
        omitted — they'd require recomputing strategy logic in Python and
        the MACD gate dominates entry timing)
      - walk forward up to `horizon` bars:
          * exit at target if high[k] >= target_price (filled at target)
          * exit at close[k] if MACD red-start at k
          * exit at close[i+horizon] if horizon expires
      - apply round-trip fee_pct to gross return

    Returns counters that let callers compare per-alpha realized PnL.
    """
    n = close.shape[0]
    if n < 2:
        return {"trades_taken": 0, "hits": 0, "hit_fraction": 0.0,
                "mean_net_profit_pct": 0.0, "total_net_profit_pct": 0.0}

    # red-start happens *between* bars; for exit purposes we treat the
    # red-start bar itself as the bar we get out on at its close.
    red_start = _macd_red_start_mask(macd_hist)
    # buy-signal approximation: MACD green-start = prev <=0 → curr > 0.
    green_start = np.zeros(n, dtype=bool)
    green_start[1:] = (macd_hist[:-1] <= 0) & (macd_hist[1:] > 0)

    trades_taken = 0
    hits = 0
    total_net_pct = 0.0

    for i in range(n - 1):
        if require_buy_signal and not green_start[i]:
            continue
        if not (atr[i] > 0 and close[i] > 0):
            continue
        pred_return = float(pred_units[i]) * atr[i] / close[i]
        if pred_return <= 0:
            continue
        # Skip trades whose predicted post-fee profit is non-positive.
        if pred_return * 100.0 - fee_pct <= 0.0:
            continue

        target_price = close[i] * (1.0 + pred_return)
        end = min(i + horizon + 1, n)
        if end <= i + 1:
            continue

        exit_price = close[end - 1]
        hit = False
        for k in range(i + 1, end):
            if high[k] >= target_price:
                exit_price = target_price
                hit = True
                break
            if red_start[k]:
                exit_price = close[k]
                break

        gross_return = (exit_price - close[i]) / close[i]
        net_pct = gross_return * 100.0 - fee_pct
        total_net_pct += net_pct
        trades_taken += 1
        if hit:
            hits += 1

    return {
        "trades_taken": int(trades_taken),
        "hits": int(hits),
        "hit_fraction": float(hits / trades_taken) if trades_taken else 0.0,
        "mean_net_profit_pct": float(total_net_pct / trades_taken) if trades_taken else 0.0,
        "total_net_profit_pct": float(total_net_pct),
    }


def _conformal_offset(residuals: np.ndarray, alpha: float) -> float:
    """Empirical α-quantile of CV residuals (y_true − y_pred).

    Adding this offset to predictions makes the *empirical* hit-rate match
    `1 − α` on iid-ish data: hit ≡ y ≥ ŷ + δ ⇔ residual ≥ δ; if δ is the
    α-quantile of residuals then P(residual ≥ δ) = 1 − α."""
    if residuals.size == 0:
        return 0.0
    return float(np.quantile(residuals, alpha))


# ── Walk-forward CV with embargo + purging ─────────────────────────────────────

def _walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    embargo: int,
    n_splits: int,
) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict] = []
    all_residuals: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        # Purge: drop the tail of train whose forward window overlaps test.
        if embargo < len(train_idx):
            train_idx = train_idx[:-embargo]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        val_size = max(50, int(0.10 * len(train_idx)))
        if val_size >= len(train_idx):
            val_size = max(1, len(train_idx) // 5)
        val_idx = train_idx[-val_size:]
        fit_idx = train_idx[:-val_size]
        # Purge again between fit and val (same forward-window leakage risk).
        if embargo < len(fit_idx):
            fit_idx = fit_idx[:-embargo]
        if len(fit_idx) == 0:
            continue

        model = _make_model(alpha)
        model.fit(
            X[fit_idx], y[fit_idx],
            sample_weight=_time_decay_weights(len(fit_idx)),
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        y_pred = model.predict(X[test_idx])
        all_residuals.append(y[test_idx] - y_pred)
        m = _eval_metrics(y[test_idx], y_pred, alpha)
        m["fold"] = fold
        m["train_n"] = int(len(fit_idx))
        m["test_n"] = int(len(test_idx))
        fold_metrics.append(m)
        logger.info(
            "  fold %d a=%.2f train=%d test=%d MAE=%.4f R²=%.3f hit=%.3f pin=%.4f",
            fold, alpha, len(fit_idx), len(test_idx),
            m["mae"], m["r2"], m["hit_rate"], m["pinball"],
        )

    if not fold_metrics:
        return {}

    keys = ("mae", "rmse", "r2", "hit_rate", "pinball")
    summary = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    summary["folds"] = fold_metrics
    residuals = np.concatenate(all_residuals) if all_residuals else np.empty(0)
    summary["calibration_offset"] = _conformal_offset(residuals, alpha)
    summary["calibration_n"] = int(residuals.size)
    return summary


# ── Model factory ──────────────────────────────────────────────────────────────

def _make_model(alpha: float) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        verbosity=0,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    symbols: list[str],
    quantile_alpha: float | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    forward_horizon: int | None = None,
    alphas: tuple[float, ...] | list[float] | None = None,
    horizons: tuple[int, ...] | list[int] | None = None,
    label_form: str = _DEFAULT_LABEL_FORM,
) -> dict:
    """
    Multi-horizon × multi-quantile training pipeline.

    Resolution:
      * `horizons`         — list of horizons (in candles) to train.
      * `forward_horizon`  — legacy single-horizon shortcut: equivalent to
                             horizons=[forward_horizon].
      * Neither            — defaults to _DEFAULT_HORIZONS.

      * `alphas`           — list of quantiles to train per horizon.
      * `quantile_alpha`   — legacy single-quantile shortcut.
      * Neither            — defaults to _DEFAULT_QUANTILES.

      * `label_form`       — "cycle_capped" (default) trains on the upside
                             reachable before the next MACD red-start (so the
                             label matches what the bot can actually realize).
                             "atr" trains on the legacy unconstrained max.
    """
    if horizons is not None:
        horizons_t: tuple[int, ...] = tuple(int(h) for h in horizons)
    elif forward_horizon is not None:
        horizons_t = (int(forward_horizon),)
    else:
        raise ValueError("forward_horizon or horizons must be a non-empty sequence of ints")

    if quantile_alpha is not None and alphas is None:
        alphas_t: tuple[float, ...] = (float(quantile_alpha),)
    elif alphas is not None:
        alphas_t = tuple(float(a) for a in alphas)
    else:
        raise ValueError("quantile_alpha or alphas must be provided")

    if not horizons_t or any(h <= 0 for h in horizons_t):
        raise ValueError("horizons must be a non-empty sequence of positive ints")
    if not alphas_t or any(not 0.0 < a < 1.0 for a in alphas_t):
        raise ValueError("alphas must be a non-empty sequence of values in (0, 1)")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if label_form not in ("cycle_capped", "atr"):
        raise ValueError(f"label_form must be 'cycle_capped' or 'atr', got {label_form!r}")

    symbols_norm = sorted({s.upper() for s in symbols})
    sym_cols = symbol_one_hot_cols(symbols_norm)
    per_symbol = _build_per_symbol(symbols_norm, lookback_days)

    cv_metrics: dict[str, dict] = {}
    bundle_models: dict[int, dict[float, XGBRegressor]] = {}
    calibration: dict[int, dict[float, float]] = {}
    backtest_metrics: dict[str, dict] = {}
    feature_cols: list[str] = []
    total_samples = 0
    fee_pct = float(settings.trading_fee_pct)

    for h in horizons_t:
        logger.info("=== horizon %d candles ===", h)
        combined, h_feature_cols = _build_horizon_dataset(per_symbol, h, sym_cols, label_form)
        if combined.empty:
            logger.warning("No labeled rows for horizon=%d – skipping", h)
            continue
        feature_cols = h_feature_cols  # identical across horizons in practice
        X = combined[feature_cols].values.astype(np.float64)
        y = combined["target"].values.astype(np.float64)
        if len(X) < 200:
            logger.warning("Skip horizon=%d: only %d labeled rows (<200)", h, len(X))
            continue
        total_samples = max(total_samples, int(len(X)))

        val_size = max(200, int(0.10 * len(X)))
        if val_size >= len(X):
            val_size = max(1, len(X) // 5)
        fit_end = len(X) - val_size
        if h < fit_end:
            fit_end -= h
        if fit_end <= 0:
            logger.warning("Skip horizon=%d: not enough rows after embargo", h)
            continue

        # OHLC/MACD slice for the val tail — used by the realized-PnL backtest
        # to score each (h, alpha) by what the bot would have actually earned.
        val_slice = combined.iloc[-val_size:]
        bt_close = val_slice["close"].values.astype(np.float64)
        bt_high = val_slice["high"].values.astype(np.float64)
        bt_atr = val_slice["atr"].values.astype(np.float64)
        bt_macd_hist = val_slice["macd_hist"].values.astype(np.float64)

        per_horizon_models: dict[float, XGBRegressor] = {}
        per_horizon_offsets: dict[float, float] = {}
        for alpha in alphas_t:
            logger.info("  --- alpha=%.2f ---", alpha)
            cv = _walk_forward_cv(X, y, alpha, h, _CV_SPLITS)
            cv_metrics[f"h{h}_a{alpha:.2f}"] = cv

            model = _make_model(alpha)
            model.fit(
                X[:fit_end], y[:fit_end],
                sample_weight=_time_decay_weights(fit_end),
                eval_set=[(X[-val_size:], y[-val_size:])],
                verbose=False,
            )
            per_horizon_models[float(alpha)] = model
            offset = float(cv.get("calibration_offset", 0.0))
            per_horizon_offsets[float(alpha)] = offset

            # Realized-PnL backtest on the val tail using calibrated predictions.
            val_pred_units = model.predict(X[-val_size:]) + offset
            bt = _backtest_realized_pnl(
                pred_units=val_pred_units,
                close=bt_close,
                high=bt_high,
                atr=bt_atr,
                macd_hist=bt_macd_hist,
                horizon=h,
                fee_pct=fee_pct,
            )
            backtest_metrics[f"h{h}_a{alpha:.2f}"] = bt
            logger.info(
                "  h=%d a=%.2f final: best_iter=%s, train_n=%d val_n=%d "
                "conformal_offset=%+.4f | backtest: trades=%d hit=%.3f "
                "mean_net=%.3f%% total_net=%.2f%%",
                h, alpha,
                getattr(model, "best_iteration", "?"),
                fit_end, val_size, offset,
                bt["trades_taken"], bt["hit_fraction"],
                bt["mean_net_profit_pct"], bt["total_net_profit_pct"],
            )

        if per_horizon_models:
            bundle_models[int(h)] = per_horizon_models
            calibration[int(h)] = per_horizon_offsets

    if not bundle_models:
        raise ValueError("No horizons produced a trained model – aborting")

    label_name = (
        "cycle_capped_atr_normalized"
        if label_form == "cycle_capped"
        else "atr_normalized_forward_max"
    )

    bundle = {
        "version":         _BUNDLE_VERSION,
        "models":          bundle_models,
        "feature_cols":    feature_cols,
        "symbol_cols":     sym_cols,
        "alphas":          [float(a) for a in alphas_t],
        "horizons":        sorted(bundle_models.keys()),
        "calibration":     calibration,
        "kline_interval":  settings.kline_interval,
        "label_form":      label_form,
        "trading_fee_pct": fee_pct,
    }
    with open(_model_path(), "wb") as fh:
        pickle.dump(bundle, fh)

    # Metadata mirrors calibration with string keys so it's JSON-safe and
    # inspectable without unpickling the bundle.
    calibration_meta = {
        str(h): {f"{a:.2f}": float(off) for a, off in alpha_map.items()}
        for h, alpha_map in calibration.items()
    }
    metadata = {
        "version":         _BUNDLE_VERSION,
        "alphas":          [float(a) for a in alphas_t],
        "horizons":        sorted(bundle_models.keys()),
        "lookback_days":   lookback_days,
        "label":           label_name,
        "label_form":      label_form,
        "feature_cols":    feature_cols,
        "symbol_cols":     sym_cols,
        "symbols":         symbols_norm,
        "total_samples":   total_samples,
        "kline_interval":  settings.kline_interval,
        "trading_fee_pct": fee_pct,
        "cv_metrics":      cv_metrics,
        "backtest":        backtest_metrics,
        "calibration":     calibration_meta,
    }
    with open(_metadata_path(), "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Regression bundle saved to %s", _model_path())

    with _lock:
        _cache.clear()
        _cache.update({
            "trained":         True,
            "models":          bundle_models,
            "feature_cols":    feature_cols,
            "symbol_cols":     sym_cols,
            "alphas":          [float(a) for a in alphas_t],
            "horizons":        sorted(bundle_models.keys()),
            "calibration":     calibration,
            "lookback_days":   lookback_days,
            "metadata":        metadata,
        })

    primary_horizon = (
        60 if 60 in bundle_models else
        sorted(bundle_models.keys())[len(bundle_models) // 2]
    )
    # Pick the primary alpha by realized PnL at the primary horizon, with
    # tie-break to median alpha and a sanity floor on trades_taken so a
    # quantile that simply refused to trade can't win by default.
    primary_alpha = _pick_primary_alpha(
        backtest_metrics, primary_horizon, alphas_t,
    )
    primary_cv = cv_metrics.get(f"h{primary_horizon}_a{primary_alpha:.2f}", {})
    primary_bt = backtest_metrics.get(f"h{primary_horizon}_a{primary_alpha:.2f}", {})

    summary_metrics = {
        "symbols":         symbols_norm,
        "features":        feature_cols,
        "total_samples":   total_samples,
        "train_samples":   int(total_samples - max(200, int(0.10 * total_samples))),
        "test_samples":    int(max(200, int(0.10 * total_samples))),
        "lookback_days":   lookback_days,
        "horizons":        sorted(bundle_models.keys()),
        "alphas":          [float(a) for a in alphas_t],
        "label":           metadata["label"],
        "label_form":      label_form,
        "kline_interval":  settings.kline_interval,
        "trading_fee_pct": fee_pct,
        "cv_metrics":      cv_metrics,
        "backtest":        backtest_metrics,
        "forward_horizon": int(primary_horizon),
        "quantile_alpha":  float(primary_alpha),
        "mae":             round(float(primary_cv.get("mae", 0.0)), 6),
        "rmse":            round(float(primary_cv.get("rmse", 0.0)), 6),
        "r2":              round(float(primary_cv.get("r2", 0.0)), 4),
        "hit_rate":        round(float(primary_cv.get("hit_rate", 0.0)), 4),
        "pinball":         round(float(primary_cv.get("pinball", 0.0)), 6),
        "realized_hit_fraction":     round(float(primary_bt.get("hit_fraction", 0.0)), 4),
        "realized_mean_profit_pct":  round(float(primary_bt.get("mean_net_profit_pct", 0.0)), 4),
        "realized_total_profit_pct": round(float(primary_bt.get("total_net_profit_pct", 0.0)), 4),
        "realized_trades_taken":     int(primary_bt.get("trades_taken", 0)),
    }
    logger.info(
        "Training complete: %s",
        {k: v for k, v in summary_metrics.items()
         if k not in ("cv_metrics", "features", "backtest")},
    )
    return summary_metrics


def _pick_primary_alpha(
    backtest_metrics: dict[str, dict],
    primary_horizon: int,
    alphas_t: tuple[float, ...],
    min_trades: int = 5,
) -> float:
    """Pick the alpha that maximizes mean net realized profit at the primary
    horizon, requiring at least `min_trades` taken so a quantile that simply
    refused to trade can't win by default. Falls back to the median alpha
    (or 0.5 if trained) when no backtest is available."""
    candidates: list[tuple[float, float, int]] = []
    for alpha in alphas_t:
        bt = backtest_metrics.get(f"h{primary_horizon}_a{alpha:.2f}", {})
        trades = int(bt.get("trades_taken", 0))
        mean_pct = float(bt.get("mean_net_profit_pct", 0.0))
        if trades >= min_trades:
            candidates.append((mean_pct, alpha, trades))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))  # max mean_pct, tie-break low alpha
        winner = candidates[0][1]
        logger.info(
            "primary alpha by realized PnL @ h=%d: %.2f "
            "(mean_net=%.3f%% over %d trades; candidates=%s)",
            primary_horizon, winner, candidates[0][0], candidates[0][2],
            [(round(c[1], 2), round(c[0], 3)) for c in candidates],
        )
        return float(winner)
    fallback = 0.5 if 0.5 in alphas_t else float(alphas_t[len(alphas_t) // 2])
    logger.warning(
        "No alpha had >= %d realized trades @ h=%d; falling back to median alpha=%.2f",
        min_trades, primary_horizon, fallback,
    )
    return fallback


# ── Loading ────────────────────────────────────────────────────────────────────

def load_bundle() -> dict:
    """Return the cached/persisted training bundle."""
    with _lock:
        if _cache.get("trained"):
            return {
                "models":       _cache["models"],
                "feature_cols": _cache["feature_cols"],
                "symbol_cols":  _cache["symbol_cols"],
                "alphas":       _cache["alphas"],
                "horizons":     _cache["horizons"],
                "calibration":  _cache.get("calibration", {}),
                "metadata":     _cache.get("metadata", {}),
            }

    mp = _model_path()
    if not mp.exists():
        raise FileNotFoundError("Regression model has not been trained yet.")

    with open(mp, "rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, dict) or "models" not in bundle or "horizons" not in bundle:
        raise FileNotFoundError(
            "Old regression model format detected – please retrain via "
            "POST /v2/regression/train."
        )

    # Bundle-vs-runtime interval mismatch is silent train/predict bug bait
    # (same horizon snaps to entirely different time windows), so refuse to
    # serve predictions from a bundle that wasn't trained on this interval.
    bundle_interval = bundle.get("kline_interval")
    if bundle_interval and bundle_interval != settings.kline_interval:
        raise FileNotFoundError(
            f"Trained bundle uses kline_interval={bundle_interval!r} but the "
            f"runtime is configured with {settings.kline_interval!r}. "
            "Retrain or set KLINE_INTERVAL to match the bundle."
        )

    metadata: dict = {}
    if _metadata_path().exists():
        with open(_metadata_path()) as fh:
            metadata = json.load(fh)

    # Calibration is a v2.0.1+ addition — older bundles fall back to no offset.
    calibration = bundle.get("calibration") or {}

    with _lock:
        _cache.update({
            "trained":      True,
            "models":       bundle["models"],
            "feature_cols": bundle["feature_cols"],
            "symbol_cols":  bundle["symbol_cols"],
            "alphas":       bundle["alphas"],
            "horizons":     bundle["horizons"],
            "calibration":  calibration,
            "lookback_days": metadata.get("lookback_days"),
            "metadata":     metadata,
        })

    return {
        "models":       bundle["models"],
        "feature_cols": bundle["feature_cols"],
        "symbol_cols":  bundle["symbol_cols"],
        "alphas":       bundle["alphas"],
        "horizons":     bundle["horizons"],
        "calibration":  calibration,
        "metadata":     metadata,
    }


def _metadata_field(key: str):
    if _cache.get("trained"):
        meta = _cache.get("metadata") or {}
        if key in meta:
            return meta[key]
        return _cache.get(key)
    if _metadata_path().exists():
        with open(_metadata_path()) as fh:
            return json.load(fh).get(key)
    return None


def get_quantile_alpha() -> float | None:
    alphas = _metadata_field("alphas")
    if isinstance(alphas, list) and alphas:
        return 0.5 if 0.5 in alphas else float(alphas[len(alphas) // 2])
    return None


def get_primary_horizon() -> int | None:
    hs = _metadata_field("horizons")
    if isinstance(hs, list) and hs:
        if 60 in hs:
            return 60
        return int(hs[len(hs) // 2])
    return None


def get_hit_rate(alpha: float | None = None, horizon: int | None = None) -> float | None:
    """
    Per (alpha, horizon) CV hit-rate. Defaults to primary alpha & horizon.
    """
    cv = _metadata_field("cv_metrics") or {}
    if alpha is None:
        alpha = get_quantile_alpha()
    if horizon is None:
        horizon = get_primary_horizon()
    if alpha is None or horizon is None:
        return None
    fold = cv.get(f"h{int(horizon)}_a{float(alpha):.2f}") or {}
    val = fold.get("hit_rate")
    return float(val) if val is not None else None


def is_trained() -> bool:
    if _cache.get("trained"):
        return True
    return _model_path().exists()


def load_model():
    """Back-compat shim: returns (model, encoder=None, feature_cols)."""
    bundle = load_bundle()
    h = get_primary_horizon() or sorted(bundle["models"].keys())[0]
    a = 0.5 if 0.5 in bundle["models"][h] else next(iter(bundle["models"][h]))
    return bundle["models"][h][a], None, bundle["feature_cols"]
