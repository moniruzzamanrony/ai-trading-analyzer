"""
Multi-symbol, multi-horizon, multi-quantile XGBoost regression trainer.

Pipeline:
  1. Fetch lookback_days of 15m candles per symbol from Binance.
  2. Compute past-only features (15m + resampled 1h/4h + microstructure +
     cyclic) once per symbol.
  3. For each training horizon H in `horizons`, label every candle with
        target = (max(high[i+1 : i+1+H]) - close[i]) / atr[i]
     (ATR-normalized so the same target value means the same statistical
     "size of move" across volatility regimes / symbols).
  4. Append per-symbol one-hot columns (replaces ordinal LabelEncoder).
  5. Time-ordered concat across symbols → walk-forward TimeSeriesSplit
     (5 folds) with embargo = horizon between train/test and fit/val to
     prevent label-window leakage.
  6. For each (horizon, alpha) pair train an XGBRegressor with quantile
     loss + early stopping on a held-out tail. Persist the nested bundle
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

_DEFAULT_LOOKBACK_DAYS = 365
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

def _create_labels_atr(df: pd.DataFrame, horizon: int) -> pd.Series:
    """target[i] = (max(high[i+1 : i+1+horizon]) - close[i]) / atr[i]"""
    high = df["high"]
    # Rolling forward-max via reverse trick (vectorized).
    fwd_max_inclusive = high[::-1].rolling(horizon, min_periods=horizon).max()[::-1]
    fwd_max = fwd_max_inclusive.shift(-1)  # max over [i+1 .. i+horizon]
    target = (fwd_max - df["close"]) / df["atr"]
    target = target.where((df["atr"] > 0) & (df["close"] > 0))
    return target


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
) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    for symbol, df in per_symbol.items():
        sub = df.copy()
        sub["target"] = _create_labels_atr(sub, horizon)
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

    symbols_norm = sorted({s.upper() for s in symbols})
    sym_cols = symbol_one_hot_cols(symbols_norm)
    per_symbol = _build_per_symbol(symbols_norm, lookback_days)

    cv_metrics: dict[str, dict] = {}
    bundle_models: dict[int, dict[float, XGBRegressor]] = {}
    calibration: dict[int, dict[float, float]] = {}
    feature_cols: list[str] = []
    total_samples = 0

    for h in horizons_t:
        logger.info("=== horizon %d candles ===", h)
        combined, h_feature_cols = _build_horizon_dataset(per_symbol, h, sym_cols)
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
            logger.info(
                "  h=%d a=%.2f final: best_iter=%s, train_n=%d val_n=%d "
                "conformal_offset=%+.4f (from %d CV residuals)",
                h, alpha,
                getattr(model, "best_iteration", "?"),
                fit_end, val_size, offset,
                int(cv.get("calibration_n", 0)),
            )

        if per_horizon_models:
            bundle_models[int(h)] = per_horizon_models
            calibration[int(h)] = per_horizon_offsets

    if not bundle_models:
        raise ValueError("No horizons produced a trained model – aborting")

    bundle = {
        "models":       bundle_models,
        "feature_cols": feature_cols,
        "symbol_cols":  sym_cols,
        "alphas":       [float(a) for a in alphas_t],
        "horizons":     sorted(bundle_models.keys()),
        "calibration":  calibration,
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
        "alphas":          [float(a) for a in alphas_t],
        "horizons":        sorted(bundle_models.keys()),
        "lookback_days":   lookback_days,
        "label":           "atr_normalized_forward_max",
        "feature_cols":    feature_cols,
        "symbol_cols":     sym_cols,
        "symbols":         symbols_norm,
        "total_samples":   total_samples,
        "cv_metrics":      cv_metrics,
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

    primary_alpha = 0.5 if 0.5 in alphas_t else float(alphas_t[len(alphas_t) // 2])
    primary_horizon = (
        60 if 60 in bundle_models else
        sorted(bundle_models.keys())[len(bundle_models) // 2]
    )
    primary_cv = cv_metrics.get(f"h{primary_horizon}_a{primary_alpha:.2f}", {})

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
        "cv_metrics":      cv_metrics,
        "forward_horizon": int(primary_horizon),
        "quantile_alpha":  float(primary_alpha),
        "mae":             round(float(primary_cv.get("mae", 0.0)), 6),
        "rmse":            round(float(primary_cv.get("rmse", 0.0)), 6),
        "r2":              round(float(primary_cv.get("r2", 0.0)), 4),
        "hit_rate":        round(float(primary_cv.get("hit_rate", 0.0)), 4),
        "pinball":         round(float(primary_cv.get("pinball", 0.0)), 6),
    }
    logger.info(
        "Training complete: %s",
        {k: v for k, v in summary_metrics.items() if k not in ("cv_metrics", "features")},
    )
    return summary_metrics


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
            "POST /regression/v2/train."
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
