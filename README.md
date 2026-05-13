# Trading AI Analyzer API — v2.0.0

XGBoost **multi-horizon, multi-quantile regression** API for crypto take-profit prediction. Trains a bundle of models per (horizon × quantile) on Binance candles and returns recommended sell prices at conservative / median / aggressive risk levels, each annotated with its walk-forward hit rate.

## What's new in v2.0.0

- **Multi-horizon + multi-quantile bundle** — one training run produces a model per `(horizon, alpha)` pair (default `horizons = [16, 60, 240]` candles × `alphas = [0.3, 0.5, 0.7]`)
- **ATR-normalized labels** — target is `(max_high − close) / ATR` instead of raw % return, so the model learns vol-adjusted reach
- **~40-feature pipeline** — adds multi-timeframe features (1h / 4h), cyclical time encoding (hour-of-day, day-of-week), volatility-of-returns, ATR percentile rank, distance from N-bar high/low, microstructure (taker-buy ratio, log-trades)
- **One-hot symbol encoding** — replaces the label encoder; bundle is self-describing, no separate `label_encoder.pkl`
- **Walk-forward CV** — replaces single 80/20 split; reports per-fold + aggregate `mae / rmse / r² / hit_rate / pinball` for every `(horizon, alpha)`
- **Pinball loss reported** — the actual training objective is now in the metrics
- **API moved to `/v2/regression/*`**
- **`forward_candles` on predict** — caller picks a horizon; request is snapped to the nearest trained horizon
- **`quantiles` block on predict response** — conservative / median / aggressive sell prices in a single call
- **Default lookback bumped to 365 days**
- **Docker image bakes in trained artifacts** — `regression_xgb_model.pkl` + `regression_metadata.json` are copied into the image at build time
- **`deploy.sh` removed from the repo** — kept locally and git-ignored (contained an SSH password)

## Features

- **Quantile regression for take-profit** — predicts the *max forward reach in ATR units* you can realistically expect after a BUY, tunable from conservative to aggressive
- **Multi-horizon predictions** — pick a forward window per request (snapped to nearest trained horizon)
- **Multi-symbol training** — one bundle that learns across BTC, ETH, BNB, SOL, etc. with per-symbol one-hot features
- **Walk-forward evaluation** — every trained model reports per-fold + aggregate metrics, including the historical hit-rate
- **Trading-fee aware** — predicted profit is reported net of a configurable round-trip fee (default `0.2%`)
- **Live inference** — predict from the latest Binance candles in a single request

## Requirements

- Python 3.12+
- OR Docker + Docker Compose

## Quick Start (local)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd ai-analyzer-api

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env as needed (see Environment Variables section)

# 5. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 9030 --reload
```

The API will be available at `http://localhost:9030`.
Interactive docs: `http://localhost:9030/docs`

## Quick Start (Docker)

The Dockerfile copies trained artifacts (`regression_xgb_model.pkl` + `regression_metadata.json`) into the image, so build **will fail if those files are missing**. Train locally first (or copy a trained pair into the project root), then:

```bash
cp .env.example .env
# Edit .env as needed

docker compose up --build
```

The container exposes port **9030** (mapped 1:1 in `docker-compose.yml`).

## Environment Variables

Copy `.env.example` to `.env` and set the values below.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_DIR` | `logs` | Directory for log files |
| `BINANCE_BASE_URL` | `https://api.binance.com` | Binance REST base URL |
| `KLINE_INTERVAL` | `15m` | Binance candle timeframe (`1m`, `5m`, `15m`, `1h`, …) |
| `KLINE_LIMIT` | `300` | Candles fetched per live prediction request |
| `MODELS_DIR` | `.` | Directory where the model bundle `.pkl` and metadata `.json` are saved |

## API Endpoints

> All endpoints are versioned under `/v2/regression`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v2/regression/train` | Train a new multi-horizon × multi-quantile bundle |
| `POST` | `/v2/regression/predict` | Predict sell prices for a BUY entry (returns conservative / median / aggressive) |
| `GET`  | `/v2/regression/status` | Check whether a trained bundle is available |

### Train

**`POST /v2/regression/train`** — request body:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "lookback_days": 365,
  "horizons": [16, 60, 240],
  "alphas": [0.3, 0.5, 0.7]
}
```

| Field | Default | Description |
|---|---|---|
| `symbols` | `["BTCUSDT","ETHUSDT","BNBUSDT"]` | Trading pairs to include in training |
| `lookback_days` | `365` | Days of historical candles to fetch per symbol (1–730) |
| `horizons` | `[16, 60, 240]` | Forward horizons in candles. Default ≈ 4h / 15h / 60h on 15m candles. One model is produced per `(horizon, alpha)` pair |
| `alphas` | `[0.3, 0.5, 0.7]` | Quantiles trained per horizon. Lower α = more conservative / higher hit-rate |
| `forward_horizon` | `null` | **Legacy** single-horizon shortcut. Equivalent to `horizons=[forward_horizon]` |
| `quantile_alpha` | `null` | **Legacy** single-quantile shortcut. Equivalent to `alphas=[quantile_alpha]` |

Response:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "features": ["ema_diff", "ema9_slope", "rsi_14", "macd_hist", "..."],
  "total_samples": 104949,
  "train_samples": 83959,
  "test_samples": 20990,
  "lookback_days": 365,
  "forward_horizon": 60,
  "horizons": [16, 60, 240],
  "alphas": [0.3, 0.5, 0.7],
  "label": "atr_normalized_forward_max",
  "quantile_alpha": 0.5,
  "mae": 2.87,
  "rmse": 4.24,
  "r2": -0.03,
  "hit_rate": 0.47,
  "pinball": 1.41,
  "cv_metrics": {
    "h60_a0.30": { "mae": ..., "hit_rate": ..., "pinball": ..., "folds": [ ... ] },
    "h60_a0.50": { "...": "..." },
    "h60_a0.70": { "...": "..." }
  }
}
```

Top-level `mae / rmse / r² / hit_rate / pinball` are reported for the **primary** `(horizon, alpha)` (median horizon, α = 0.5 if trained). The full per-fold breakdown for every pair is under `cv_metrics`.

`hit_rate` = fraction of validation rows where the actual ATR-normalized max forward reach was ≥ the predicted value. Lower the alpha to push this higher.

### Predict

**`POST /v2/regression/predict`** — request body:
```json
{
  "symbol": "BTCUSDT",
  "buy_price": 65000.0,
  "forward_candles": 60
}
```

| Field | Required | Description |
|---|---|---|
| `symbol` | yes | Trading pair (e.g. `BTCUSDT`) |
| `buy_price` | yes | Entry price, must be `> 0` |
| `forward_candles` | no | Forward window in candles. Snapped to the nearest **trained** horizon. If omitted, uses the bundle's primary horizon |

Response:
```json
{
  "sell_price": 66495.0,
  "profitPercentage": 2.10,
  "accuracy": 0.62,
  "forward_candles_requested": 60,
  "forward_candles_used": 60,
  "available_horizons": [16, 60, 240],
  "quantiles": {
    "conservative": { "alpha": 0.3, "sell_price": 65780.0, "profitPercentage": 1.00, "accuracy": 0.66 },
    "median":       { "alpha": 0.5, "sell_price": 66495.0, "profitPercentage": 2.10, "accuracy": 0.48 },
    "aggressive":   { "alpha": 0.7, "sell_price": 67510.0, "profitPercentage": 3.66, "accuracy": 0.31 }
  }
}
```

| Field | Description |
|---|---|
| `sell_price` | Recommended take-profit price at the **primary** quantile (median) |
| `profitPercentage` | Net profit % at the primary quantile, after subtracting the round-trip trading fee (default `0.2%`). Can be negative if predicted return is below the fee |
| `accuracy` | Walk-forward hit-rate (0–1) for the primary `(horizon, alpha)` |
| `forward_candles_requested` | The `forward_candles` value the caller asked for |
| `forward_candles_used` | The trained horizon actually used (nearest snap) |
| `available_horizons` | Horizons present in the loaded bundle |
| `quantiles` | Per-quantile take-profit suggestions. `conservative` / `median` / `aggressive` each include `alpha`, `sell_price`, `profitPercentage`, and that quantile's own `accuracy` |

> Call `POST /v2/regression/train` once before `POST /v2/regression/predict`. The bundle is held in memory after training and auto-loaded from disk on next API start if `regression_xgb_model.pkl` + `regression_metadata.json` exist.

### Status

**`GET /v2/regression/status`** — response:
```json
{
  "trained": true,
  "quantile_alpha": 0.5,
  "hit_rate": 0.4733
}
```

## How it works

```
POST /v2/regression/train
  └─ for each symbol:
       fetch lookback_days of 15m candles  →  compute ~40 features (past-only):
         - 15m: EMA diff/slope, RSI-14, MACD hist, ATR ratio, BB width/position,
                volume ratio, short-term returns, vol-of-returns, ATR pct-rank,
                distance from N-bar high/low, range/body norm,
                taker-buy ratio (+ MA), log-trades, hod/dow sin/cos
         - 1h aggregates: EMA diff, RSI-14, ATR ratio, return
         - 4h aggregates: EMA diff, RSI-14, ATR ratio, return
         - one-hot symbol columns (is_BTCUSDT, is_ETHUSDT, …)
  └─ for each horizon h in horizons:
       label[i] = (max(high[i+1 : i+1+h]) - close[i]) / atr[i]
       merge symbols, sort by time
       for each alpha in alphas:
         walk-forward CV (5 folds)  →  per-fold + aggregate mae/rmse/r²/hit/pinball
         final fit on full series (with embargo) using objective="reg:quantileerror"
  └─ save bundle: regression_xgb_model.pkl + regression_metadata.json

POST /v2/regression/predict
  └─ fetch latest candles for symbol
  └─ build features for the most recent row
  └─ snap forward_candles → nearest trained horizon
  └─ run all alphas at that horizon (clamped to ≥ 0)
  └─ return primary sell_price + per-quantile breakdown + accuracy
```

## Project Structure

```
ai-analyzer-api/
├── app/
│   ├── api/
│   │   └── regression_routes.py     # /v2/regression/train, /predict, /status
│   ├── core/
│   │   ├── config.py                # Settings (pydantic-settings + .env)
│   │   └── logging_config.py
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   ├── services/
│   │   ├── data_fetcher.py          # Binance REST kline fetcher (paginated)
│   │   ├── regression_features.py   # ~40-feature pipeline (15m + 1h + 4h)
│   │   ├── regression_trainer.py    # Multi-horizon × multi-quantile training + walk-forward CV
│   │   └── regression_predictor.py  # Live inference + per-quantile sell prices
│   └── main.py                      # FastAPI app (auto-loads bundle on startup)
├── requirements.txt
├── Dockerfile                       # Bakes model artifacts into the image
├── docker-compose.yml               # Port 9030
├── regression_xgb_model.pkl         # Trained bundle (gitignored)
├── regression_metadata.json         # Bundle metadata (features, horizons, alphas, cv_metrics)
└── .env.example
```

## Deployment

`deploy.sh` is **not tracked in git** (it contains an SSH password). Keep your local copy outside of commits. The deploy flow is:

1. Train the bundle locally — produces `regression_xgb_model.pkl` + `regression_metadata.json` in the project root.
2. Run `bash deploy.sh` (your local copy) — packages the source + artifacts, ships them to the remote host, and rebuilds the container.

If you need a starter `deploy.sh`, request one — it should never be committed.

## Tech Stack

| Layer | Library |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ML | XGBoost (quantile regression, walk-forward CV), scikit-learn |
| Technical indicators | `ta` (pandas-based) |
| Market data | Binance REST |
| Data | pandas, numpy |
| Config | pydantic-settings |
| HTTP client | httpx |
