# Trading AI Analyzer API

XGBoost **quantile regression** API for crypto take-profit prediction. Train a multi-symbol max-return model on Binance candles and get a recommended sell price for any BUY entry, with a built-in historical hit-rate so you know how reliable the prediction is.

## Features

- **Quantile regression for take-profit** — predicts the *max forward return* you can realistically expect after a BUY, tunable from conservative to aggressive
- **13-feature pipeline** — EMA diff/slope, RSI-14, MACD histogram, ATR ratio, Bollinger width/position, volume ratio, short-term returns, candle range/body, symbol encoding
- **Multi-symbol training** — one model that learns across BTC, ETH, BNB, SOL, etc. with a learned per-symbol embedding
- **80/20 time-based split** — no shuffle, no future leakage into features
- **Hit-rate metric** — every trained model reports the fraction of historical cycles where actual price reached the prediction
- **Trading-fee aware** — predicted profit is reported net of a configurable round-trip fee
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
uvicorn app.main:app --host 0.0.0.0 --port 9020 --reload
```

The API will be available at `http://localhost:9020`.
Interactive docs: `http://localhost:9020/docs`

## Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env as needed

docker compose up --build
```

The container exposes port **9010** (mapped 1:1 from host to container in `docker-compose.yml`).

## Environment Variables

Copy `.env.example` to `.env` and set the values below.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_DIR` | `logs` | Directory for log files |
| `BINANCE_BASE_URL` | `https://api.binance.com` | Binance REST base URL |
| `KLINE_INTERVAL` | `15m` | Binance candle timeframe (`1m`, `5m`, `15m`, `1h`, …) |
| `KLINE_LIMIT` | `300` | Candles fetched per live prediction request |
| `MODELS_DIR` | `.` | Directory where trained model `.pkl` and metadata `.json` are saved |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/regression/train` | Train a new quantile regression model |
| `POST` | `/regression/predict` | Predict sell price for a BUY entry |
| `GET`  | `/regression/status` | Check whether a trained model is available |

### Train

**`POST /regression/train`** — request body:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "lookback_days": 60,
  "forward_horizon": 60,
  "quantile_alpha": 0.5
}
```

| Field | Default | Description |
|---|---|---|
| `symbols` | `["BTCUSDT","ETHUSDT","BNBUSDT"]` | Trading pairs to include in training |
| `lookback_days` | `60` | Days of historical candles to fetch per symbol (1–730) |
| `forward_horizon` | `60` | Number of future candles scanned for the max-return label (1–500) |
| `quantile_alpha` | `0.5` | Quantile target. `0.5` ≈ median (~50% hit rate). `0.3` ≈ conservative (~70% hit rate). `0.7` ≈ aggressive (~30% hit rate). |

Response:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "features": ["ema_diff", "ema9_slope", "rsi_14", "macd_hist", "..."],
  "total_samples": 17280,
  "train_samples": 13824,
  "test_samples": 3456,
  "lookback_days": 60,
  "forward_horizon": 60,
  "quantile_alpha": 0.5,
  "mae": 0.0064,
  "rmse": 0.0089,
  "r2": 0.12,
  "hit_rate": 0.62
}
```

`hit_rate` = fraction of test rows where the actual max forward return was ≥ the predicted return. Tune `quantile_alpha` lower if you want this number higher.

### Predict

**`POST /regression/predict`** — request body:
```json
{
  "symbol": "BTCUSDT",
  "buy_price": 65000.0
}
```

| Field | Required | Description |
|---|---|---|
| `symbol` | yes | Trading pair (e.g. `BTCUSDT`) |
| `buy_price` | yes | Entry price, must be `> 0` |

Response:
```json
{
  "sell_price": 66495.0,
  "profitPercentage": 2.10,
  "accuracy": 0.62
}
```

| Field | Description |
|---|---|
| `sell_price` | Recommended take-profit price = `buy_price × (1 + predicted_return)` |
| `profitPercentage` | Net profit % after subtracting the round-trip trading fee (currently `0.2%`). Can be negative if predicted return is below the fee. |
| `accuracy` | Historical hit-rate from training metadata (0–1). `null` if metadata is missing. |

> Call `POST /regression/train` once before `POST /regression/predict`. The model is held in memory after training and auto-loaded from disk on next API start if `regression_xgb_model.pkl`, `regression_label_encoder.pkl`, and `regression_metadata.json` exist.

### Status

**`GET /regression/status`** — response:
```json
{
  "trained": true,
  "quantile_alpha": 0.3,
  "hit_rate": 0.6207
}
```

## How it works

```
POST /regression/train
  └─ for each symbol:
       fetch lookback_days of candles  →  compute 13 features (past-only)
       label each candle with target_return = max forward return over
       the next forward_horizon candles
  └─ merge symbols, sort by time, 80/20 split (no shuffle)
  └─ train XGBRegressor with objective="reg:quantileerror"
  └─ save regression_xgb_model.pkl + regression_label_encoder.pkl
                                   + regression_metadata.json

POST /regression/predict
  └─ fetch latest candles for symbol
  └─ build features for the most recent row
  └─ predict max forward return (clamped to ≥ 0)
  └─ return sell_price = buy_price × (1 + predicted_return),
            profitPercentage net of trading fee,
            accuracy = stored hit_rate
```

## Project Structure

```
ai-analyzer-api/
├── app/
│   ├── api/
│   │   └── regression_routes.py     # /regression/train, /predict, /status
│   ├── core/
│   │   ├── config.py                # Settings (pydantic-settings + .env)
│   │   └── logging_config.py
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   ├── services/
│   │   ├── data_fetcher.py          # Binance REST kline fetcher (paginated)
│   │   ├── regression_features.py   # 13-feature pipeline + MACD-cycle helpers
│   │   ├── regression_trainer.py    # Quantile XGBRegressor training + persistence
│   │   └── regression_predictor.py  # Live inference + sell-price calculation
│   └── main.py                      # FastAPI app
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── deploy.sh                        # One-command remote deploy via SSH
└── .env.example
```

## Deployment

`deploy.sh` packages the source, uploads it to a remote server via SCP, and restarts Docker Compose:

```bash
bash deploy.sh
```

Edit the `REMOTE` and `REMOTE_DIR` variables at the top of the script to match your server.

## Tech Stack

| Layer | Library |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ML | XGBoost (quantile regression), scikit-learn |
| Technical indicators | `ta` (pandas-based) |
| Market data | Binance REST |
| Data | pandas, numpy |
| Config | pydantic-settings |
| HTTP client | httpx |