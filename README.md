# Trading AI Analyzer API

XGBoost regression API for crypto take-profit prediction. Train a multi-symbol return model on Binance 5-min candles and get a predicted TP price for any BUY entry.

## Features

- **Regression take-profit** — train on historical EMA-crossover BUY events and predict expected return to next EMA cross-down
- **13-feature pipeline** — EMA diff/slope, RSI-7, MACD histogram, ATR ratio, Bollinger width/position, volume ratio, price-action features, symbol encoding
- **80/20 time-based split** — no shuffle, no future leakage into features
- **Live inference** — predict TP from the latest candle via Binance REST

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

The API runs on port **9010** when started via Docker Compose (mapped from container port 9020).

## Environment Variables

Copy `.env.example` to `.env` and set the values below.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `KLINE_INTERVAL` | `5m` | Binance candle timeframe |
| `KLINE_LIMIT` | `300` | Candles fetched for live prediction warm-up |
| `MODELS_DIR` | `.` | Directory where trained model `.pkl` files are saved |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/regression/train` | Train a new regression model on historical data |
| `POST` | `/regression/predict` | Predict take-profit price for a BUY entry |
| `GET` | `/regression/status` | Check whether a trained model is available |

### Train

**`POST /regression/train`** — request body:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "min_return_threshold": 0.01
}
```

Response:
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "features": ["ema_diff", "ema9_slope", "rsi_7", "..."],
  "total_samples": 240,
  "train_samples": 192,
  "test_samples": 48,
  "mae": 0.0031,
  "rmse": 0.0048,
  "r2": 0.61
}
```

### Predict

**`POST /regression/predict`** — request body:
```json
{
  "symbol": "BTCUSDT",
  "buy_price": 65000.0,
  "min_return_threshold": 0.01
}
```

Pass `buy_price: 0` to use the latest close price as the entry.

Response:
```json
{
  "symbol": "BTCUSDT",
  "current_price": 65420.10,
  "buy_price": 65000.0,
  "predicted_return": 0.023,
  "take_profit": 66495.0,
  "signal_taken": true,
  "min_return_threshold": 0.01
}
```

`signal_taken` is `true` when `predicted_return > min_return_threshold`.

> Call `POST /regression/train` before `POST /regression/predict`. The model is held in memory and must be retrained after a restart (or reloaded automatically from disk if `.pkl` files exist).

### Status

**`GET /regression/status`** — response:
```json
{ "trained": true }
```

## Workflow

```
POST /regression/train   →  fetches 1000 candles/symbol, engineers features,
                            creates EMA-crossover labels, trains XGBRegressor,
                            saves regression_xgb_model.pkl + regression_label_encoder.pkl

POST /regression/predict →  fetches latest candles, builds features for most
                            recent row, predicts return, returns TP price
```

## Project Structure

```
ai-analyzer-api/
├── app/
│   ├── api/
│   │   └── regression_routes.py   # /regression/train, /predict, /status
│   ├── core/
│   │   ├── config.py              # Settings (pydantic-settings + .env)
│   │   └── logging_config.py
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── data_fetcher.py        # Binance REST kline fetcher
│   │   ├── regression_features.py # 13-feature pipeline + EMA crossover detection
│   │   ├── regression_trainer.py  # XGBRegressor training + persistence
│   │   └── regression_predictor.py# Live inference + TP calculation
│   └── main.py                    # FastAPI app
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── deploy.sh                      # One-command remote deploy via SSH
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
| ML | XGBoost, scikit-learn |
| Technical indicators | `ta` (pandas-based) |
| Market data | Binance REST |
| Data | pandas, numpy |
| Config | pydantic-settings |
| HTTP client | httpx |
