FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/

# Copy trained model artifacts (pkl bundle + metadata).
# Build will fail if either is missing — deploy.sh enforces this pre-flight.
COPY regression_xgb_model.pkl regression_metadata.json ./

ENV LOG_DIR=/app/logs

# Non-root user for safety
RUN useradd -m appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 9030

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9030"]
