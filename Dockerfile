FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY *.pkl ./

ENV LOG_DIR=/app/logs

# Non-root user for safety
RUN useradd -m appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 9010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9010"]
