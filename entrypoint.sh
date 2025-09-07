#!/usr/bin/env sh
set -e

# Optional: retrain model if requested (useful for dev or scheduled refresh)
if [ "${RETRAIN_MODEL:-0}" = "1" ]; then
  echo "[entrypoint] Retraining model via main.py..."
  python main.py || {
    echo "[entrypoint] Training failed" >&2
    exit 1
  }
fi

# Check artifacts exist
if [ ! -f "spam_classifier.pkl" ] || [ ! -f "vectorizer.pkl" ]; then
  echo "[entrypoint] Missing artifacts. Attempting to train..."
  python main.py || {
    echo "[entrypoint] Could not create artifacts. Exiting." >&2
    exit 1
  }
fi

echo "[entrypoint] Starting Gunicorn..."
exec gunicorn -c gunicorn.conf.py app:app


