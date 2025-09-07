# Multi-stage build: training stage then slim production image

############################
# Stage 1: builder/training #
############################
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building/scikit-learn, numpy etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirement files and install (cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Optionally allow training dataset paths via build args
ARG TRAIN_DATASET_PRIMARY=spam.csv
ARG TRAIN_DATASET_SECONDARY=Dataset_5971.csv
ENV TRAIN_DATASET_PRIMARY=${TRAIN_DATASET_PRIMARY}
ENV TRAIN_DATASET_SECONDARY=${TRAIN_DATASET_SECONDARY}

# Train the model to produce artifacts
RUN python main.py


############################
# Stage 2: production image #
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=5000

WORKDIR /app

# Add non-root user
RUN addgroup --system app && adduser --system --ingroup app app

# Install only runtime deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code (only necessary files)
COPY --from=builder /app/app.py ./
COPY --from=builder /app/data_processing.py ./
COPY --from=builder /app/model_training.py ./
COPY --from=builder /app/main.py ./
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/static ./static
COPY --from=builder /app/gunicorn.conf.py ./
COPY --from=builder /app/entrypoint.sh ./

# Copy trained artifacts from builder
COPY --from=builder /app/spam_classifier.pkl ./
COPY --from=builder /app/vectorizer.pkl ./

# Permissions
RUN chown -R app:app /app \
    && chmod +x /app/entrypoint.sh

USER app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request,sys; \
  import os; url=f'http://127.0.0.1:{os.getenv("APP_PORT","5000")}/healthz'; \
  urllib.request.urlopen(url).read() and sys.exit(0)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]


