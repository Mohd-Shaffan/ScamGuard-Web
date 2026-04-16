FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Production stage ────────────────────────────────────────────
FROM python:3.12-slim

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:/install/bin:$PATH"
ENV PYTHONPATH="/install/lib/python3.12/site-packages"

WORKDIR /code

# Copy installed dependencies from builder stage
COPY --from=builder /install /install

# Copy project files
COPY --chown=user . /code

# Fix pickling/version mismatch by retraining the model natively
RUN python train_vishing_model.py

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/v1/health')" || exit 1

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Production environment
ENV ENV=production
ENV PORT=7860
ENV ALLOWED_ORIGINS=*

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]