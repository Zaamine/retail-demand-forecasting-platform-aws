# Lightweight CPU-only runtime. Build for linux/amd64, not Mac ARM64.
FROM python:3.12.9-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
RUN pip install --no-cache-dir \
    xgboost==3.4.1 numpy==2.5.3 pandas==3.0.6 \
    pyarrow==22.0.0 scikit-learn==1.7.2
WORKDIR /opt/ml/processing
CMD ["python", "--version"]
