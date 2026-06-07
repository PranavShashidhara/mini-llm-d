FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip curl git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Python deps — install vLLM first (large, benefits from layer caching)
RUN pip install --no-cache-dir \
    vllm==0.4.3 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    xxhash==3.4.1 \
    pyyaml==6.0.1 \
    opentelemetry-sdk==1.24.0 \
    opentelemetry-api==1.24.0 \
    opentelemetry-exporter-otlp-proto-grpc==1.24.0

COPY config.yaml .
COPY server.py .

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PORT=8000
EXPOSE 8000

CMD ["python", "server.py"]
