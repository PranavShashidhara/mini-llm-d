FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    httpx==0.27.0 \
    xxhash==3.4.1 \
    prometheus-client==0.20.0 \
    opentelemetry-sdk==1.24.0 \
    opentelemetry-api==1.24.0 \
    opentelemetry-exporter-otlp-proto-grpc==1.24.0

COPY prefix_cache.py metrics.py router.py ./

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')"

ENV PORT=9000
EXPOSE 9000

CMD ["python", "router.py"]
