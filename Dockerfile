FROM python:3.15-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/budgetroute/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[api,learned,postgres,reporting]"

COPY configs ./configs
COPY data ./data

RUN addgroup --system budgetroute \
    && adduser --system --ingroup budgetroute --home /home/budgetroute budgetroute \
    && mkdir -p /app/outputs /home/budgetroute/.cache/huggingface \
    && chown -R budgetroute:budgetroute /app /home/budgetroute

USER budgetroute
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"

CMD ["python", "-m", "budgetroute", "serve", "--config", "configs/serving/secure.yaml"]
