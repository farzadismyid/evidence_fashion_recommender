FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --extra cpu
COPY src ./src
COPY configs ./configs
COPY data/kb ./data/kb

ENTRYPOINT ["uv", "run", "efr"]
CMD ["--config", "configs/default.yaml", "doctor"]
