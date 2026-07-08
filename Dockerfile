FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY backend ./backend
COPY configs ./configs
COPY outputs/models/clustomer_segmenter.joblib ./outputs/models/clustomer_segmenter.joblib
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
