FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS application
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTO_HR_DB=/app/runtime/auto_hr.sqlite3
WORKDIR /app
COPY backend/requirements.lock backend/requirements.lock
RUN pip install --no-cache-dir -r backend/requirements.lock \
    && useradd --uid 10001 --create-home app \
    && mkdir -p /app/runtime \
    && chown app:app /app/runtime
COPY NLP/ NLP/
COPY ML1/ ML1/
COPY ML2/ ML2/
COPY data/processed/vacancies_with_skills.csv data/processed/vacancies_with_skills.csv
COPY backend/ backend/
COPY --from=frontend /app/frontend/dist frontend/dist/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
