FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "bankpulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
