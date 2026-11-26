FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/
COPY run_tests.ps1 ./run_tests.ps1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.api:app"]
