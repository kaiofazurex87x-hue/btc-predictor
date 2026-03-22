FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models/predictor models/tiered

EXPOSE 8080

CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080"]
