FROM python:3.11-slim

WORKDIR /app

COPY backend_api/requirements.txt /app/backend_api/requirements.txt
RUN pip install --no-cache-dir -r /app/backend_api/requirements.txt

COPY . /app

EXPOSE 8766
CMD ["uvicorn", "backend_api.app.main:app", "--host", "127.0.0.1", "--port", "8766"]
