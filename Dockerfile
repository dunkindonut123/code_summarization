FROM python:3.10-slim

WORKDIR /code

# System deps (optional but helps with some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}