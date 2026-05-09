FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# Install system deps required for OpenCV / TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only requirements first for better caching
COPY requirements-api.txt /app/requirements-api.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements-api.txt

# copy the rest of the code
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
