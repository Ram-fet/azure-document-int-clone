FROM python:3.11-slim

# Use HTTPS Debian mirrors
RUN echo "deb https://deb.debian.org/debian trixie main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://deb.debian.org/debian-security trixie-security main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://deb.debian.org/debian trixie-updates main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        libgl1 \
        zbar-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
