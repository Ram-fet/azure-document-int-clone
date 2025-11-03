# ================================
#  Azure Document Intelligence Clone
#  FastAPI + React (Vite) + OCR + AI
# ================================

########## 1. Build Frontend ##########
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY frontend/ .
RUN npm run build

########## 2. Build Backend ##########
FROM python:3.11-slim AS backend
ENV DEBIAN_FRONTEND=noninteractive

# --- System dependencies for OCR ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr-all \
    poppler-utils \
    libgl1 \
    zbar-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .

# --- Install Python & AI deps ---
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       torch torchvision torchaudio \
       transformers==4.44.0 sentencepiece==0.1.99 accelerate==0.33.0

COPY backend/ .

########## 3. Copy Frontend build into Backend ##########
COPY --from=frontend-build /app/frontend/dist /app/static
ENV FRONTEND_DIR=/app/static

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
