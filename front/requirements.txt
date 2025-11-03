# Web framework
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9

# OCR / AI stack
torch==2.3.1                 # CPU build; if you need CUDA, pin to the matching wheel
transformers==4.42.4
pytesseract==0.3.10
Pillow==10.4.0
opencv-python-headless==4.10.0.84

# PDF rendering
pdf2image==1.17.0

# Data validation
pydantic==2.8.2

# Barcode detection
pyzbar==0.1.9

# Translation dependencies (needed for Helsinki-NLP / MarianMT)
sentencepiece==0.2.0
sacremoses==0.1.1
