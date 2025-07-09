FROM python:3.10-slim

# Cài thư viện hệ thống cần thiết cho OpenCV và YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Cài thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Mở cổng
ENV PORT=10000
EXPOSE 10000

# Chạy Flask app
CMD ["python", "app.py"]
