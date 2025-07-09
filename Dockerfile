FROM python:3.10-slim

# Cài các gói hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Mở cổng Flask
EXPOSE 10000

# Chạy ứng dụng
CMD ["python", "app.py"]
