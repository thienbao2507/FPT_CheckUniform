FROM python:3.10-slim

# Cài thư viện cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy code và requirements
COPY . .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Chạy app
CMD ["python", "app.py"]
