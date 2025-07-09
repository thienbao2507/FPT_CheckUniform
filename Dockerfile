FROM python:3.10-slim

# Cài libGL và libglib2.0 cho OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Cài các thư viện Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Mở port (tùy chọn)
EXPOSE 10000

# Chạy Flask
CMD ["python", "app.py"]
