# Sử dụng Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy tất cả file vào container
COPY . .

# Cài đặt thư viện
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Cấu hình cổng Flask
ENV PORT=10000
EXPOSE 10000

# Chạy app Flask
CMD ["python", "app.py"]
