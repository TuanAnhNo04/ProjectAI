# Ảnh chính thức của dự án. Docker chưa có trên máy dev nên file này
# **chưa được build thử** — xem ghi chú đầu `docker-compose.yml`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Ghi chú: file này chỉ liệt kê thư viện đã có sẵn trên máy dev (đã kiểm
# ngày 03/10/2026). `psycopg`/`pgvector` **chưa** cài — sẽ thêm khi chuyển sang
# Postgres (Day 18) và cần chủ dự án duyệt trước (AGENTS.md mục 2).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Chỉ danh mục MẪU (dữ liệu giả) để seed khi CSDL rỗng. KHÔNG copy data/real/ hay *.db
# (xem .dockerignore).
COPY data/medicines.dev.csv ./data/medicines.dev.csv

# Mặc định khi deploy (Railway): seed danh mục mẫu nếu bảng thuốc rỗng.
ENV MEDASSIST_SEED_IF_EMPTY=1

# B-04: chạy bằng user không phải root; /data (volume Railway) và /srv thuộc user này.
RUN useradd --system --create-home --uid 10001 app     && mkdir -p /data     && chown -R app:app /srv /data
USER app

EXPOSE 8000
# Trong CONTAINER phải nghe 0.0.0.0 thì nền tảng (Railway) mới chuyển request vào được;
# cổng lấy từ $PORT do Railway cấp (mặc định 8000). Đây KHÔNG phải máy dev: chạy local
# vẫn dùng `uvicorn app.main:app --host 127.0.0.1` (README), còn docker-compose chỉ map
# ra 127.0.0.1. HTTPS do Railway cung cấp ở tên miền *.up.railway.app.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --no-access-log"]
