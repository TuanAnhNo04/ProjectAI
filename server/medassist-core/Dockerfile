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

EXPOSE 8000
# Không chạy `--host 0.0.0.0` ngoài docker network; trong compose đã bind
# 127.0.0.1 nên API chỉ nghe localhost.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]