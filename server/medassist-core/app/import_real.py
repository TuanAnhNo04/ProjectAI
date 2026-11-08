"""Nạp danh mục thuốc THẬT vào CSDL của server — công cụ chạy trên máy CỦA BẠN.

    python -m app.import_real --csv "D:\\du-lieu\\thuoc.csv" --source "Cục Quản lý Dược, 2026-10" --apply

## Nguyên tắc (AGENTS.md mục 1, 3)

- **Chỉ in số liệu, không bao giờ in nội dung dòng** (tên thuốc, công dụng...). Kết quả
  có thể bị dán vào chat/log; dữ liệu thật có thể nhạy cảm hoặc có bản quyền.
- **Mặc định chạy thử** (không ghi gì). Muốn ghi phải có `--apply`.
- **Đánh dấu "nguồn lâm sàng" (`--authoritative`) phải kèm `--reviewed-by`**: ai đã kiểm
  duyệt nguồn này. Tên người được ghi vào nhãn nguồn của từng bản ghi.
- Trước khi ghi, tự **sao lưu** file CSDL (`.bak-<thời gian>`).
- Nguồn nào cũng phải có `--source` (ghi vào từng bản ghi để trích dẫn đúng nguồn).
- `--replace-source`: nạp lại cùng một nguồn mà không bị nhân đôi.
- `--drop-dev-sample`: gỡ 10 thuốc mẫu `dev-sample:` để dữ liệu thật không lẫn với dữ liệu giả.

⚠️ Đánh dấu `is_authoritative` KHÔNG tự mở các câu hỏi liều dùng/tương tác: các câu đó vẫn
bị chặn bởi bộ lọc phạm vi (`app/agent.py`) cho đến khi có dược sĩ duyệt quy tắc.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from app.ingest import (
    DEFAULT_COLUMNS,
    IngestError,
    IngestReport,
    load_csv_mapped,
    to_rows,
)
from app.sqlite_repository import SQLiteRepository

#: Tiền tố nhãn của dữ liệu mẫu (xem `ingest.load_csv`).
DEV_SOURCE_PREFIX = "dev-sample:"


def _parse_col_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, col = item.partition("=")
        key, col = key.strip(), col.strip()
        if not sep or key not in DEFAULT_COLUMNS or not col:
            raise IngestError(
                f"--col sai: '{item}'. Dạng đúng: khoá=Tên cột; khoá thuộc "
                + ", ".join(DEFAULT_COLUMNS)
            )
        out[key] = col
    return out


def _print_report(report: IngestReport, out) -> None:
    print("--- Báo cáo (chỉ số liệu) ---", file=out)
    print(f"Cột có trong file : {', '.join(report.columns_found) or '(không có)'}", file=out)
    if report.missing_columns:
        print(f"Cột không thấy    : {', '.join(report.missing_columns)} (các trường này sẽ để trống)", file=out)
    print(f"Số dòng đọc       : {report.rows_read}", file=out)
    print(f"Sẽ nạp            : {report.imported}", file=out)
    print(f"Bỏ (không có tên) : {report.skipped_no_name}", file=out)
    print(f"Bỏ (trùng)        : {report.duplicates}", file=out)
    print(f"Ô quá dài bị cắt  : {report.truncated_fields}", file=out)


def run(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="python -m app.import_real", description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", required=True, help="Đường dẫn file CSV (UTF-8)")
    p.add_argument("--source", required=True, help='Nhãn nguồn, vd "Cục Quản lý Dược, 2026-10"')
    p.add_argument("--db", help="File SQLite đích (mặc định: biến MEDASSIST_DB)")
    p.add_argument("--col", action="append", default=[], metavar="khoá=Tên cột",
                   help="Đổi tên cột: name, composition, uses, side_effects, manufacturer")
    p.add_argument("--authoritative", action="store_true", help="Đánh dấu là nguồn lâm sàng đã kiểm duyệt")
    p.add_argument("--reviewed-by", help="Tên người đã kiểm duyệt nguồn (bắt buộc với --authoritative)")
    p.add_argument("--source-url", help="URL gốc (https) của tài liệu nguồn")
    p.add_argument("--retrieved-at", help="Ngày lấy dữ liệu, ISO YYYY-MM-DD")
    p.add_argument("--license", help="Giấy phép/điều khoản sử dụng (ngắn)")
    p.add_argument("--replace-source", action="store_true", help="Xoá bản ghi cùng nguồn trước khi nạp")
    p.add_argument("--drop-dev-sample", action="store_true", help="Gỡ thuốc mẫu dev-sample")
    p.add_argument("--keep-duplicates", action="store_true", help="Không loại dòng trùng")
    p.add_argument("--apply", action="store_true", help="Thực sự ghi vào CSDL (mặc định chỉ chạy thử)")
    args = p.parse_args(argv)

    try:
        source = args.source.strip()
        if args.authoritative:
            if not (args.reviewed_by and args.reviewed_by.strip()):
                raise IngestError("--authoritative cần --reviewed-by \"Tên người kiểm duyệt\".")
            source = f"{source}; kiểm duyệt: {args.reviewed_by.strip()}"
        if source.lower().startswith(DEV_SOURCE_PREFIX):
            raise IngestError("Nhãn nguồn không được bắt đầu bằng 'dev-sample:' (dành cho dữ liệu mẫu).")

        drugs, report = load_csv_mapped(
            args.csv,
            _parse_col_overrides(args.col),
            source=source,
            authoritative=args.authoritative,
            dedupe=not args.keep_duplicates,
            source_url=args.source_url,
            retrieved_at=args.retrieved_at,
            license=args.license,
            reviewed_by=args.reviewed_by,
        )
    except IngestError as exc:
        print(f"LỖI: {exc}", file=out)
        return 2

    print(f"Nguồn             : {source}", file=out)
    print(f"Nguồn lâm sàng    : {'CÓ' if args.authoritative else 'không (chỉ tham khảo)'}", file=out)
    _print_report(report, out)

    if report.imported == 0:
        print("LỖI: không có dòng nào để nạp.", file=out)
        return 2
    if not args.apply:
        print("\nĐÂY LÀ CHẠY THỬ — chưa ghi gì. Thêm --apply để ghi.", file=out)
        return 0

    db_path = args.db or os.environ.get("MEDASSIST_DB")
    if not db_path or db_path == ":memory:":
        print("LỖI: cần --db hoặc biến MEDASSIST_DB trỏ tới file SQLite (không dùng :memory:).", file=out)
        return 2
    db_file = Path(db_path)
    if db_file.exists():
        backup = db_file.with_name(f"{db_file.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(db_file, backup)
        print(f"Đã sao lưu CSDL   : {backup.name}", file=out)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    repo = SQLiteRepository(db_file)
    try:
        if args.replace_source:
            print(f"Đã xoá bản cũ cùng nguồn: {repo.delete_medicines_by_source(source)}", file=out)
        if args.drop_dev_sample:
            print(f"Đã gỡ thuốc mẫu   : {repo.delete_medicines_by_source(DEV_SOURCE_PREFIX, prefix=True)}", file=out)
        added = repo.add_medicines(to_rows(drugs))
        print(f"\nĐÃ NẠP {added} thuốc.", file=out)
        print("--- Danh mục hiện có (theo nguồn) ---", file=out)
        for src, (total, auth) in repo.medicine_counts_by_source().items():
            print(f"  {total:>7}  (lâm sàng: {auth})  {src or '(không nguồn)'}", file=out)
    finally:
        repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
