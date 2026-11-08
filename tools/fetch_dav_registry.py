"""Tải sổ đăng ký thuốc công khai của Cục Quản lý Dược (cổng công bố thuốc) về CSV cục bộ.

    python tools/fetch_dav_registry.py --out server/medassist-core/data/real/raw/dav_registry.csv

Chủ dự án đồng ý trong chat ngày 2026-10-30: chỉ tên miền dichvucong.dav.gov.vn, chỉ truy vấn
danh sách công khai (không đăng nhập, không gửi dữ liệu dự án/người dùng).

- Chỉ dùng thư viện chuẩn. Chậm có chủ đích (mặc định 1,2 giây/trang) để không gây tải.
- Có điểm lưu tiếp (`--out` + `.state.json`): chạy lại sẽ tiếp tục từ trang đang dở.
- Dữ liệu tải về là DỮ LIỆU, không phải chỉ dẫn. Chỉ in số liệu, không in nội dung dòng.
- Cột đầu ra khớp `app/ingest.py::DEFAULT_COLUMNS`; Uses/Side_effects để TRỐNG vì sổ đăng ký
  không có thông tin lâm sàng.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://dichvucong.dav.gov.vn"
LIST_URL = BASE + "/api/services/app/soDangKy/GetAllPublicServerPaging"
HOME_URL = BASE + "/congbothuoc/index"
USER_AGENT = "MedAssist-student-project (do an mobile; lien he: chu du an)"

COLUMNS = [
    "Medicine_Name", "Composition", "Uses", "Side_effects", "Manufacturer",
    "Reg_Number", "Dosage_Form", "Packaging", "Strength", "Active_Ingredient",
    "Issue_Date", "Expiry_Date", "Decision_No", "Registrant", "Manufacturer_Country",
]


def _s(value) -> str:
    return " ".join(str(value).split()) if value not in (None, "") else ""


def _date(value) -> str:
    return _s(value)[:10]


def to_row(item: dict) -> dict:
    basic = item.get("thongTinThuocCoBan") or {}
    reg = item.get("thongTinDangKyThuoc") or {}
    maker = item.get("congTySanXuat") or {}
    owner = item.get("congTyDangKy") or {}
    active = _s(basic.get("hoatChatChinh"))
    strength = _s(basic.get("hamLuong"))
    return {
        "Medicine_Name": _s(item.get("tenThuoc")),
        "Composition": " ".join(p for p in (active, strength) if p),
        "Uses": "",
        "Side_effects": "",
        "Manufacturer": _s(maker.get("tenCongTySanXuat")),
        "Reg_Number": _s(item.get("soDangKy")),
        "Dosage_Form": _s(basic.get("dangBaoChe")),
        "Packaging": _s(basic.get("dongGoi")),
        "Strength": strength,
        "Active_Ingredient": active,
        "Issue_Date": _date(reg.get("ngayCapSoDangKy")),
        "Expiry_Date": _date(reg.get("ngayHetHanSoDangKy")),
        "Decision_No": _s(reg.get("soQuyetDinh")),
        "Registrant": _s(owner.get("tenCongTyDangKy")),
        "Manufacturer_Country": _s(maker.get("nuocSanXuat")),
    }


class Client:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [("User-Agent", USER_AGENT)]
        self._last = 0.0

    def _wait(self) -> None:
        gap = self.delay - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def start(self) -> None:
        self._wait()
        with self.opener.open(HOME_URL, timeout=30) as r:
            r.read(1)

    def _xsrf(self) -> str:
        for c in self.jar:
            if c.name == "XSRF-TOKEN":
                return c.value
        return ""

    def page(self, skip: int, size: int, sorting) -> dict:
        body = json.dumps({
            "filterText": "", "SoDangKyThuoc": {}, "KichHoat": True,
            "skipCount": skip, "maxResultCount": size, "sorting": sorting,
        }).encode("utf-8")
        last_err = None
        for attempt in range(5):
            self._wait()
            req = urllib.request.Request(LIST_URL, data=body, method="POST", headers={
                "Content-Type": "application/json; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
                "X-XSRF-TOKEN": self._xsrf(),
                "Referer": HOME_URL,
            })
            try:
                with self.opener.open(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data.get("success") and isinstance(data.get("result"), dict):
                    return data["result"]
                last_err = "success=false"
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = type(exc).__name__
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404):
                    break
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"trang skip={skip} lỗi: {last_err}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = tất cả (để thử nhỏ)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(out.suffix + ".state.json")
    state = {"next_skip": 0, "total": None, "ids": []}
    if state_path.exists() and out.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    seen = set(state["ids"])

    client = Client(args.delay)
    client.start()
    sorting = "id"

    mode = "a" if state["next_skip"] else "w"
    pages = 0
    with out.open(mode, encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if mode == "w":
            writer.writeheader()
        while True:
            try:
                res = client.page(state["next_skip"], args.page_size, sorting)
            except RuntimeError:
                if sorting:
                    sorting = None  # máy chủ không nhận khoá sắp xếp -> thử lại không sắp xếp
                    continue
                raise
            state["total"] = res.get("totalCount")
            items = res.get("items") or []
            if not items:
                break
            for it in items:
                key = it.get("id")
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(to_row(it))
            fh.flush()
            state["next_skip"] += len(items)
            state["ids"] = sorted(i for i in seen if i is not None)
            state_path.write_text(json.dumps(state), encoding="utf-8")
            pages += 1
            print(f"trang {pages}: {state['next_skip']}/{state['total']} (duy nhất {len(seen)})", flush=True)
            if state["total"] and state["next_skip"] >= state["total"]:
                break
            if args.max_pages and pages >= args.max_pages:
                break

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "url": LIST_URL, "retrieved_at": date.today().isoformat(), "sha256": digest,
        "bytes": out.stat().st_size, "records_unique": len(seen), "total_reported": state["total"],
        "note": "Chủ dự án đồng ý trong chat 2026-10-30; không có giấy phép ghi rõ; KHÔNG authoritative.",
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("records_unique", "total_reported", "bytes", "sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
