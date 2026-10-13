"""Mock Flask server cho dev (Day 8) — **KHÔNG phải backend thật**.

Vì sao có file này: repo app *không* chứa mã nguồn server Flask cũ (đã kiểm tra:
cả ổ đĩa không có file `.py` nào). Nhưng Day 8 cần server chạy để test tay 5 luồng,
và Day 9 cần server để nối UI. File này trả **đúng contract** mà app đang parse:

    POST   /process-image    multipart field "image"  -> {"medicine_info": [Medicine]}
    GET    /search-medicine  ?name=...                -> {"results": [...], "query", "total"}
    GET    /history                                   -> {"history": [{id, filepath, search_results}]}
    DELETE /history/<id>                              -> {"success": true}
    POST   /login                                     -> {"success": true, "token", "user"}
    POST   /register                                  -> {"success": true, "message"}
    POST   /logout                                    -> {"success": true}
    GET    /health                                    -> {"status": "ok"}

`Medicine` dùng **đúng tên trường của server cũ** (có chữ hoa và dấu cách — đây
là hợp đồng thật, xem `lib/data/models/medicine.dart`):

    Medicine_Name · Composition · Uses · Side_effects · Manufacturer ·
    "Excellent Review %" · "Average Review %" · "Poor Review %"

Chạy:
    python tools/mock_server/app.py          # nghe trên http://0.0.0.0:5000

Day 15 sẽ thay bằng FastAPI Core thật (`/v1/...`); file này là cầu tạm cho dev.
"""

from __future__ import annotations

import csv
import os
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Console Windows mặc định dùng codepage cp1252 → `print("Đăng nhập...")` ném
# `UnicodeEncodeError` và **server chết trước khi kịp mở cổng** (đã gặp thật khi
# chạy trên máy dev này). Ép stdout/stderr sang UTF-8, và `errors="replace"` để
# nếu console không hiển thị được ký tự nào thì log ra "?" chứ không làm chết app.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Stream đã bị thay thế hoặc không hỗ trợ reconfigure (ví dụ khi bị
        # nhúng vào nơi khác) — không phải lỗi chặn, bỏ qua.
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "medicines.csv"
UPLOAD_DIR = BASE_DIR / "uploads"

# Cột CSV = đúng tên trường JSON mà server cũ trả về.
COLUMNS = [
    "Medicine_Name",
    "Composition",
    "Uses",
    "Side_effects",
    "Manufacturer",
    "Excellent Review %",
    "Average Review %",
    "Poor Review %",
]

app = Flask(__name__)
# Cho phép gọi từ web/emulator trong lúc dev (release không dùng file này).
CORS(app)


def _fold(text: str) -> str:
    """Bỏ dấu tiếng Việt + hạ chữ thường để tìm kiếm không dấu.

    `unicodedata.normalize("NFD")` tách dấu thanh khỏi ký tự gốc, còn `đ`
    không tách được nên phải thay tay. Nhờ vậy gõ "thuoc" vẫn ra "thuốc".
    """
    text = text.lower().replace("đ", "d")
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _load_medicines() -> list[dict]:
    """Đọc `medicines.csv`; thiếu file thì dùng dữ liệu mẫu trong code."""
    if not DATA_FILE.exists():
        return [dict(zip(COLUMNS, row)) for row in _SEED_ROWS]

    with DATA_FILE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Chuẩn hoá: ô thiếu cột -> "", và bỏ dòng không có tên thuốc.
    cleaned = []
    for row in rows:
        item = {column: (row.get(column) or "").strip() for column in COLUMNS}
        if item["Medicine_Name"]:
            cleaned.append(item)
    return cleaned


# Dữ liệu mẫu dùng khi chưa có CSV: cố ý để vài ô rỗng/null giống CSV thật
# (đúng kiểu dữ liệu mà `json_utils.dart` phải chịu được).
_SEED_ROWS = [
    ["Panadol 500mg", "Paracetamol 500mg", "Pain relief, fever reduction",
     "Nausea, allergic reactions", "GSK", "93%", "5%", "2%"],
    ["Efferalgan", "Paracetamol 1g", "", "", "UPSA", "88%", "", ""],
]

MEDICINES = _load_medicines()

# --- Người dùng (in-memory) -------------------------------------------------
# Server thật lưu DB + hash mật khẩu; mock giữ trong RAM cho gọn. Mật khẩu ở
# đây là dữ liệu TEST ("demo123"), không phải tài khoản thật.
USERS: dict[str, dict] = {
    "demo": {
        "id": 1,
        "username": "demo",
        "password": "demo123",
        "email": "demo@example.com",
    },
}
NEXT_USER_ID = 2

# Tài khoản đặc biệt cho ĐÚNG MỘT mục đích: trả `HTTP 200 + {"success": false}`
# — kiểu phản hồi mà `RemoteAuthRepository._accept` cũng coi là đăng nhập thất
# bại (một số bản Flask cũ trả kiểu này thay vì 401). Đăng nhập bằng user này
# luôn thất bại, nhờ vậy nhánh đó được kiểm tra thật mà không phải sửa code.
LEGACY_FAILURE_USER = "legacy"


def _credentials() -> dict:
    """Đọc body JSON; nếu thiếu Content-Type thì đọc form cho dễ test tay."""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return request.form.to_dict()


@app.post("/login")
def login():
    body = _credentials()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    user = USERS.get(username)

    if username == LEGACY_FAILURE_USER:
        # 200 + success:false (không phải 401) — nhánh "server cũ".
        return jsonify(
            {"success": False, "message": "Sai tài khoản hoặc mật khẩu"}
        )

    if user is None or user["password"] != password:
        # 401 + message tiếng Việt: đúng thứ `AppExceptionKind.unauthorized`
        # của app mong đợi, và app cũ cũng hiện nguyên văn message này.
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Sai tài khoản hoặc mật khẩu",
                }
            ),
            401,
        )

    return jsonify(
        {
            "success": True,
            "token": f"mock-{uuid.uuid4().hex}",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
            },
        }
    )


@app.post("/register")
def register():
    body = _credentials()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""

    if not username or not password:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Thiếu tên đăng nhập hoặc mật khẩu",
                }
            ),
            400,
        )
    if username in USERS:
        return (
            jsonify({"success": False, "message": "Tên đăng nhập đã tồn tại"}),
            409,
        )

    global NEXT_USER_ID
    USERS[username] = {
        "id": NEXT_USER_ID,
        "username": username,
        "password": password,
        "email": (body.get("email") or "").strip(),
    }
    NEXT_USER_ID += 1

    # Bản cũ chỉ trả message, KHÔNG trả token -> `AuthSession.token == null`,
    # nên app phải quay ra màn Đăng nhập (đúng hành vi bản cũ).
    return jsonify({"success": True, "message": "User registered"}), 201


@app.post("/logout")
def logout():
    # Server cũ có thể không có endpoint này; adapter vẫn cố gọi (best-effort)
    # nên mock trả 200 cho gọn log.
    return jsonify({"success": True, "message": "Đã đăng xuất"})

# --- Lịch sử quét ảnh (in-memory) -------------------------------------------
NEXT_HISTORY_ID = 1
HISTORY: list[dict] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_history(filepath: str, medicines: list[dict]) -> dict:
    global NEXT_HISTORY_ID
    item = {
        "id": NEXT_HISTORY_ID,
        "filepath": filepath,
        "search_results": medicines,
        "created_at": _now(),
    }
    HISTORY.append(item)
    NEXT_HISTORY_ID += 1
    return item


def _match(needle: str) -> list[dict]:
    """Tìm theo tên + hoạt chất + nhà sản xuất, không phân biệt dấu/hoa thường.

    Mọi token đều phải xuất hiện (AND) nên gõ "para 500" ra đúng
    "Panadol 500mg", và gõ "thuoc" (không dấu) vẫn khớp ô tiếng Việt.
    """
    tokens = [token for token in _fold(needle).split() if token]
    if not tokens:
        return []

    found: list[dict] = []
    for medicine in MEDICINES:
        haystack = _fold(
            " ".join(
                [
                    medicine["Medicine_Name"],
                    medicine["Composition"],
                    medicine["Manufacturer"],
                ]
            )
        )
        if all(token in haystack for token in tokens):
            found.append(medicine)
    return found


@app.get("/search-medicine")
def search_medicine():
    name = (request.args.get("name") or "").strip()
    results = _match(name)
    # `results` là khoá mà `MedicineSearchResponse` (Day 6) đọc. Trả kèm
    # `search_results` để màn hình cũ chưa kịp thay vẫn chạy (Day 9 mới thay).
    return jsonify(
        {
            "query": name,
            "total": len(results),
            "results": results,
            "search_results": results,
        }
    )


@app.post("/process-image")
def process_image():
    upload = request.files.get("image")
    if upload is None:
        return (
            jsonify(
                {"success": False, "message": "Thiếu file ở field 'image'"}
            ),
            400,
        )

    UPLOAD_DIR.mkdir(exist_ok=True)
    safe_name = Path(upload.filename or "photo.jpg").name
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    upload.save(stored)

    # Mock không có OCR, nên quy ước theo TÊN FILE để test được cả 2 nhánh:
    #   tên chứa "empty" / "khong-thay"  -> ảnh không đọc ra thuốc nào
    #   còn lại                          -> trả về Panadol 500mg
    lowered = safe_name.lower()
    wants_empty = (
        "empty" in lowered
        or "khong-thay" in lowered
        or request.form.get("empty") == "1"
    )
    found = [] if wants_empty else _match("Panadol")[:1]

    _add_history(str(stored), found)

    # `medicine_info` là hợp đồng cũ; kèm `image_path`/`analyzed_at` mà
    # `AnalysisResult` đã đọc sẵn (Day 12 dùng cho cache).
    return jsonify(
        {
            "medicine_info": found,
            "image_path": str(stored),
            "analyzed_at": _now(),
        }
    )


@app.get("/history")
def get_history():
    # Mới nhất lên đầu, giống UI cũ (mục vừa quét nằm trên cùng).
    return jsonify({"history": list(reversed(HISTORY))})


@app.delete("/history/<int:history_id>")
def delete_history(history_id: int):
    for item in HISTORY:
        if item["id"] == history_id:
            HISTORY.remove(item)
            return jsonify(
                {"success": True, "message": f"Đã xoá history {history_id}"}
            )
    return (
        jsonify(
            {
                "success": False,
                "message": f"Không tìm thấy history {history_id}",
            }
        ),
        404,
    )


@app.get("/health")
def health():
    # Nằm trong `ApiClient.publicPaths` — dùng để biết server đã sống chưa.
    return jsonify(
        {
            "status": "ok",
            "service": "mock-flask",
            "medicines": len(MEDICINES),
            "history": len(HISTORY),
        }
    )

if __name__ == "__main__":
    # host 0.0.0.0 để ĐIỆN THOẠI THẬT cùng LAN gọi được (khi đó phải dùng IP LAN
    # của máy tính và thêm IP đó vào `android/app/src/debug/res/xml/
    # network_security_config.xml`). Emulator Android chỉ cần `10.0.2.2` — đã
    # được cho phép sẵn trong file đó.
    port = int(os.environ.get("PORT", "5000"))
    # Mặc định TẮT reloader để không sinh tiến trình con (dễ tắt khi test tự
    # động). Bật khi cần sửa code tự nạp lại: $env:MOCK_DEBUG='1'
    debug = os.environ.get("MOCK_DEBUG", "0") == "1"

    print(f"Mock Flask: http://0.0.0.0:{port} "
          f"({len(MEDICINES)} thuốc, {len(HISTORY)} history)")
    print(f"Emulator Android: flutter run --dart-define=API_BASE_URL="
          f"http://10.0.2.2:{port}")
    print("Đăng nhập thử: demo / demo123")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)