#!/usr/bin/env python
"""Quét dự án MedAssist và sinh `docs/project-map.html` — sơ đồ tương tác (chạy cục bộ).

    python tools/project_map/build_map.py

## Quy tắc (AGENTS.md)

- **Chạy hoàn toàn cục bộ**, không gọi mạng, không gửi mã đi đâu. Trang HTML sinh ra
  không tải thư viện/phông chữ ngoài (mở được khi mất mạng).
- Chỉ đọc các file mã nguồn `.dart`, `.py`, `.ps1` và vài file khai báo thư viện
  (`pubspec.yaml`, `requirements.txt`). KHÔNG đọc: `.env*`, khoá/keystore, `key.properties`,
  `google-services.json`, CSDL, CSV, ảnh, thư mục `data/` (kể cả `data/real/`), `uploads/`.
- Sơ đồ có hai phần: (1) **tự trích xuất** từ mã (import, lớp, route, bảng, biến môi trường,
  thư viện); (2) **luồng chạy do con người viết** (`FLOWS`), được **kiểm tra** với mã thật:
  mọi node nhắc tới phải tồn tại, mọi endpoint phải có route thật — nếu không, script báo lỗi.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "project-map.html"
TEMPLATE = Path(__file__).with_name("template.html")
PKG = "image_processing_app"
SERVER = "server/medassist-core"

# ---------------------------------------------------------------- phạm vi quét

#: Gốc quét. Mọi thứ khác (android/, ios/, data/, build/, test/...) bị bỏ qua.
SCAN_ROOTS = ["lib", f"{SERVER}/app", "tools", f"{SERVER}/eval.py"]
ALLOWED_EXT = {".dart", ".py", ".ps1"}
SKIP_PARTS = {
    "build", ".dart_tool", ".git", "__pycache__", "node_modules", ".kilo", ".idea",
    "uploads", "generated", "venv", ".venv", "project_map",
}
#: Thư mục dữ liệu của server (CSDL, CSV thật `data/real/`, bản sao lưu) — KHÔNG BAO GIỜ quét.
SKIP_PREFIXES = (f"{SERVER}/data/", f"{SERVER}/.pytest_cache/")
SECRET_NAME = re.compile(
    r"(\.env|\.pem$|\.p12$|\.jks$|\.keystore$|key\.properties|google-services|GoogleService-Info)", re.I
)


def want(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path.suffix.lower() not in ALLOWED_EXT or SECRET_NAME.search(path.name):
        return False
    if rel.as_posix().startswith(SKIP_PREFIXES):
        return False
    return not (set(rel.parts) & SKIP_PARTS)


def files() -> list[Path]:
    out: list[Path] = []
    for r in SCAN_ROOTS:
        p = ROOT / r
        if p.is_file():
            if want(p):
                out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and want(f)]
    return sorted(set(out))


# -------------------------------------------------------------------- làn (lane)

LANES = [
    ("user", "Người dùng & thiết bị"),
    ("ui", "Giao diện (app)"),
    ("state", "Trạng thái · hợp đồng · mô hình"),
    ("adapter", "Adapter & dịch vụ trên máy"),
    ("api", "API (HTTP / SSE)"),
    ("server", "Backend FastAPI"),
    ("ai", "Agent · truy hồi · AI"),
    ("store", "Lưu trữ"),
    ("devtool", "Công cụ dev & dữ liệu"),
]


def classify(rel: str) -> tuple[str, str, str]:
    """(lane, kind, group) từ đường dẫn."""
    p = rel
    if p.startswith("lib/screens/"):
        return ("state", "controller", "screens") if p.endswith("_controller.dart") else ("ui", "screen", "screens")
    if p.startswith("lib/widgets/"):
        return "ui", "widget", "widgets"
    if p == "lib/main.dart":
        return "ui", "entry", "main"
    if p.startswith("lib/core/theme"):
        return "ui", "theme", "core/theme"
    if p == "lib/core/providers.dart" or p == "lib/core/voice_providers.dart":
        return "state", "state", "core"
    if p.startswith("lib/core/config"):
        return "state", "config", "core/config"
    if p.startswith("lib/core/network"):
        return "adapter", "network", "core/network"
    if p.startswith("lib/domain/ports"):
        return "state", "port", "domain/ports"
    if p.startswith("lib/data/models"):
        return "state", "model", "data/models"
    if p.startswith("lib/data/adapters"):
        return "adapter", "adapter", "data/adapters"
    if p.startswith("lib/data/local"):
        return "adapter", "local", "data/local"
    if p.startswith(f"{SERVER}/app/routers/"):
        return "server", "router", "routers"
    if p.endswith("/app/main.py"):
        return "server", "entry", "app"
    if p.endswith(("/deps.py", "/security.py", "/jwt_tokens.py", "/rate_limit.py", "/errors.py",
                   "/observability.py", "/privacy.py", "/schemas.py")):
        return "server", "service", "app"
    if p.endswith(("/orchestrator.py", "/agent.py", "/prompt.py", "/retrieval.py", "/tools.py",
                   "/mcp_tools.py", "/mcp_server.py", "/textnorm.py", "/sessions.py")):
        return "ai", "agent", "agent"
    if p.endswith(("/llm.py", "/llm_ollama.py")):
        return "ai", "llm", "llm"
    if p.endswith(("/sqlite_repository.py", "/repository.py")):
        return "store", "repo", "repo"
    if p.endswith(("/ingest.py", "/import_real.py", "/seed.py")) or p.endswith("eval.py"):
        return "devtool", "script", "data tools"
    if p.startswith("tools/"):
        return "devtool", "script", "tools"
    return "server", "module", "app"


# ------------------------------------------------------------------ trích xuất

def first_line(text: str | None) -> str:
    for ln in (text or "").splitlines():
        ln = ln.strip().lstrip("/#*").strip()
        if ln and not ln.startswith(("```", "---", "##")):
            return ln.replace("**", "").replace("`", "")   # bỏ ký hiệu markdown
    return ""


DART_DECL = re.compile(
    r"^(?:(?:abstract|sealed|final|base|interface)\s+)*(class|mixin|enum|extension)\s+(\w+)", re.M
)
DART_PROVIDER = re.compile(r"^final\s+(?:[\w<>?,\s]+?\s+)?(\w+Provider)\b\s*=", re.M)
DART_IMPORT = re.compile(r"""^\s*(?:import|export)\s+'([^']+)'""", re.M)
DART_ENV = re.compile(r"String\.fromEnvironment\(\s*'(\w+)'")


def dart_doc_before(lines: list[str], idx: int) -> str:
    docs: list[str] = []
    i = idx - 1
    while i >= 0 and (lines[i].strip().startswith("///") or lines[i].strip().startswith("@")):
        if lines[i].strip().startswith("///"):
            docs.append(lines[i].strip()[3:].strip())
        i -= 1
    return first_line("\n".join(reversed(docs)))


def parse_dart(path: Path, rel: str) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    desc = ""
    m = re.search(r"((?:^///.*\n)+)", src, re.M)
    if m:
        desc = first_line(m.group(1))
    symbols = []
    for dm in DART_DECL.finditer(src):
        line_idx = src.count("\n", 0, dm.start())
        symbols.append({"name": dm.group(2), "kind": dm.group(1), "doc": dart_doc_before(lines, line_idx)})
    for pm in DART_PROVIDER.finditer(src):
        line_idx = src.count("\n", 0, pm.start())
        symbols.append({"name": pm.group(1), "kind": "provider", "doc": dart_doc_before(lines, line_idx)})
    imports: list[str] = []
    for im in DART_IMPORT.finditer(src):
        target = im.group(1)
        if target.startswith(f"package:{PKG}/"):
            imports.append("lib/" + target[len(f"package:{PKG}/"):])
        elif target.startswith("."):
            imports.append(((path.parent / target).resolve().relative_to(ROOT)).as_posix())
    ext_pkgs = sorted({t[len("package:"):].split("/")[0] for t in DART_IMPORT.findall(src)
                       if t.startswith("package:") and not t.startswith(f"package:{PKG}/")})
    env = sorted(set(DART_ENV.findall(src)))
    return {"desc": desc, "symbols": symbols, "imports": imports, "ext_pkgs": ext_pkgs,
            "env": env, "lines": len(lines), "src": src}


def parse_python(path: Path, rel: str, module_map: dict[str, str]) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    desc = first_line(ast.get_docstring(tree))
    symbols = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append({"name": node.name, "kind": "class", "doc": first_line(ast.get_docstring(node))})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            symbols.append({"name": node.name, "kind": "def", "doc": first_line(ast.get_docstring(node))})
    ext_pkgs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if mod == "app" or mod.startswith("app."):
                if mod in module_map:
                    imports.append(module_map[mod])
                for alias in node.names:
                    sub = f"{mod}.{alias.name}"
                    if sub in module_map:
                        imports.append(module_map[sub])
            else:
                ext_pkgs.add(mod.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    if alias.name in module_map:
                        imports.append(module_map[alias.name])
                else:
                    ext_pkgs.add(alias.name.split(".")[0])
    routes = []
    prefix = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "id", "") == "APIRouter":
            for kw in node.value.keywords:
                if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                    prefix = kw.value.value
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) \
                        and getattr(dec.func.value, "id", "") == "router" \
                        and dec.func.attr in {"get", "post", "put", "delete", "patch"}:
                    sub = ""
                    if dec.args and isinstance(dec.args[0], ast.Constant):
                        sub = dec.args[0].value
                    summary = ""
                    for kw in dec.keywords:
                        if kw.arg == "summary" and isinstance(kw.value, ast.Constant):
                            summary = kw.value.value
                    needs_auth = any(
                        isinstance(d, ast.Call) and getattr(d.func, "id", "") == "Depends"
                        and d.args and getattr(d.args[0], "id", "") == "current_user_id"
                        for d in list(node.args.defaults) + list(node.args.kw_defaults)
                        if d is not None
                    )
                    routes.append({"method": dec.func.attr.upper(), "path": prefix + sub,
                                   "handler": node.name, "summary": summary,
                                   "doc": first_line(ast.get_docstring(node)), "auth": needs_auth})
    env = sorted(set(re.findall(r"\b(MEDASSIST_[A-Z_]+)\b", src)))
    tables = []
    for tm in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\);", src, re.S):
        cols = []
        for ln in tm.group(2).splitlines():
            ln = ln.strip().rstrip(",")
            cm = re.match(r"^(\w+)\s+(INTEGER|TEXT|REAL|BLOB)\b(.*)$", ln)
            if cm and not ln.startswith("--"):
                cols.append(f"{cm.group(1)} {cm.group(2)}{' ' + cm.group(3).strip() if cm.group(3).strip() else ''}")
        tables.append({"name": tm.group(1), "cols": cols, "where": "server"})
    return {"desc": desc, "symbols": symbols, "imports": sorted(set(imports)), "ext_pkgs": sorted(ext_pkgs),
            "routes": routes, "env": env, "tables": tables, "lines": len(src.splitlines()), "src": src}


def parse_ps1(path: Path, rel: str, module_map: dict[str, str], ps1_names: dict[str, str]) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    head = []
    for ln in lines:
        if ln.startswith("#"):
            head.append(ln.lstrip("# ").rstrip())
        elif head:
            break
    desc = first_line("\n".join(h for h in head if h))
    params = re.findall(r"^\s*\[?[\w\[\]]*\]?\s*\$(\w+)\s*(?:=|,|\))", src.split("param(", 1)[1].split("\n)\n", 1)[0], re.M) \
        if "param(" in src else []
    symbols = [{"name": "-" + p, "kind": "param", "doc": ""} for p in params[:12]]
    imports = []
    for mod in re.findall(r"-m\s+(app\.\w+)", src):
        if mod in module_map:
            imports.append(module_map[mod])
    for name, relp in ps1_names.items():
        if name != Path(rel).name and re.search(rf"\b{re.escape(name)}\b", src):
            imports.append(relp)
    return {"desc": desc, "symbols": symbols, "imports": sorted(set(imports)), "ext_pkgs": [],
            "env": sorted(set(re.findall(r"\$env:(MEDASSIST_\w+|OLLAMA_\w+)", src))),
            "lines": len(lines), "src": src}


# ------------------------------------------------------------ nút "bên ngoài"

EXTERNALS = [
    # id, lane, nhãn, mô tả
    ("ext:user", "user", "Người dùng", "Người dùng cuối thao tác trên app (đăng nhập, tìm thuốc, hỏi đáp, quét hộp thuốc, đặt nhắc)."),
    ("ext:photos", "user", "Bộ chọn ảnh Android", "Photo picker của hệ điều hành: chỉ thấy ảnh nằm TRONG thiết bị (MediaStore). Ảnh từ máy tính phải được chép sang bằng adb."),
    ("ext:notif", "user", "Thông báo Android (AlarmManager)", "Báo thức chính xác + thông báo. Cần quyền POST_NOTIFICATIONS và SCHEDULE_EXACT_ALARM; dựng lại khi khởi động máy."),
    ("ext:pc", "user", "Thư mục ảnh trên máy tính", "Thư mục anh_thuoc\\ (hoặc thư mục bạn chọn). Script sync_images.ps1 theo dõi và tự chép ảnh sang thiết bị."),
    ("ext:mlkit", "adapter", "ML Kit Text Recognition", "Mô hình OCR của Google chạy NGAY TRÊN ĐIỆN THOẠI: ảnh không rời máy."),
    ("ext:keystore", "store", "Android Keystore", "Nơi lưu access/refresh token (flutter_secure_storage). Không bao giờ dùng shared_preferences cho token."),
    ("ext:prefs", "store", "shared_preferences", "Lưu cache công khai (kết quả tìm kiếm) và cờ đồng ý. KHÔNG chứa bí mật."),
    ("ext:local_db", "store", "SQLite trên máy · health_profile.db", "Hồ sơ sức khoẻ, lịch nhắc thuốc, nhật ký uống thuốc. Không rời máy."),
    ("ext:server_db", "store", "SQLite server · .db", "medassist.dev.db (demo, 10 thuốc mẫu) hoặc data/real/medassist.real.db (dữ liệu thật). Đặt bằng MEDASSIST_DB."),
    ("ext:ram", "store", "RAM của server", "Phiên chat, số liệu sử dụng, bộ đếm giới hạn tần suất. Mất khi khởi động lại server."),
    ("ext:ollama", "ai", "Ollama · qwen2.5:3b", "LLM thật chạy trên chính máy này (127.0.0.1:11434). Chỉ nhận địa chỉ loopback; bật bằng MEDASSIST_LLM=ollama."),
    ("ext:csv", "devtool", "CSV danh mục thuốc", "File CSV do chủ dự án cung cấp (data/real/raw/…). Công cụ chỉ in số liệu, không in nội dung."),
    ("ext:adb", "devtool", "adb (Android Debug Bridge)", "Cầu nối máy tính ↔ máy ảo/điện thoại: chép ảnh, mở app, chỉnh đồng hồ khi thử."),
]

#: (regex trên nội dung, id ngoài, nhãn quan hệ, chỉ áp dụng cho đường dẫn khớp regex này)
EXT_PATTERNS = [
    (r"package:sqflite", "ext:local_db", "đọc/ghi SQLite trên máy", r"^lib/"),
    (r"package:shared_preferences", "ext:prefs", "đọc/ghi cache & cờ đồng ý", r"^lib/"),
    (r"package:flutter_secure_storage", "ext:keystore", "lưu token an toàn", r"^lib/"),
    (r"package:google_mlkit_text_recognition", "ext:mlkit", "OCR ảnh trên máy", r"^lib/"),
    (r"package:flutter_local_notifications", "ext:notif", "đặt thông báo nhắc thuốc", r"^lib/"),
    (r"package:image_picker", "ext:photos", "chọn ảnh từ thư viện ảnh", r"^lib/"),
    (r"\bsqlite3\.connect\(", "ext:server_db", "kết nối SQLite", rf"^{SERVER}/app/"),
    (r"/api/chat", "ext:ollama", "gọi LLM cục bộ", rf"^{SERVER}/app/llm_ollama\.py$"),
    (r"adb\.exe|\badb\b", "ext:adb", "điều khiển thiết bị qua adb", r"\.ps1$"),
    (r"csv\.DictReader", "ext:csv", "đọc CSV danh mục", rf"^{SERVER}/app/ingest\.py$"),
]
#: Nối tay những quan hệ không nhận ra được bằng regex (đã kiểm tra tồn tại file).
EXT_MANUAL = [
    (f"{SERVER}/app/sessions.py", "ext:ram", "giữ phiên chat trong RAM"),
    (f"{SERVER}/app/rate_limit.py", "ext:ram", "bộ đếm giới hạn tần suất"),
    (f"{SERVER}/app/observability.py", "ext:ram", "đếm số liệu sử dụng"),
    ("tools/sync_images.ps1", "ext:pc", "theo dõi thư mục ảnh trên máy tính"),
    (f"{SERVER}/app/import_real.py", "ext:csv", "đọc CSV để nạp"),
    (f"{SERVER}/app/ingest.py", "ext:csv", "đọc CSV để nạp"),
    ("lib/screens/process_image_screen.dart", "ext:user", "người dùng chụp/chọn ảnh"),
    ("lib/screens/login_screen.dart", "ext:user", "người dùng nhập tên + mật khẩu"),
]

# --------------------------------------------------------------- luồng chạy (tay)
# Mỗi bước: node (đường dẫn file | "api:METHOD /đường/dẫn" | "ext:..."), tiêu đề, chi tiết,
# dữ liệu vào/ra. Script kiểm tra mọi node/endpoint có thật.
S = SERVER + "/app/"
L = "lib/"
FLOWS = [
    {
        "id": "login", "title": "Đăng nhập", "icon": "🔐",
        "summary": "Từ ô nhập tên + mật khẩu đến token lưu trong Keystore.",
        "steps": [
            ("ext:user", "Nhập tên đăng nhập + mật khẩu", "Ô đăng nhập dùng TÊN ĐĂNG NHẬP, không dùng email.", "vào: username, password", ""),
            (L + "screens/login_screen.dart", "Màn hình Đăng nhập", "Kiểm tra form rồi gọi AuthController.login.", "", ""),
            (L + "core/providers.dart", "AuthController (Riverpod)", "AsyncNotifier giữ AuthSession; chỉ biết cổng AuthRepository, không biết HTTP.", "", "AuthSession?"),
            (L + "domain/ports/auth_repository.dart", "Cổng AuthRepository", "Hợp đồng thuần Dart: login(username, password) → Result<AuthSession>.", "", ""),
            (L + "data/adapters/remote_auth_repository.dart", "Adapter HTTP", "Dựng body JSON, gọi ApiClient, parse token + user.", "{username, password}", "{user, token}"),
            (L + "core/network/api_client.dart", "ApiClient (Dio)", "Thêm header, che Authorization/mật khẩu khỏi log; gặp 401 thì thử làm mới token.", "", ""),
            ("api:POST /v1/auth/login", "POST /v1/auth/login", "Điểm vào công khai (không cần token).", "{username, password}", "{user, token{access_token, refresh_token, expires_in…}}"),
            (S + "routers/auth.py", "Router auth", "Giới hạn tần suất (429 + Retry-After), tra user, so khớp mật khẩu.", "", ""),
            (S + "security.py", "Băm / kiểm tra mật khẩu", "bcrypt; từ chối mật khẩu quá 72 byte.", "mật khẩu thô → bản băm", "đúng/sai"),
            (S + "jwt_tokens.py", "Ký token", "Access token ngắn hạn + refresh token (HS256); chỉ lưu BẢN BĂM refresh token.", "", "access + refresh"),
            (S + "sqlite_repository.py", "Kho SQLite", "Đọc users, ghi refresh_tokens (token_hash, family_id).", "", ""),
            (L + "data/adapters/secure_token_store.dart", "Lưu token", "Chép token vào Keystore; không bao giờ ghi log.", "token", ""),
            ("ext:keystore", "Android Keystore", "Token nằm ở đây, không ở shared_preferences.", "", ""),
            (L + "screens/auth_gate.dart", "Cổng xác thực", "Có phiên hợp lệ → vào màn hình chính.", "", ""),
        ],
    },
    {
        "id": "search", "title": "Tìm thuốc theo tên", "icon": "🔎",
        "summary": "Gõ tên thuốc → tìm trong SQLite → hiện kết quả (có cache offline).",
        "steps": [
            ("ext:user", "Gõ tên hoặc hoạt chất", "Ví dụ: Panadol, paracetamol (gõ không dấu cũng được).", "vào: chuỗi tìm kiếm", ""),
            (L + "screens/medicine_search_screen.dart", "Màn hình tìm thuốc", "Gọi SearchController.search; luôn hiện disclaimer y khoa.", "", ""),
            (L + "core/providers.dart", "SearchController + OfflineCache", "Có mạng thì lưu kết quả công khai vào cache; mất mạng thì dùng lại.", "", "SearchSnapshot"),
            (L + "domain/ports/medicine_repository.dart", "Cổng MedicineRepository", "Hợp đồng tìm thuốc/lịch sử.", "", ""),
            (L + "data/adapters/remote_medicine_repository.dart", "Adapter HTTP", "Gọi GET /v1/search?name=…", "name, limit, offset", "danh sách Medicine"),
            ("api:GET /v1/search", "GET /v1/search", "Tìm theo tên hoặc hoạt chất.", "?name=&limit=&offset=", "{results[], total}"),
            (S + "routers/data.py", "Router data", "Chuẩn hoá giới hạn (tối đa 100), gọi kho.", "", ""),
            (S + "sqlite_repository.py", "Kho: _search", "LIKE trên normalized_name / ingredients, tham số hoá (không SQL injection).", "", ""),
            ("ext:server_db", "Bảng medicines", "Chuẩn hoá bỏ dấu để người Việt gõ không dấu vẫn tìm được.", "", ""),
            (L + "data/local/cache_store.dart", "Cache offline", "Chỉ cache dữ liệu công khai; token không nằm ở đây.", "kết quả tìm kiếm", ""),
            ("ext:prefs", "shared_preferences", "Cache kết quả để dùng khi mất mạng.", "", ""),
        ],
    },
    {
        "id": "chat", "title": "Hỏi đáp thuốc (chat + AI)", "icon": "💬",
        "summary": "Câu hỏi → lọc an toàn → truy hồi nguồn → (LLM diễn đạt) → kiểm chứng trích dẫn → SSE.",
        "steps": [
            ("ext:user", "Gõ câu hỏi", "Ví dụ: “Panadol có tác dụng phụ gì”.", "vào: message", ""),
            (L + "screens/agent_chat_screen.dart", "Màn hình chat", "Hiện câu trả lời chạy dần và các thẻ nguồn.", "", ""),
            (L + "screens/chat_controller.dart", "ChatController", "Giữ danh sách tin nhắn; đính kèm thuốc đang dùng từ hồ sơ (nếu có).", "message, sessionId, currentMedications", ""),
            (L + "domain/ports/agent_client.dart", "Cổng AgentClient", "Sự kiện kín: ToolCallStarted, Citation, TokenDelta, Done/Failure.", "", ""),
            (L + "data/adapters/sse_agent_client.dart", "Adapter SSE", "POST rồi đọc luồng text/event-stream.", "", "AgentEvent…"),
            ("api:POST /v1/agent/chat", "POST /v1/agent/chat", "Cần token; user_id lấy từ token, không từ body.", "{message, session_id, current_medications}", "SSE: tool_call · citation · token · done"),
            (S + "routers/agent.py", "Router agent", "Xác thực, gọi orchestrator, phát từng sự kiện SSE.", "", ""),
            (S + "orchestrator.py", "Orchestrator.chat", "Lọc ngoài phạm vi + injection; planner (luật) chọn công cụ; ghép câu trả lời TỪ kết quả công cụ.", "", ""),
            (S + "agent.py", "Lọc an toàn", "is_out_of_scope chặn liều dùng, chẩn đoán, tương tác, cấp cứu.", "", "từ chối nếu vi phạm"),
            (S + "prompt.py", "Chống prompt injection", "Phát hiện câu hỏi đáng ngờ; dựng prompt với nguồn đã escape.", "", ""),
            (S + "tools.py", "Công cụ explain_for_patient", "Không tự diễn giải: chỉ gom trích dẫn truy hồi được.", "query", "citations"),
            (S + "retrieval.py", "Truy hồi BM25", "Chấm điểm, ngưỡng liên quan, neo theo tên thuốc, sinh mã R1, R2…", "câu hỏi + danh sách thuốc ứng viên", "đoạn trích kèm nguồn"),
            (S + "sqlite_repository.py", "candidate_medicines", "Chỉ mục từ khoá trên TOÀN danh mục (không còn giới hạn 100 thuốc đầu).", "", "ứng viên ≤ 1000"),
            ("ext:server_db", "Bảng medicines", "Nguồn dữ liệu duy nhất của câu trả lời.", "", ""),
            (S + "orchestrator.py", "_llm_answer (tuỳ chọn)", "Nếu bật LLM thật: nhờ diễn đạt lại; lỗi/không đạt thì giữ câu trích nguyên văn.", "", ""),
            (S + "llm.py", "get_llm_provider", "Mặc định mock; MEDASSIST_LLM=ollama thì dùng Ollama.", "", ""),
            (S + "llm_ollama.py", "Adapter Ollama", "Chỉ loopback, không proxy, không redirect; không log nội dung.", "messages", "text"),
            ("ext:ollama", "Ollama qwen2.5:3b", "LLM chạy trên máy, dữ liệu không rời máy.", "", ""),
            (S + "agent.py", "verify_grounding", "Mỗi dòng phải có mã trích dẫn tồn tại và bám nội dung nguồn.", "", "đạt / bị loại"),
            (S + "sessions.py", "Lưu phiên (RAM)", "Lưu hai chiều lượt chat; mất khi khởi động lại server.", "", ""),
        ],
    },
    {
        "id": "ocr", "title": "Quét hộp thuốc (OCR trên máy)", "icon": "📷",
        "summary": "Chọn ảnh → ML Kit đọc chữ ngay trên điện thoại → khớp với danh mục thuốc.",
        "steps": [
            ("ext:pc", "Ảnh trên máy tính (tuỳ chọn)", "Bỏ ảnh vào thư mục anh_thuoc\\ khi chạy sync_images.ps1.", "", ""),
            ("tools/sync_images.ps1", "sync_images.ps1", "Theo dõi thư mục, chờ chép xong rồi đẩy ảnh.", "", ""),
            ("tools/push_image.ps1", "push_image.ps1", "adb push vào /sdcard/Pictures/MedAssist rồi báo MediaStore quét.", "", ""),
            ("ext:adb", "adb", "Chép qua cáp/adb, không qua mạng.", "ảnh", ""),
            ("ext:photos", "Bộ chọn ảnh Android", "Thấy ảnh trong album MedAssist.", "", ""),
            (L + "screens/process_image_screen.dart", "Màn hình Quét thuốc", "Chọn ảnh rồi bấm Phân tích.", "đường dẫn ảnh (chỉ trên máy)", ""),
            (L + "core/providers.dart", "AnalysisController", "Chọn analyzer theo --dart-define=ANALYZER (remote | ondevice | hybrid).", "", "AnalysisResult"),
            (L + "data/adapters/hybrid_analyzer.dart", "HybridAnalyzer", "Ưu tiên OCR trên máy; có thể kết hợp server.", "", ""),
            (L + "data/adapters/ondevice_analyzer.dart", "OnDeviceAnalyzer", "Chạy ML Kit rồi khớp chữ đọc được với tên thuốc trong danh mục.", "ảnh", "danh sách thuốc khớp"),
            ("ext:mlkit", "ML Kit OCR", "Chạy trên máy; chạy được cả khi bật chế độ máy bay.", "", "văn bản trong ảnh"),
            (L + "data/adapters/remote_medicine_catalog.dart", "Tải danh mục thuốc", "Lấy danh mục một lần (cho khớp tên trên máy).", "", "List<Medicine>"),
            ("api:GET /v1/medicines", "GET /v1/medicines", "Danh mục thuốc phân trang (công khai).", "?limit&offset", "{results[], total}"),
        ],
    },
    {
        "id": "reminder", "title": "Nhắc uống thuốc", "icon": "⏰",
        "summary": "Tạo lịch → lưu SQLite trên máy → đặt báo thức hằng tuần → thông báo đúng giờ.",
        "steps": [
            ("ext:user", "Tạo lịch nhắc", "Tên thuốc, liều, giờ, các ngày trong tuần.", "vào: medicine, hour:minute, days", ""),
            (L + "screens/reminder_screen.dart", "Màn hình Nhắc thuốc", "Hộp thoại Đặt giờ nhắc.", "", ""),
            (L + "core/providers.dart", "ReminderController", "Lưu DB rồi lập lịch thông báo; làm mới danh sách.", "", ""),
            (L + "data/local/health_database.dart", "HealthDatabase", "Bảng medication_reminders, adherence_log.", "", ""),
            ("ext:local_db", "health_profile.db", "SQLite trên máy; không rời máy.", "", ""),
            (L + "data/local/notification_service.dart", "NotificationService", "nextReminderTime + zonedSchedule mỗi ngày 1 báo thức (lặp theo tuần). Tên múi giờ dạng UTC+07:00 để Java hiểu.", "hour, minute, weekdays", "7 báo thức"),
            ("ext:notif", "AlarmManager + Thông báo", "Nổ đúng giờ; dựng lại sau khi khởi động máy.", "", "thông báo “Nhắc uống thuốc”"),
            (L + "main.dart", "Khởi động app", "rescheduleAll: dựng lại mọi báo thức mỗi lần mở app.", "", ""),
        ],
    },
    {
        "id": "health", "title": "Hồ sơ sức khoẻ & đồng ý", "icon": "🩺",
        "summary": "Đồng ý → lưu thuốc đang dùng trên máy → gửi kèm câu hỏi chat để cảnh báo.",
        "steps": [
            (L + "screens/consent_screen.dart", "Màn hình đồng ý", "Hiện ở lần mở đầu; thu hồi được ở Quyền riêng tư.", "", ""),
            (L + "data/local/consent_store.dart", "ConsentStore", "Lưu cờ đồng ý.", "", ""),
            ("ext:prefs", "shared_preferences", "Cờ đồng ý (không phải bí mật).", "", ""),
            (L + "screens/health_profile_screen.dart", "Hồ sơ sức khoẻ", "Dị ứng, bệnh nền, thuốc đang dùng.", "", ""),
            (L + "data/local/health_database.dart", "HealthDatabase", "Bảng health_profile.", "", ""),
            ("ext:local_db", "health_profile.db", "Dữ liệu sức khoẻ chỉ ở trên máy.", "", ""),
            (L + "screens/chat_controller.dart", "Chat đính kèm thuốc đang dùng", "Chỉ gửi DANH SÁCH TÊN thuốc (≤ 20) khi hỏi agent.", "current_medications", ""),
            (S + "orchestrator.py", "Cảnh báo tương tác", "Server KHÔNG lưu danh sách này; chỉ thêm cảnh báo “hãy hỏi dược sĩ” vào câu trả lời.", "", ""),
        ],
    },
    {
        "id": "delete", "title": "Xoá tài khoản & dữ liệu", "icon": "🗑️",
        "summary": "Quyền được quên: xoá phía server và dọn cache trên máy.",
        "steps": [
            (L + "screens/privacy_screen.dart", "Màn hình Quyền riêng tư", "Nhập mật khẩu để xác nhận.", "mật khẩu", ""),
            (L + "core/providers.dart", "AuthController.deleteAccount", "Gọi cổng rồi dọn cache/token.", "", ""),
            (L + "data/adapters/remote_auth_repository.dart", "Adapter HTTP", "POST xoá tài khoản.", "", ""),
            ("api:POST /v1/auth/delete-account", "POST /v1/auth/delete-account", "Cần token + mật khẩu; có giới hạn tần suất.", "{password}", "204"),
            (S + "routers/auth.py", "Router auth", "Kiểm tra mật khẩu rồi xoá.", "", ""),
            (S + "sqlite_repository.py", "Xoá dữ liệu người dùng", "users, history, refresh_tokens (ghi vào audit không có thông tin cá nhân).", "", ""),
            (S + "sessions.py", "Xoá phiên chat", "Dọn hội thoại trong RAM.", "", ""),
            (L + "data/adapters/secure_token_store.dart", "Xoá token trên máy", "Đăng xuất cũng xoá cache cục bộ.", "", ""),
        ],
    },
    {
        "id": "import", "title": "Nạp dữ liệu thuốc thật", "icon": "📥",
        "summary": "CSV → chạy thử (chỉ số liệu) → ghi SQLite → server tự thấy dữ liệu mới.",
        "steps": [
            ("ext:csv", "File CSV của bạn", "Cột mặc định: Medicine_Name, Composition, Uses, Side_effects, Manufacturer.", "", ""),
            ("tools/import_real_data.ps1", "import_real_data.ps1", "Mặc định CHẠY THỬ; thêm -Apply mới ghi.", "", ""),
            (S + "import_real.py", "app.import_real", "Chỉ in số liệu; sao lưu .db trước khi ghi; -Authoritative cần -ReviewedBy.", "", "báo cáo số liệu"),
            (S + "ingest.py", "load_csv_mapped", "Ánh xạ cột, loại dòng trùng/không tên, cắt ô quá dài, ô rỗng → None.", "", "IngestedDrug[]"),
            (S + "sqlite_repository.py", "add_medicines / delete_by_source", "Ghi bảng medicines; tăng _med_gen; PRAGMA data_version đổi.", "", ""),
            ("ext:server_db", "data/real/medassist.real.db", "CSDL thật, tách khỏi CSDL demo; git bỏ qua.", "", ""),
            (S + "sqlite_repository.py", "Chỉ mục tự làm mới", "Server đang chạy tự thấy dữ liệu do tiến trình khác nạp (data_version).", "", ""),
        ],
    },
    {
        "id": "demo", "title": "Khởi động demo", "icon": "🚀",
        "summary": "demo_server.ps1 + demo_app.ps1: server cục bộ, máy ảo và app.",
        "steps": [
            ("tools/demo_server.ps1", "demo_server.ps1", "Khoá GIẢ, chỉ nghe 127.0.0.1; -Llm nạp sẵn model; -Real dùng CSDL thật (cần khoá thật).", "", ""),
            (S + "seed.py", "Nạp 10 thuốc mẫu", "Chỉ khi bảng thuốc còn rỗng.", "", ""),
            ("ext:server_db", "medassist.dev.db", "", "", ""),
            ("tools/demo_app.ps1", "demo_app.ps1", "Mở máy ảo, chờ khởi động xong, cài/mở app (mặc định không build lại).", "", ""),
            ("ext:adb", "adb", "", "", ""),
            (L + "main.dart", "App MedAssist Dev", "Gọi server qua 10.0.2.2:8000 (địa chỉ máy chủ nhìn từ máy ảo).", "", ""),
        ],
    },
]


# ------------------------------------------------------------------- xây dữ liệu

def main() -> int:
    paths = files()
    rels = {p: p.relative_to(ROOT).as_posix() for p in paths}
    module_map: dict[str, str] = {}
    for p, rel in rels.items():
        if rel.startswith(f"{SERVER}/app/") and p.suffix == ".py":
            mod = rel[len(f"{SERVER}/"):-3].replace("/", ".")
            module_map[mod] = rel
            if mod.endswith(".__init__"):
                module_map[mod[: -len(".__init__")]] = rel
    ps1_names = {Path(r).name: r for r in rels.values() if r.endswith(".ps1")}

    nodes: dict[str, dict] = {}
    parsed: dict[str, dict] = {}
    for p, rel in rels.items():
        if p.name == "__init__.py":
            continue
        if p.suffix == ".dart":
            info = parse_dart(p, rel)
        elif p.suffix == ".py":
            info = parse_python(p, rel, module_map)
        else:
            info = parse_ps1(p, rel, module_map, ps1_names)
        lane, kind, group = classify(rel)
        parsed[rel] = info
        nodes[rel] = {
            "id": rel, "label": Path(rel).name, "lane": lane, "kind": kind, "group": group,
            "lines": info["lines"], "desc": info["desc"], "symbols": info["symbols"][:40],
            "routes": info.get("routes", []), "tables": info.get("tables", []),
            "env": info.get("env", []), "pkgs": info.get("ext_pkgs", []),
            "lang": {"dart": "Dart", "py": "Python", "ps1": "PowerShell"}[p.suffix[1:]],
        }

    # Nhãn trùng (vd routers/agent.py và agent.py) -> thêm tên thư mục cha cho khỏi nhầm.
    label_count: dict[str, int] = defaultdict(int)
    for n in nodes.values():
        label_count[n["label"]] += 1
    for rel, n in nodes.items():
        if label_count[n["label"]] > 1:
            n["label"] = f"{Path(rel).parent.name}/{n['label']}"

    # bảng SQLite trong Dart (health_database)
    for rel, info in parsed.items():
        if rel.endswith(".dart") and "CREATE TABLE" in info["src"]:
            for tm in re.finditer(r"CREATE TABLE (\w+)\s*\((.*?)\n\s*\)", info["src"], re.S):
                cols = []
                for ln in tm.group(2).splitlines():
                    ln = ln.strip().rstrip(",")
                    cm = re.match(r"^(\w+)\s+(INTEGER|TEXT|REAL)\b(.*)$", ln)
                    if cm:
                        cols.append(f"{cm.group(1)} {cm.group(2)}{' ' + cm.group(3).strip() if cm.group(3).strip() else ''}")
                nodes[rel]["tables"].append({"name": tm.group(1), "cols": cols, "where": "device"})

    for ext_id, lane, label, desc in EXTERNALS:
        nodes[ext_id] = {"id": ext_id, "label": label, "lane": lane, "kind": "external", "group": "external",
                         "lines": 0, "desc": desc, "symbols": [], "routes": [], "tables": [], "env": [],
                         "pkgs": [], "lang": ""}
    # gắn bảng vào nút lưu trữ để hiện ở panel
    to_attach = []
    for rel, n in nodes.items():
        if rel.startswith("ext:"):
            continue
        for t in n["tables"]:
            to_attach.append(("ext:server_db" if t["where"] == "server" else "ext:local_db", {**t, "defined_in": rel}))
    for target, t in to_attach:   # thêm SAU khi duyệt xong (không sửa danh sách đang duyệt)
        nodes[target]["tables"].append(t)

    # endpoint
    api_prefix = "/v1"
    mm = re.search(r'API_PREFIX\s*=\s*"([^"]+)"', parsed.get(f"{SERVER}/app/main.py", {}).get("src", ""))
    if mm:
        api_prefix = mm.group(1)
    endpoints: dict[str, dict] = {}
    edges: list[dict] = []
    for rel, n in list(nodes.items()):
        for r in n["routes"]:
            key = f"api:{r['method']} {api_prefix}{r['path']}"
            endpoints[key] = {**r, "path": api_prefix + r["path"], "router": rel}
            nodes[key] = {"id": key, "label": f"{r['method']} {api_prefix}{r['path']}", "lane": "api", "kind": "endpoint",
                          "group": "api", "lines": 0,
                          "desc": r["summary"] or r["doc"] or r["handler"], "symbols": [], "routes": [r],
                          "tables": [], "env": [], "pkgs": [], "lang": "",
                          "auth": r["auth"], "handler": r["handler"], "router": rel}
            edges.append({"s": key, "t": rel, "kind": "handled_by", "label": f"hàm {r['handler']}()"})

    # cạnh import
    for rel, info in parsed.items():
        for tgt in info["imports"]:
            if tgt in nodes and tgt != rel:
                edges.append({"s": rel, "t": tgt, "kind": "import", "label": "import"})

    # cạnh tới nút ngoài (tự nhận bằng regex)
    for rel, info in parsed.items():
        for rx, ext_id, label, scope in EXT_PATTERNS:
            if re.search(scope, rel) and re.search(rx, info["src"]):
                edges.append({"s": rel, "t": ext_id, "kind": "uses", "label": label})
    problems: list[str] = []
    for rel, ext_id, label in EXT_MANUAL:
        if rel not in nodes:
            problems.append(f"EXT_MANUAL: không thấy file {rel}")
            continue
        edges.append({"s": rel, "t": ext_id, "kind": "uses", "label": label})

    # cạnh gọi API (adapter → endpoint): đọc literal '/v1/...' trong Dart + ràng buộc đã kiểm tra
    api_bindings = {
        L + "data/adapters/remote_auth_repository.dart": [
            "POST /v1/auth/login", "POST /v1/auth/register", "POST /v1/auth/logout",
            "POST /v1/auth/refresh", "POST /v1/auth/delete-account", "GET /v1/auth/me"],
        L + "data/adapters/session_refresher.dart": ["POST /v1/auth/refresh"],
        L + "data/adapters/remote_medicine_repository.dart": [
            "GET /v1/search", "GET /v1/history", "POST /v1/history", "DELETE /v1/history/{item_id}"],
        L + "data/adapters/remote_medicine_catalog.dart": ["GET /v1/medicines"],
        L + "data/adapters/remote_analyzer.dart": ["POST /v1/analyze"],
        L + "data/adapters/sse_agent_client.dart": ["POST /v1/agent/chat"],
    }
    for rel, eps in api_bindings.items():
        if rel not in nodes:
            problems.append(f"api_bindings: không thấy file {rel}")
            continue
        for ep in eps:
            key = "api:" + ep
            if key not in nodes:
                problems.append(f"api_bindings: không có route thật cho {ep}")
                continue
            edges.append({"s": rel, "t": key, "kind": "calls", "label": "gọi HTTP"})
    for rel, info in parsed.items():
        if rel.endswith(".dart"):
            for lit in set(re.findall(r"'(/v1/[\w/\-{}]+)'", info["src"])):
                for key in nodes:
                    if key.startswith("api:") and key.endswith(" " + lit) and rel not in (
                            L + "core/config/api_paths.dart", L + "core/network/api_client.dart"):
                        edges.append({"s": rel, "t": key, "kind": "calls", "label": "gọi HTTP"})

    # khử trùng cạnh
    seen = set()
    uniq = []
    for e in edges:
        k = (e["s"], e["t"], e["kind"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    edges = uniq

    # ---- luồng: kiểm tra với mã thật
    flows = []
    for f in FLOWS:
        steps = []
        for node, title, detail, data_in, data_out in f["steps"]:
            if node not in nodes:
                problems.append(f"luồng '{f['id']}': không thấy node {node}")
                continue
            steps.append({"node": node, "title": title, "detail": detail, "in": data_in, "out": data_out})
        flows.append({"id": f["id"], "title": f["title"], "icon": f["icon"], "summary": f["summary"], "steps": steps})

    # ---- thư viện
    deps = []
    pub = ROOT / "pubspec.yaml"
    if pub.exists():
        section = None
        for ln in pub.read_text(encoding="utf-8").splitlines():
            if re.match(r"^(dependencies|dev_dependencies):", ln):
                section = ln.split(":")[0]
                continue
            if re.match(r"^\S", ln):
                section = None
            mm2 = re.match(r"^  ([a-z_0-9]+):\s*([^\s#]*)\s*(?:#\s*(.*))?$", ln)
            if section and mm2 and mm2.group(1) not in ("flutter", "flutter_test", "flutter_localizations", "sdk"):
                deps.append({"name": mm2.group(1), "ver": mm2.group(2), "note": (mm2.group(3) or "").strip(),
                             "kind": "Flutter · " + ("dev" if section == "dev_dependencies" else "chạy"),
                             "ecosystem": "dart"})
    req = ROOT / SERVER / "requirements.txt"
    pyimport = {"PyJWT": "jwt", "mcp": "mcp", "httpx-sse": "httpx_sse", "pyyaml": "yaml"}
    if req.exists():
        for ln in req.read_text(encoding="utf-8").splitlines():
            mm2 = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*([<>=!~][^\s#]*)?\s*(?:#\s*(.*))?$", ln.strip())
            if mm2 and not ln.strip().startswith("#"):
                name = mm2.group(1)
                deps.append({"name": name, "ver": mm2.group(2) or "", "note": (mm2.group(3) or "").strip(),
                             "kind": "Python", "ecosystem": "python", "import": pyimport.get(name, name.lower().replace("-", "_"))})
    for d in deps:
        users = []
        for rel, n in nodes.items():
            if d["ecosystem"] == "dart" and d["name"] in n["pkgs"]:
                users.append(rel)
            if d["ecosystem"] == "python" and d.get("import") in n["pkgs"]:
                users.append(rel)
        d["used_by"] = users

    # bảng: ai dùng
    for n in nodes.values():
        for t in n["tables"]:
            t["used_by"] = [rel for rel, info in parsed.items()
                            if re.search(rf"\b{t['name']}\b", info["src"]) and rel != t.get("defined_in", rel)
                            and rel in nodes]
    stats = {
        "files": len(parsed),
        "lines": sum(i["lines"] for i in parsed.values()),
        "dart": sum(1 for r in parsed if r.endswith(".dart")),
        "python": sum(1 for r in parsed if r.endswith(".py")),
        "ps1": sum(1 for r in parsed if r.endswith(".ps1")),
        "edges": len(edges), "endpoints": len(endpoints), "flows": len(flows),
        "tables": sum(len(n["tables"]) for k, n in nodes.items() if not k.startswith("ext:")),
    }

    # chỉ giữ nút có liên quan (không bỏ nút nào; file không có cạnh vẫn hiện)
    data = {"lanes": [{"id": i, "title": t} for i, t in LANES], "nodes": list(nodes.values()), "edges": edges,
            "flows": flows, "deps": deps, "stats": stats, "project": "MedAssist"}

    if problems:
        print("⚠️ CÓ VẤN ĐỀ KHI ĐỐI CHIẾU LUỒNG VỚI MÃ THẬT:")
        for p in problems:
            print("  -", p)
        return 2

    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"OK: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  {stats['files']} file ({stats['dart']} Dart, {stats['python']} Python, {stats['ps1']} PowerShell), "
          f"{stats['lines']} dòng | {len(nodes)} nút, {stats['edges']} liên kết, {stats['endpoints']} endpoint, "
          f"{stats['tables']} bảng, {stats['flows']} luồng")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
