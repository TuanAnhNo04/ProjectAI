#!/usr/bin/env python
"""Sinh bộ file Postman đầy đủ cho MedAssist Core → `docs/postman/`.

    python tools/postman/build_collection.py                 # chạy với server ở http://127.0.0.1:8000
    python tools/postman/build_collection.py --no-examples   # không cần server (bỏ ví dụ phản hồi)

Ba file được tạo (nhập vào Postman bằng nút **Import**):

- `MedAssist-Local.postman_collection.json`   — toàn bộ request (mọi endpoint trong openapi.json),
  mỗi request có test + ví dụ phản hồi (dữ liệu GIẢ, token đã che);
- `MedAssist-Local.postman_environment.json`  — biến môi trường (baseUrl, tài khoản demo…);
- `search-terms.json`                         — dữ liệu cho Collection Runner (tìm hàng loạt).

## An toàn (AGENTS.md)

- Chỉ gọi **localhost** (script từ chối địa chỉ khác); không gửi gì ra ngoài.
- Dùng tài khoản TẠM (tên ngẫu nhiên) và **xoá** nó ở cuối; mật khẩu/email đều là dữ liệu giả.
- Ví dụ phản hồi được **che** `access_token`/`refresh_token`/chuỗi dạng JWT trước khi lưu.
- Danh sách tên thuốc trong `search-terms.json` là tên của bộ **thuốc mẫu giả** (10 thuốc dev).
- Kiểm tra độ đầy đủ: mọi (phương thức, đường dẫn) trong `openapi.json` phải có request, nếu không
  script báo lỗi.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import secrets
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "postman"
OPENAPI = ROOT / "server" / "medassist-core" / "openapi.json"
STATUS_TEXT = {200: "OK", 201: "Created", 204: "No Content", 401: "Unauthorized", 404: "Not Found",
               409: "Conflict", 422: "Unprocessable Entity", 429: "Too Many Requests", 501: "Not Implemented"}

SAVE_TOKENS = """
const j = pm.response.json();
pm.collectionVariables.set("accessToken", j.token.access_token);
pm.collectionVariables.set("refreshToken", j.token.refresh_token);
pm.test("Có access_token + refresh_token", function () {
  pm.expect(j.token.access_token).to.be.a("string");
  pm.expect(j.token.refresh_token).to.be.a("string");
});
"""

# ---------------------------------------------------------------- danh sách request
# (folder, name, method, path, status, opts)
#   path: dùng {{biến}} Postman; `oa` = đường dẫn dạng OpenAPI để kiểm tra độ đầy đủ.
#   opts: body, query, auth(False=không cần token), test(đoạn JS thêm), desc, oa, example(False = không lưu ví dụ)
ITEMS: list[tuple[str, str, str, str, int, dict]] = [
    ("1. Health", "Kiểm tra server", "GET", "/v1/health", 200, {"auth": False}),

    ("2. Tài khoản", "Đăng ký", "POST", "/v1/auth/register", 201, {
        "auth": False, "test": SAVE_TOKENS,
        "body": {"username": "{{username}}", "password": "{{password}}", "email": "{{email}}"},
        "desc": "Trả 409 nếu tên đã tồn tại — khi đó dùng 'Đăng nhập'. Tên đăng nhập ≥ 3 ký tự, mật khẩu ≥ 6 ký tự."}),
    ("2. Tài khoản", "Đăng nhập", "POST", "/v1/auth/login", 200, {
        "auth": False, "test": SAVE_TOKENS, "body": {"username": "{{username}}", "password": "{{password}}"},
        "desc": "Ô đăng nhập dùng TÊN ĐĂNG NHẬP, không dùng email. Token được tự lưu vào biến collection."}),
    ("2. Tài khoản", "Kiểm tra độ mạnh mật khẩu", "POST", "/v1/auth/validate-password", 200, {
        "auth": False, "body": {"username": "{{username}}", "password": "{{password}}"}}),
    ("2. Tài khoản", "Thông tin tài khoản (me)", "GET", "/v1/auth/me", 200, {}),
    ("2. Tài khoản", "Đổi refresh token", "POST", "/v1/auth/refresh", 200, {
        "auth": False, "test": SAVE_TOKENS, "body": {"refresh_token": "{{refreshToken}}"},
        "desc": "Refresh token dùng một lần (xoay vòng). Dùng lại token cũ sẽ thu hồi cả phiên."}),

    ("3. Thuốc", "Tìm thuốc theo tên", "GET", "/v1/search", 200, {
        "auth": False, "query": [("name", "Panadol"), ("limit", "10"), ("offset", "0")],
        "test": 'const j = pm.response.json();\npm.test("Có results + total", function () { pm.expect(j).to.have.property("results"); pm.expect(j).to.have.property("total"); });'}),
    ("3. Thuốc", "Tìm thuốc — phân trang", "GET", "/v1/search", 200, {
        "auth": False, "query": [("name", "Paracetamol"), ("limit", "2"), ("offset", "0")],
        "test": 'pm.test("Tối đa 2 kết quả", function () { pm.expect(pm.response.json().results.length).to.be.at.most(2); });',
        "desc": "limit tối đa 100. Dùng offset để lấy trang kế tiếp."}),
    ("3. Thuốc", "Tìm thuốc — không có kết quả", "GET", "/v1/search", 200, {
        "auth": False, "query": [("name", "xyzkhongtontai")],
        "test": 'pm.test("Không có kết quả", function () { pm.expect(pm.response.json().total).to.eql(0); });'}),
    ("3. Thuốc", "Danh mục thuốc (cho OCR)", "GET", "/v1/medicines", 200, {
        "auth": False, "query": [("limit", "50"), ("offset", "0")]}),

    ("4. Lịch sử & ảnh", "Ghi một bản ghi lịch sử", "POST", "/v1/history", 201, {
        "body": {"filepath": "demo-fake-image.jpg", "medicines": []},
        "desc": "filepath chỉ là tên giả; không dùng đường dẫn ảnh thật."}),
    ("4. Lịch sử & ảnh", "Danh sách lịch sử", "GET", "/v1/history", 200, {
        "query": [("limit", "20"), ("offset", "0")],
        "test": 'const h = pm.response.json().history || [];\nif (h.length) { pm.collectionVariables.set("historyId", h[0].id); }',
        "desc": "Lưu id bản ghi đầu tiên vào biến historyId để request 'Xoá' dùng."}),
    ("4. Lịch sử & ảnh", "Xoá một bản ghi lịch sử", "DELETE", "/v1/history/{{historyId}}", 204, {
        "oa": "/v1/history/{item_id}", "desc": "Chạy 'Danh sách lịch sử' trước để có historyId. Trả 404 nếu không phải của bạn."}),
    ("4. Lịch sử & ảnh", "Nhận diện từ ảnh (giữ chỗ — trả 501)", "POST", "/v1/analyze", 501, {
        "desc": "Cố ý trả 501: server không nhận ảnh người dùng. OCR chạy trên điện thoại (ML Kit)."}),

    ("5. Trợ lý AI", "Hỏi đáp — tác dụng phụ", "POST", "/v1/agent/ask", 200, {
        "body": {"question": "Panadol có tác dụng phụ gì", "top_k": 5},
        "test": 'const j = pm.response.json();\npm.test("Có trường answered + disclaimer", function () { pm.expect(j).to.have.property("answered"); pm.expect(j.disclaimer).to.be.a("string"); });',
        "desc": "Với dữ liệu mẫu: hỏi công dụng/tác dụng phụ → trả lời kèm trích dẫn [R1]; mỗi trích dẫn có source."}),
    ("5. Trợ lý AI", "Hỏi đáp — liều dùng (bị từ chối)", "POST", "/v1/agent/ask", 200, {
        "body": {"question": "liều dùng Panadol cho trẻ em"},
        "test": 'const j = pm.response.json();\npm.test("Bị từ chối: out_of_scope", function () { pm.expect(j.answered).to.eql(false); pm.expect(j.refusal_reason).to.eql("out_of_scope"); });',
        "desc": "Thiết kế an toàn: không tư vấn liều dùng, chẩn đoán, tương tác."}),
    ("5. Trợ lý AI", "Chat (SSE streaming)", "POST", "/v1/agent/chat", 200, {
        "body": {"message": "Panadol có công dụng gì", "session_id": "{{sessionId}}", "current_medications": []},
        "test": 'pm.test("Có sự kiện done", function () { pm.expect(pm.response.text()).to.include("event: done"); });',
        "desc": "Phản hồi là luồng SSE: tool_call · citation · token · done. Xem dần ở tab Response."}),
    ("5. Trợ lý AI", "Chat — triệu chứng sau khi uống thuốc", "POST", "/v1/agent/chat", 200, {
        "body": {"message": "Uống Panadol 500mg bị buồn nôn có sao không", "session_id": "{{sessionId}}",
                 "current_medications": ["Panadol"]},
        "test": 'pm.test("Có đối chiếu tác dụng phụ", function () { pm.expect(pm.response.text()).to.include("event: done"); });',
        "desc": "Đối chiếu triệu chứng với tác dụng phụ đã nạp, có trích dẫn; luôn nói rõ KHÔNG khẳng định thuốc là nguyên nhân."}),
    ("5. Trợ lý AI", "Chat — dấu hiệu khẩn cấp", "POST", "/v1/agent/chat", 200, {
        "body": {"message": "tôi bị khó thở và đau ngực", "session_id": "{{sessionId}}", "current_medications": []},
        "test": 'pm.test("Có sự kiện emergency", function () { pm.expect(pm.response.text()).to.include("event: emergency"); });',
        "desc": "Dấu hiệu cấp cứu được xử lý TRƯỚC mọi bước khác, không gọi AI: khuyên gọi 115 + cơ sở y tế gần nhất."}),
    ("5. Trợ lý AI", "Gợi ý lối sống", "POST", "/v1/agent/lifestyle", 200, {
        "body": {"current_medications": ["Panadol"], "conditions": ["tăng huyết áp"], "recent_topics": [],
                 "cloud_ai_consent": False},
        "test": 'const j = pm.response.json();\npm.test("Có tips + disclaimer", function () { pm.expect(j.tips).to.be.an("array"); pm.expect(j.disclaimer).to.be.a("string"); });',
        "desc": "Tối đa 5 gợi ý. Gợi ý theo thuốc/bệnh nền kèm trích dẫn nguồn. Server không lưu, không log request."}),
    ("5. Trợ lý AI", "Số liệu sử dụng", "GET", "/v1/agent/usage", 200, {}),
    ("5. Trợ lý AI", "Xoá phiên hội thoại", "DELETE", "/v1/agent/session/{{sessionId}}", 204, {
        "oa": "/v1/agent/session/{session_id}", "desc": "Chạy 'Chat' trước để phiên tồn tại (không thì 404)."}),

    ("6. Tình huống lỗi", "Đăng nhập sai mật khẩu → 401", "POST", "/v1/auth/login", 401, {
        "auth": False, "body": {"username": "{{username}}", "password": "sai-mat-khau-xyz"}}),
    ("6. Tình huống lỗi", "Đăng ký trùng tên → 409", "POST", "/v1/auth/register", 409, {
        "auth": False, "body": {"username": "{{username}}", "password": "{{password}}", "email": "{{email}}"}}),
    ("6. Tình huống lỗi", "Mật khẩu quá yếu → 422", "POST", "/v1/auth/validate-password", 422, {
        "auth": False, "body": {"username": "abc", "password": "abc"}}),
    ("6. Tình huống lỗi", "Không có token → 401", "GET", "/v1/auth/me", 401, {"auth": False}),
    ("6. Tình huống lỗi", "Hỏi đáp không có token → 401", "POST", "/v1/agent/ask", 401, {
        "auth": False, "body": {"question": "Panadol là gì"}}),
    ("6. Tình huống lỗi", "Tìm thuốc thiếu tham số name → 422", "GET", "/v1/search", 422, {"auth": False}),
    ("6. Tình huống lỗi", "Xoá lịch sử không tồn tại → 404", "DELETE", "/v1/history/999999", 404, {"oa": "/v1/history/{item_id}"}),

    ("7. Chạy hàng loạt (Runner)", "Tìm thuốc theo {{term}}", "GET", "/v1/search", 200, {
        "auth": False, "query": [("name", "{{term}}"), ("limit", "5")], "example": False,
        "test": ('const min = pm.iterationData.get("expect_min");\n'
                 'if (min !== undefined && min !== null) {\n'
                 '  pm.test("Ít nhất " + min + " kết quả cho \\"" + pm.variables.get("term") + "\\"", function () {\n'
                 '    pm.expect(pm.response.json().results.length).to.be.at.least(Number(min));\n'
                 '  });\n}'),
        "desc": "Dùng với Collection Runner: chọn file search-terms.json làm Data. Chạy lẻ thì dùng biến term."}),

    ("8. Tin tức", "Tin tức thuốc & y tế", "GET", "/v1/news", 200, {
        "auth": False,
        "test": 'const j = pm.response.json();\npm.test("Có articles + is_sample", function () { pm.expect(j.articles).to.be.an("array"); pm.expect(j).to.have.property("is_sample"); });',
        "desc": "Server chạy -News: tin thật FDA Recalls (is_sample=false, link https fda.gov). Không -News: bài mẫu, không gọi mạng."}),

    ("9. Dọn dẹp (chạy cuối)", "Đăng xuất", "POST", "/v1/auth/logout", 204, {
        "auth": False, "body": {"refresh_token": "{{refreshToken}}"}}),
    ("9. Dọn dẹp (chạy cuối)", "Xoá tài khoản (KHÔNG hoàn tác)", "POST", "/v1/auth/delete-account", 204, {
        "body": {"password": "{{password}}"},
        "desc": "Xoá tài khoản và toàn bộ dữ liệu. Chỉ chạy khi muốn dọn dữ liệu demo."}),
]

#: Ví dụ phản hồi lấy theo thứ tự chạy này (token tạm, tài khoản tạm).
SEARCH_TERMS = [  # tên của bộ thuốc MẪU GIẢ (data/medicines.dev.csv)
    {"term": "Panadol", "expect_min": 1}, {"term": "Efferalgan", "expect_min": 1},
    {"term": "Hapacol", "expect_min": 1}, {"term": "Para 500", "expect_min": 1},
    {"term": "Decolgen", "expect_min": 1}, {"term": "Amoxicillin", "expect_min": 1},
    {"term": "Strepsils", "expect_min": 1}, {"term": "paracetamol", "expect_min": 2},
    {"term": "dau gio", "expect_min": 1}, {"term": "xyzkhongtontai", "expect_min": 0},
]


# ---------------------------------------------------------------- dựng JSON Postman

def _url(path: str, query=None) -> dict:
    raw = "{{baseUrl}}" + path
    u = {"raw": raw, "host": ["{{baseUrl}}"], "path": [p for p in path.strip("/").split("/") if p]}
    if query:
        u["query"] = [{"key": k, "value": v} for k, v in query]
        u["raw"] = raw + "?" + "&".join(f"{k}={v}" for k, v in query)
    return u


def _request(method, path, o: dict) -> dict:
    r: dict = {"method": method, "header": [], "url": _url(path, o.get("query"))}
    if o.get("desc"):
        r["description"] = o["desc"]
    if "body" in o:
        r["header"].append({"key": "Content-Type", "value": "application/json"})
        r["body"] = {"mode": "raw", "raw": json.dumps(o["body"], ensure_ascii=False, indent=2),
                     "options": {"raw": {"language": "json"}}}
    if o.get("auth") is False:
        r["auth"] = {"type": "noauth"}
    return r


def _test_event(status: int, extra: str = "") -> dict:
    lines = [f'pm.test("Mã trạng thái {status}", function () {{ pm.response.to.have.status({status}); }});']
    if extra:
        lines += extra.strip().split("\n")
    return {"listen": "test", "script": {"type": "text/javascript", "exec": lines}}


def build_collection(examples: dict[int, dict] | None) -> dict:
    folders: dict[str, list] = {}
    for idx, (folder, name, method, path, status, o) in enumerate(ITEMS):
        req = _request(method, path, o)
        item: dict = {"name": name, "event": [_test_event(status, o.get("test", ""))], "request": req}
        ex = (examples or {}).get(idx)
        if ex and o.get("example", True):
            item["response"] = [{
                "name": f"{ex['status']} {STATUS_TEXT.get(ex['status'], '')} — ví dụ (dữ liệu giả, token đã che)",
                "originalRequest": copy.deepcopy(req), "status": STATUS_TEXT.get(ex["status"], ""), "code": ex["status"],
                "_postman_previewlanguage": ex["lang"], "header": [{"key": "Content-Type", "value": ex["ctype"]}],
                "cookie": [], "body": ex["body"]}]
        folders.setdefault(folder, []).append(item)
    return {
        "info": {
            "_postman_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "medassist-core-local-collection")),
            "name": "MedAssist Core (local)",
            "description": ("Bộ request đầy đủ cho backend MedAssist chạy trên máy: http://127.0.0.1:8000.\n\n"
                            "**Chỉ dùng dữ liệu GIẢ.**\n\n"
                            "Cách dùng: Import file này + file environment. Chọn environment \"MedAssist Local\". "
                            "Chạy theo thứ tự từng thư mục (1 → 9): Đăng nhập tự lưu token vào biến collection. "
                            "Để chạy cả bộ: nút Run collection (thư mục 7 dùng file search-terms.json làm Data). "
                            "Thư mục 9 xoá tài khoản — để cuối cùng."),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
        "auth": {"type": "bearer", "bearer": [{"key": "token", "value": "{{accessToken}}", "type": "string"}]},
        "variable": [
            {"key": "baseUrl", "value": "http://127.0.0.1:8000"},
            {"key": "username", "value": "demo"}, {"key": "email", "value": "demo@example.com"},
            {"key": "password", "value": "Demo@12345678"},
            {"key": "accessToken", "value": ""}, {"key": "refreshToken", "value": ""},
            {"key": "sessionId", "value": "demo-session-0001"}, {"key": "historyId", "value": "1"},
            {"key": "term", "value": "Panadol"}],
        "item": [{"name": f, "item": items} for f, items in folders.items()],
    }


def build_environment() -> dict:
    # ⚠️ KHÔNG đưa accessToken/refreshToken vào environment: biến environment đè biến collection,
    # nên token (do script lưu vào biến collection) sẽ bị che bởi giá trị rỗng.
    vals = [("baseUrl", "http://127.0.0.1:8000", "default"), ("username", "demo", "default"),
            ("email", "demo@example.com", "default"), ("password", "Demo@12345678", "secret"),
            ("sessionId", "demo-session-0001", "default")]
    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "medassist-core-local-env")), "name": "MedAssist Local",
            "values": [{"key": k, "value": v, "type": t, "enabled": True} for k, v, t in vals],
            "_postman_variable_scope": "environment",
            "_postman_exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "_postman_exported_using": "Postman/10.0"}


# ---------------------------------------------------------------- chạy thật để lấy ví dụ

JWT_RE = re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]+")


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("<đã che>" if k in ("access_token", "refresh_token") else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, str):
        return JWT_RE.sub("<đã che>", obj)
    return obj


def run_examples(base: str) -> tuple[dict[int, dict], list[str]]:
    host = urlparse(base).hostname
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit("Chỉ được chạy với server trên máy này (127.0.0.1/localhost).")
    user = "pmgen" + secrets.token_hex(4)
    vars_ = {"baseUrl": base, "username": user, "email": f"{user}@example.com", "password": "Postman@12345678",
             "accessToken": "", "refreshToken": "", "sessionId": "pmgen-session-" + secrets.token_hex(3),
             "historyId": "1", "term": "Panadol"}
    sub = lambda s: re.sub(r"\{\{(\w+)\}\}", lambda m: str(vars_.get(m.group(1), m.group(0))), s)
    examples: dict[int, dict] = {}
    problems: list[str] = []
    for idx, (folder, name, method, path, want, o) in enumerate(ITEMS):
        if folder.startswith("7."):
            vars_["term"] = "Panadol"
        url = sub("{{baseUrl}}" + path)
        if o.get("query"):
            from urllib.parse import quote
            url += "?" + "&".join(f"{k}={quote(sub(v))}" for k, v in o["query"])
        data = sub(json.dumps(o["body"])).encode() if "body" in o else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if o.get("auth") is not False and vars_["accessToken"]:
            req.add_header("Authorization", "Bearer " + vars_["accessToken"])
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                code, ctype, body = r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            code, ctype, body = e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8", "replace")
        if code != want:
            problems.append(f"[{folder}] {name}: mong đợi {want}, nhận {code}")
        # cập nhật biến như script Postman
        if code in (200, 201) and '"access_token"' in body:
            t = json.loads(body)["token"]
            vars_["accessToken"], vars_["refreshToken"] = t["access_token"], t["refresh_token"]
        if method == "GET" and path == "/v1/history" and code == 200:
            h = json.loads(body).get("history") or []
            if h:
                vars_["historyId"] = str(h[0]["id"])
        if body.strip():
            if "json" in ctype:
                try:
                    pretty = json.dumps(_redact(json.loads(body)), ensure_ascii=False, indent=2)
                    lang = "json"
                except ValueError:
                    pretty, lang = _redact(body), "text"
            else:
                pretty, lang = JWT_RE.sub("<đã che>", body), "text"
            pretty = pretty.replace(user, "demo")   # tên tài khoản tạm -> "demo" cho ví dụ gọn
            examples[idx] = {"status": code, "ctype": ctype.split(";")[0] or "text/plain", "lang": lang,
                             "body": pretty[:6000]}
    return examples, problems


def check_coverage() -> list[str]:
    spec = json.loads(OPENAPI.read_text(encoding="utf-8"))
    want = {(m.upper(), p) for p, ops in spec["paths"].items() for m in ops}
    have = {(m, o.get("oa", p)) for _, _, m, p, _, o in ITEMS}
    return [f"openapi có {m} {p} nhưng collection chưa có request" for m, p in sorted(want - have)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()

    missing = check_coverage()
    if missing:
        print("THIẾU request:\n  " + "\n  ".join(missing))
        return 2

    examples, problems = None, []
    if not a.no_examples:
        try:
            examples, problems = run_examples(a.base)
        except urllib.error.URLError as e:
            print(f"Không kết nối được {a.base} ({e.reason}). Chạy tools\\demo_server.ps1 hoặc dùng --no-examples.")
            return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "MedAssist-Local.postman_collection.json").write_text(
        json.dumps(build_collection(examples), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "MedAssist-Local.postman_environment.json").write_text(
        json.dumps(build_environment(), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "search-terms.json").write_text(json.dumps(SEARCH_TERMS, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ex = len(examples or {})
    print(f"OK: {len(ITEMS)} request, {n_ex} ví dụ phản hồi → {OUT_DIR.relative_to(ROOT)}")
    for p in problems:
        print("  ⚠️", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
