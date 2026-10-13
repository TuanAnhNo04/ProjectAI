# QUYẾT ĐỊNH BẢO MẬT — MedAssist

> **Mục đích:** ghi lại các quyết định bảo mật đã **chốt** với chủ dự án, để người
> (hoặc phiên AI) đọc lại sau không vô tình phá vỡ chúng.
>
> Nguồn chuẩn vẫn là [`../AGENTS.md`](../AGENTS.md). File này ghi **cách áp
> dụng cụ thể trong code** và các điểm đã *cố ý* chưa làm.
>
> Cập nhật: 03/10/2026 (sau Day 19)

---

## D1 — LLM chỉ là MOCK, sau interface; provider ngoài mặc định TẮT

**Đã chốt.** Vì sao: `AGENTS.md` mục 2 — adapter gửi dữ liệu ra ngoài chỉ được
viết khi có **cả ba**: (a) consent rõ ràng trong UI, (b) khử định danh,
(c) `docs/PRIVACY.md`. Chưa đủ ba thì *"write mocks only; do NOT write adapters
that send real data"*.

Cụ thể:
- Mọi LLM đi qua interface `LLMProvider` (Day 22).
- Chỉ có `MockLLMProvider` (xác định, không mạng).
- **Không có** API key trong code, **không có** adapter thật.
- Provider ngoài (nếu sau này viết) **mặc định tắt**, chỉ chạy khi bật bằng cờ
  môi trường **và** chỉ đi qua **module ẩn danh** (Day 24).

> ⛔ Không được viết adapter thật khi chưa đủ consent + khử định danh + PRIVACY.md.

## D2 — KHÔNG cài package; cần gì thì liệt kê và DỪNG

**Đã chốt.** `AGENTS.md` mục 2 cấm tự thêm `pip`/`pub` package.
Khi cần package mới: **liệt kê tên + lý do rồi dừng**, chờ chủ dự án tự cài.

Máy dev **đã có sẵn**: `fastapi`, `uvicorn`, `bcrypt`, `httpx`, `pytest`, `yaml`,
`numpy`, `sklearn`, `PyJWT` (thêm 05/10/2026, **đã được chủ dự án duyệt**).
Còn thiếu: `redis`, `asyncpg`, `pgvector`, `sqlalchemy` → **không được tự cài**.

## D3 — KHÔNG bịa dữ liệu y khoa / tương tác thuốc

**Đã chốt.** `AGENTS.md` mục 3: mọi khẳng định y khoa phải có nguồn truy được.

Cụ thể:
- Tên thuốc / hoạt chất chỉ đọc từ file **cục bộ** (`data/medicines.dev.csv`).
  Dòng thiếu tên → **bỏ**, không tự đặt.
- `source` trỏ về **đúng file đã nạp** (`dev-sample: medicines.dev.csv`) —
  **không** gán tên cơ quan lâm sàng.
- Mỗi bản ghi có **`is_authoritative=False`** → khi cờ này `False`, agent **phải
  từ chối** trả lời y khoa và khuyên hỏi dược sĩ/bác sĩ.
- **Tương tác thuốc: cố ý KHÔNG dựng bảng.** `has_interaction_data()` luôn trả
  `False` — không có nguồn để kiểm chứng thì không được bịa.

## D4 — Log KHÔNG chứa nội dung nhạy cảm

**Đã chốt.** `AGENTS.md` mục 3: không log `Authorization`, mật khẩu, email, nội dung
chat, prompt, tên thuốc.

Log chỉ ghi: **`trace_id`**, `latency_ms`, `error_code`, `route`, `user_id` (đã
mã hoá/pseudonymize nếu cần đếm theo người dùng).
⛔ Không ghi: nội dung tin nhắn, prompt, tên thuốc, đường dẫn ảnh.

## D5 — Memory/SSE theo `user_id`; có endpoint xoá dữ liệu người dùng

**Đã chốt.**
- Mọi bản ghi hội thoại/session gắn `user_id`; đọc và xoá đều lọc theo `user_id`
  lấy từ **token** (chống IDOR — xem `app/routers/data.py`).
- Phải có **test hai người dùng**: A không thấy / không xoá được dữ liệu B.
- Phải có **endpoint xoá toàn bộ dữ liệu của người dùng** (quyền kho dữ liệu /
  "xóa tài khoản").

---

## Điểm CHƯA hoàn thành (nợ kỹ thuật, đã thống nhất là phải sửa)

| # | Vấn đề | Yêu cầu | Trạng thái |
|---|---|---|---|
| N1 | ~~Refresh token **chưa** thu hồi cả chuỗi khi phát hiện tái sử dụng~~ | Thêm `token family`: dùng lại token đã xoay ⇒ thu hồi **cả family** | ✅ **ĐÃ XONG 03/10/2026** — `family_id` + `revoke_token_family()` + migration SQLite |
| N2 | ~~JWT secret **fallback ngẫu nhiên theo tiến trình**~~ | Lấy từ env, **≥ 32 byte**, thiếu/ngắn thì **từ chối khởi động** | ✅ **ĐÃ XONG 03/10/2026** — `load_secret()` + `validate_config()` gọi lúc import `app.main` |
| N3 | ~~bcrypt **cắt cụt** mật khẩu > 72 byte~~ | **Từ chối** mật khẩu > 72 byte, `rounds >= 12` | ✅ **ĐÃ XONG 03/10/2026** — `PasswordTooLongError`; đăng ký trả 422, đăng nhập trả 401 |
| N4 | ~~JWT tự viết tay~~ | Thay bằng **PyJWT** | ✅ **ĐÃ XONG 05/10/2026** — chủ dự án duyệt `pip install "PyJWT>=2.8"`; `jwt.encode`/`jwt.decode` + allowlist `algorithms=["HS256"]` + 11 test mới |

> ⚠️ **Lệch RFC 7519 đã biết:** `sub` của ta là **int** (để `deps.py` dùng làm
> `user_id`), còn `PyJWT` từ 2.10 ném `InvalidSubjectError` nếu `sub` không phải
> chuỗi. Ta tắt `verify_sub` rồi **tự kiểm tra tường minh** trong `decode()`:
> sai kiểu → `None`. Có test `test_sub_phai_la_int` chặn lại.

> ✅ **Test tấn công JWT mới (11 test):** `alg: none` · `HS512` ngoài allowlist ·
> payload vỡ · token thiếu `exp` · chữ ký bị sửa 1 ký tự · `sub` sai kiểu
> (chuỗi / float / bool / thiếu) · token do PyJWT phát ra phải đọc được lại.

> ⚠️ **Đổi `MEDASSIST_SECRET_KEY` sẽ làm mất hiệu lực mọi token cũ.** Đây là hệ quả
> đúng thiết kế (N2), nhưng nghĩ tới trước khi xoay khoá trên môi trường thật.

> ✅ Kiểm chứng N2 ở mức tiến trình: thiếu biến → `ConfigError: Thiếu biến môi
> trường MEDASSIST_SECRET_KEY`; 13 byte → `ConfigError: phải dài ít nhất 32 byte`;
> 64 byte → khởi động OK.
>
> ✅ `python -m pytest` → **97/97 xanh** (72 → 97, thêm 25 test)
>
> ---
>
> ✅ **Day 20–21 xong** — `python -m pytest` → **182/182 xanh**.
>
> **D1 đã hiện thực đúng như ghi:** `app/llm.py` có `LLMProvider` + `MockLLMProvider`
> (tất định, offline). **Không** adapter thật, **không** API key, **không** import
> `requests`/`httpx`/`socket` — có test quét AST để chặn.
>
> **D3 đã hiện thực:** không nguồn → `no_evidence`; provider bịa → `unsupported_claim`;
> nguồn `is_authoritative=False` → disclaimer **bắt buộc** ghi rõ "dữ liệu mẫu".
>
> **D5 (đầy đủ):** `/v1/agent/ask` và `/v1/agent/chat` bắt buộc `Depends(current_user_id)`;
> route **không log gì cả** (đúng hành vi). Danh sách trường log hợp lệ đặt ở
> `app/schemas.py::ALLOWED_LOG_FIELDS` = {`trace_id`, `latency_ms`, `error_code`, `route`}.
> `SessionStore` lọc theo cặp `(user_id, session_id)` và có
> `DELETE /v1/agent/session/{id}` — đây là phần **quyền được quên**.

> ✅ **Day 22–23 xong** — `python -m pytest` → **231/231 xanh**.
>
> **Planner là luật xác định, không phải LLM.** Orchestrator chỉ ghép câu trả lời từ
> `ToolResult`. Nếu để LLM tự chọn tool, ta đã trao cho thứ không kiểm chứng quyền
> quyết định hệ thống đi vào dữ liệu y tế nào — đúng cái lỗ đã chặn ở Day 21.
>
> ⛔ **Giới hạn chưa vượt được, ghi rõ để không ai tưởng đã xong:** `check_interaction`
> luôn từ chối (không có nguồn tương tác); hội thoại nhiều lượt chưa kiểm chứng được vì
> `MockLLMProvider` không dùng lịch sử. Xem `docs/ROADMAP.md` mục Day 22–23.
>
> 📌 **Nền tảng cho D2 (cài package):** cần dùng mấy package sau, theo thứ tự ưu
> tiên. Tôi **không tự cài** — bạn duyệt rồi tôi mới làm:
> | Package | Vì sao cần |
> |---|---|
> | ~~`PyJWT>=2.8`~~ | ~~Thay JWT tự viết tay~~ → ✅ **đã cài 05/10/2026** |
> | `redis` + `asyncpg` | Rate limit / Postgres khi lên máy thật |
> | `pgvector` + Postgres | Nâng retrieval từ từ khoá lên embedding — **cần cài Docker Desktop** |
>
> 🚫 **pgvector đang bị chặn bởi phần cứng, không phải bởi quyết định.** Máy này
> **không có Docker, cũng không có Docker Desktop**. Cài nó là thay đổi hệ thống
> (cần admin, ~5 GB, bật WSL2/Hyper-V) → việc của chủ máy, không phải của tôi.