# Báo cáo kiểm tra trước phát hành — MedAssist (29/10/2026)

Phạm vi: audit chỉ-đọc (không sửa mã nguồn; đã kiểm tra: chỉ file do công cụ sinh ra thay đổi). 3 vòng: (1) 3 agent song song A/B/C, (2) kiểm chứng có mục tiêu, (3) hồi quy + cổng phát hành.

## CẬP NHẬT 30/10/2026 — chạy lại sau các sửa lỗi
Các mục bên dưới là bản audit gốc ngày 29/10 (giữ để truy vết). Trạng thái hiện tại:

| Mã | Trạng thái 30/10 | Bằng chứng |
|---|---|---|
| B-01 Critical | **Đã sửa** | `orchestrator.py::_select_provider_for_request` rơi về mock khi chưa đồng ý; test `tests/test_remediation_b01_b04.py`, `tests/test_diet_and_mode.py` |
| B-02 High | **Đã sửa** | `safety.py` có nhóm tự hại, đột quỵ, hô hấp + biến thể/đảo trật tự; test `tests/test_safety_lifestyle.py` |
| B-03 High→Medium | **Đã sửa một phần** | `deidentify.py` đã thêm BHYT, ngày chữ; vẫn là lưới an toàn, KHÔNG đảm bảo tuyệt đối (tên trần tiếng Việt không bắt được) |
| A-01 High | **Đã sửa** | `app_lock_store.dart`: đếm sai + khoá tạm tăng dần, fail-closed khi kho không đọc được. PIN vẫn lưu nguyên văn trong secure storage (băm cần gói mới, chưa duyệt) |
| A-03 Medium | **Đã sửa** | `providers.dart` gọi `deleteAll()` khi đổi/xoá tài khoản |
| A-05 Low | **Đã sửa** | `main.dart` chỉ `debugPrint` khi `kDebugMode` |
| A-02, A-04, B-04, B-05, C-01..C-03 | **Chưa xác minh lại** | chưa đối chiếu lại trong lần này |

Hồi quy 30/10/2026 (chạy cục bộ, dữ liệu giả, không gọi mạng):

| Bộ test | Kết quả |
|---|---|
| Flutter `analyze` | 1 info cũ (gói `path` chưa khai báo) |
| Flutter `test` | **476/476** pass |
| Backend `pytest` | pass toàn bộ, không lỗi |
| `eval.py` | **130/130** (100% mọi nhóm, gồm injection 15/15, out_of_scope 25/25, smalltalk 21/21) |

Còn chưa kiểm tra: đường gọi Claude thật qua giao diện app (cần khoá API + gửi dữ liệu ra ngoài, chờ chủ dự án), thử trên điện thoại thật, dược sĩ đối chiếu, deploy Railway/HTTPS, SQLite chưa mã hoá.

**Kết luận mới:** buổi demo trên lớp chấp nhận được (cổng đồng ý đã vá). Phát hành cho người dùng/dữ liệu thật vẫn **NO-GO** cho tới khi: có nguồn y khoa đã duyệt, dược sĩ đối chiếu, mã hoá SQLite, HTTPS + keystore thật.

---

## Bản audit gốc 29/10/2026
**Chưa nên phát hành cho người dùng thật / dữ liệu thật (NO-GO)** cho tới khi sửa B-01 (Critical). *(đã sửa, xem trên)*
**Buổi demo trên lớp: chấp nhận được** nếu server demo KHÔNG bật Claude (`MEDASSIST_LLM` không đặt). Nếu bật `-Claude` khi chưa sửa B-01 thì mọi câu hỏi chat đều có thể gửi tới Claude dù người dùng chưa đồng ý.

## Hồi quy (Vòng 3)
| Bộ test | Kết quả |
|---|---|
| Flutter `analyze` | 1 info cũ (gói `path` chưa khai báo) |
| Flutter `test` | 381/381 pass |
| Backend `pytest` | pass, không lỗi |
| `eval.py` | 109/109 |
| Kiểm tra tích hợp (agent C, server cổng 8010, dữ liệu giả) | SSE, lifestyle, news, 401/422/404 khớp bộ đọc Dart |

Các test hiện có KHÔNG bắt được B-01 (xem "Test hồi quy đề xuất").

## Phát hiện nghiêm trọng, đã chẩn đoán (chưa sửa)

### B-01 [Critical] Cổng đồng ý đám mây bị lách
- Luồng: chat với server bật `MEDASSIST_LLM=claude`, người dùng chưa đồng ý (`cloud_ai_consent=false`).
- Đã tái hiện: **Có** (script `verify_c2.py`, khoá giả, không gọi mạng): `configured: claude | selected w/o consent: claude | BYPASS`.
- Nguyên nhân gốc (**Đã xác nhận**): `server/medassist-core/app/orchestrator.py:313` trả `get_llm_provider()` làm "dự phòng", nhưng hàm này đọc biến môi trường nên trả lại chính Claude. Comment ở dòng đó ("luôn là mock") sai. `llm.py::get_llm_provider` và lớp thứ hai cùng dựa vào một nguồn nên không còn "hai lớp độc lập".
- Tác động: vi phạm AGENTS.md §2(a) (đồng ý rõ ràng). Câu hỏi vẫn được khử định danh (nhưng xem B-03), nhưng không có sự đồng ý.
- Khuyến nghị sửa: ở nhánh chưa đồng ý trả thẳng `MockLLMProvider` (instance cố định `_default_provider` trong `app/llm.py`), không gọi lại hàm đọc env.
- Test hồi quy đề xuất: đặt `MEDASSIST_LLM=claude` + khoá giả, gọi `_select_provider_for_request(get_llm_provider(), False)` và qua route chat; khẳng định `.name == "mock"` và `generate()` của Claude không bao giờ được gọi (MockTransport đếm 0 request).
- Đã chạy hồi quy sau sửa: **Chưa** (chưa có giai đoạn sửa).

### B-02 [High] Từ khoá cấp cứu thiếu
- Đã tái hiện: **Có**. KHÔNG phát hiện: "tôi muốn chết", "nghi đột quỵ", "tôi thở khó", "không thở được". (Phát hiện đúng: "dau nguc", "tự tử"; không báo nhầm "ngạt mũi", "buồn nôn nhẹ".) Lưu ý: B báo "tự tử" bị bỏ sót nhưng kiểm tra lại thì không.
- Nguyên nhân gốc (**Đã xác nhận ở mức triệu chứng; gốc là danh sách cố định**): `app/safety.py::detect_red_flags` khớp theo danh sách từ khoá cố định, không có biến thể ("thở khó", "không thở được") và nhóm tự hại/đột quỵ.
- Khuyến nghị: bổ sung biến thể + nhóm tự hại (hiện kèm số hỗ trợ phù hợp) + đột quỵ (FAST: "méo miệng", "yếu nửa người" đã có); thêm test bảng.
- Test hồi quy: bảng câu thật (có/không dấu, đảo trật tự) → khẳng định có `emergency`; và nhóm câu vô hại → không `emergency`.
- Đã chạy hồi quy sau sửa: Chưa.

### B-03 [High, có điều kiện] Khử định danh còn hở
- Đã tái hiện: **Có**. Không bị che: ngày viết chữ ("5 tháng 3 năm 1980"), ngày ISO, địa chỉ, số BHYT dạng "DN4797912345678"; tên chỉ che một phần ("Lê Thị B" → "<NAME> B").
- Nguyên nhân gốc (**Đã xác nhận**): `app/deidentify.py` dùng regex theo mẫu có sẵn, chưa phủ các dạng trên. Regex không thể đảm bảo khử định danh đầy đủ.
- Điều kiện: mức High chỉ khi B-01 còn (gửi mà không có đồng ý). Sau khi sửa B-01, đây thành rủi ro Medium (chỉ ảnh hưởng người đã đồng ý) và cần nói rõ trong màn đồng ý rằng khử định danh là "hết sức" chứ không tuyệt đối.
- Khuyến nghị: thêm mẫu ngày chữ/ISO, BHYT chữ+số, địa chỉ (số nhà + đường/quận), tên viết tắt sau họ tên; thêm test bảng.
- Đã chạy hồi quy sau sửa: Chưa.

### A-01 [High → Medium khi demo] Khoá PIN yếu
- Chưa tái hiện bằng test; đọc code (**Giả thuyết có căn cứ**, mã do tôi viết): không giới hạn số lần thử (`lock_screen.dart`), PIN lưu nguyên văn trong secure storage (`app_lock_store.dart`), lỗi đọc secure storage làm `hasPin()` trả `false` nên mở khoá (fail-open).
- Khuyến nghị: đếm sai + khoá tạm tăng dần, lưu băm có muối (cần `crypto` hoặc `cryptography` — phải xin duyệt gói) HOẶC giữ nguyên văn nhưng ghi rõ mô hình đe doạ; khi lỗi đọc thì coi như CÓ PIN (fail-closed) và yêu cầu đăng nhập lại.
- Test hồi quy: widget test 5 lần sai → bị khoá tạm; store lỗi → vẫn khoá.

## Phát hiện Medium / Low (rút gọn)
- **A-02 [Medium]** Lời khuyên gửi tên thuốc lên server của bạn kể cả khi `cloud_ai_consent=false` (server ở Railway = ngoài máy). Không phải lách cổng Claude, nhưng cần nói rõ trong màn đồng ý/PRIVACY.md rằng tên thuốc đi tới server của MedAssist. Xem `lifestyle_tip.dart:128-136`.
- **A-03 [Medium]** Đăng xuất không xoá DB sức khoẻ trên máy; dùng chung máy, người sau thấy thuốc người trước. Xoá tài khoản không xoá PIN. (`providers.dart` ~499-536; đã đọc xác nhận `logout()` không gọi `deleteAll`). Quyết định thiết kế: gắn dữ liệu theo user hoặc hỏi "xoá dữ liệu trên máy?" khi đăng xuất.
- **A-04 [Low]** `doseStatusForTaken` tính theo giờ hẹn của HÔM NAY (liều 23:30 ghi lúc 00:30 hôm sau không thành `late`); `logDose` không chặn ghi trùng; `consumePills` có thể đua. (`providers.dart:1170-1212`, đã đọc.)
- **A-05 [Low]** `debugPrint` URL API chạy cả bản release (`main.dart:16`).
- **B-04 [Medium]** `/docs` luôn bật; Dockerfile chạy bằng root. **B-05 [Low]** vài điểm lạ ở allowlist tin tức (host vẫn an toàn).
- **C-01..C-03 [Low]** trả lời `no_data` chưa nhắc hỏi dược sĩ/bác sĩ; câu trả lời lặp lại truy vấn đã bỏ dấu; `tool_call.arguments` là chuỗi rỗng (app chịu được).

## Đã đạt
Token lưu ở secure storage, redaction log, FLAG_SECURE, `openUrl` chỉ https, consent đám mây mặc định TẮT, migration DB v1→v3 và `deleteAll`, parser đơn thuốc bỏ dòng thông tin cá nhân, allowlist tin tức chống lookalike/userinfo/port/redirect, lifestyle không lưu dữ liệu.

## Chưa kiểm tra được
Đường gọi Claude thật (cần khoá); `/history` authz (chưa soi riêng); relock và `dose_actions` trên máy ảo; màn Tin tức/Cơ sở y tế/Tiến độ/Chat trên máy ảo; PRIVACY.md và DEPLOY_RAILWAY.md chưa đối chiếu từng câu.

## Thứ tự sửa đề xuất (giai đoạn sửa riêng, cần bạn cho phép)
1. B-01 (một dòng + test) → 2. B-02/B-03 (mở rộng danh sách/regex + test bảng) → 3. A-03/A-01 → 4. còn lại.
