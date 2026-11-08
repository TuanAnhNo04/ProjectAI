# QA PLAN — MedAssist pre-release audit (nguồn sự thật duy nhất)

Điều phối: agent chính. Mọi agent con ĐỌC file này trước, chỉ làm trong phạm vi của mình.

## Quy tắc cứng
- **KHÔNG sửa mã nguồn app**: không ghi vào `lib/`, `test/`, `android/`, `server/medassist-core/app/`, `server/medassist-core/tests/`, `pubspec.yaml`, `assets/`. Chỉ được ghi vào `docs/qa/` (file findings của mình) và thư mục scratchpad (script thử tạm, DB tạm).
- Bỏ qua bản lồng `ProjectAI-main/ProjectAI-main/`. KHÔNG mở `anh_thuoc/`, `server/medassist-core/data/real/`, `.env*`, khoá, `uploads/`.
- Không gọi mạng ngoài `localhost`/`127.0.0.1`/`10.0.2.2`. Không `git`. Không cài gói. Dữ liệu thử chỉ là dữ liệu GIẢ.
- Python: `& "C:\Users\Tuan Anh\AppData\Local\Temp\..."` KHÔNG; dùng `C:\Users\Tuan Anh\AppData\Local\Programs\Python\Python312\python.exe`. Flutter: `$env:Path = "D:\flutter_windows_3.19.4-stable\flutter\bin;$env:Path"`. adb: `D:\Android\Sdk\platform-tools`.
- Chỉ chạy bộ test hiện có MỘT lần (không lặp). Không chạy lại toàn bộ chỉ vì thấy một lỗi.
- Hiệu quả token: đọc mỗi file một lần, trích dẫn `file:dòng`, không dán log dài (≤10 dòng/bằng chứng).

## Phạm vi (không chồng nhau)
| Agent | Phạm vi | File kết quả |
|---|---|---|
| A — App Flutter | `lib/**`, `android/app/**`, `pubspec.yaml`, `test/**` (chỉ đọc + chạy analyze/test) | `docs/qa/findings_app.md` |
| B — Server | `server/medassist-core/app/**`, `tests/**`, Dockerfile, railway.json, docs/PRIVACY.md, DEPLOY_RAILWAY.md | `docs/qa/findings_server.md` |
| C — Tích hợp/runtime | hợp đồng app↔server (SSE `emergency`, `/v1/agent/lifestyle`, `/v1/news`, `/v1/agent/chat`), smoke trên máy ảo | `docs/qa/findings_integration.md` |

Cùng một phát hiện đã có ID ở file khác → chỉ ghi `Xem <ID>`, không lặp lại.

## Ma trận kiểm tra (mỗi agent chỉ làm dòng của mình)
**A (App):** AGENTS.md §3 (log lộ Authorization/email/tên thuốc/đường dẫn ảnh/nội dung chat; token qua shared_preferences; cleartext/`badCertificateCallback`/`logTraffic` ở release; khoá/URL cứng); quyền Android (chỉ nên thêm vị trí); consent đám mây mặc định TẮT và thực sự chặn `cloud_ai_consent`/`health_context`; khoá PIN (relock, bypass, brute-force, PIN lưu ở đâu); FLAG_SECURE; migration DB v1→v2→v3 (dữ liệu cũ không mất, cột mặc định); logic liều (taken/late/skipped/snoozed, ranh giới ngày/múi giờ, streak, trừ số viên không âm, trùng liều); parser đơn thuốc (nhiễu OCR, dòng thông tin cá nhân phải bị bỏ, không lưu text OCR/ảnh); `openUrl` chỉ https; xoá dữ liệu/đăng xuất xoá hết (bảng mới, cache lifestyle, consent); chạy `flutter analyze` + `flutter test` một lần.
**B (Server):** authn/authz theo `user_id` trên mọi `/history` và `/agent`; cổng đồng ý đám mây hai lớp (cờ server + `cloud_ai_consent`) không thể lách (kể cả `MEDASSIST_LLM=claude` mà consent=false); khử định danh VN (SĐT, CCCD, email, ngày, tên có/không dấu, NFD); từ khoá cấp cứu (false negative nguy hiểm, false positive vô hại); health_context/`recent_topics` như dữ liệu không tin cậy (prompt injection); lifestyle/chat không lưu/ghi log nội dung; allowlist tin tức (lookalike, userinfo, port, redirect, `javascript:`); cấu hình production (SECRET_KEY yếu, CORS, /docs lộ, bind 0.0.0.0, volume DB); ranh giới kích thước/độ dài, lỗi trả ra không lộ nội bộ; chạy `pytest -q` + `eval.py` một lần.
**C (Tích hợp):** dựng server cục bộ riêng (cổng 8010, DB tạm trong scratchpad, khoá giả ≥32 ký tự, mock LLM), tạo user giả bằng API, rồi so khớp phản hồi thật với parser Dart (đọc code Dart tương ứng): SSE `emergency`/`token`/`citation`/`done`, `/v1/agent/lifestyle`, `/v1/news` (+`verified_sources`), lỗi 401/422/404. Smoke trên máy ảo (nếu đang chạy; FLAG_SECURE chặn ảnh chụp → dùng `uiautomator dump`): mở app, khoá PIN, Trang chủ, Nhắc thuốc, Tiến độ, Quyền riêng tư, Tin tức, cơ sở y tế. Không chạm server demo đang chạy ở cổng 8000.

## Định dạng mỗi phát hiện (ngắn)
```
### <A|B|C>-NN  [Critical|High|Medium|Low]  <tiêu đề>
- Thành phần/luồng:
- Tái hiện: <test nhỏ nhất / lệnh>  → Đã tái hiện: Có/Không/Chưa
- Nguyên nhân gốc: <file:dòng> — Độ tin cậy: Đã xác nhận/Giả thuyết (thiếu bằng chứng gì)
- Bằng chứng: <≤10 dòng>
- Khuyến nghị sửa:
- Test hồi quy đề xuất:
```
Chỉ Critical/High cần đủ các mục. Medium/Low: 2 dòng. Gom các lỗi cùng nguyên nhân gốc thành MỘT phát hiện.

## Khi xong
Cuối file findings: bảng đếm theo mức độ + "Chưa kiểm tra được vì…". Báo cáo cho điều phối ≤120 từ (chỉ số phát hiện mới, blocker). Không tạo agent con.
