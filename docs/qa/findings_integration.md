# Findings C — Tích hợp/runtime (app <-> server)

Môi trường: server riêng cổng 8010 (DB tạm, khoá giả, không LLM, seed dev-sample), user giả `qa_user_c`. Đã dừng server khi xong. Cổng 8000 không đụng.

## Kết quả hợp đồng (khớp parser Dart)
| Điểm | Thực tế server | Parser Dart | Kết luận |
|---|---|---|---|
| Auth `/v1/auth/*` | `{"user":..,"token":{access_token,refresh_token,expires_in}}` lồng | `lib/data/models/user.dart:105` xử lý token lồng | Khớp |
| SSE `emergency` | `{"matched":[..],"message":".."}` rồi `token` rồi `done` | `sse_agent_client.dart:~135` | Khớp |
| SSE `token`/`citation`/`done` | `text`; `documentId,quote,title,source,isAuthoritative`; `sessionId` | `sse_agent_client.dart:~119-140` (khoá thừa bị bỏ qua) | Khớp |
| SSE `tool_call` | `arguments` là chuỗi rỗng `""` | `:~123` chỉ nhận Map, khác -> `{}` | Khớp (dung sai) |
| SSE `refusal` | `{"reason":"out_of_scope"\|"no_data"}` + `token` thân thiện | bỏ qua `refusal`, hiện `token` | Khớp |
| `/v1/agent/lifestyle` | tips/disclaimer/generated_by=rules; citations `{id,medicine_name,source,excerpt}` | `lifestyle_tip.dart` | Khớp; consent=true khi không có LLM vẫn trả `rules` |
| `/v1/news` | `articles[].url=""`, `is_sample=true`, `failed_sources`, `verified_sources`, `disclaimer` | `news_article.dart` | Khớp |
| Lỗi | 401 `{code:unauthorized}`, 422 `validation_error` (+detail), 404 `not_found` | — | Nhất quán |

Ràng buộc cần ghi nhớ: `session_id` server yêu cầu 8–64 ký tự (`schemas.py:324`); app sinh id ở `chat_controller.dart` (không kiểm tra độ dài tại đây — chưa xác minh app luôn sinh >=8, Giả thuyết).

## Phát hiện
### C-01  [Low]  Phản hồi `no_data` không khuyên hỏi dược sĩ/bác sĩ
- Luồng: chat, truy vấn không có tài liệu -> token "Không tìm thấy thông tin trong nguồn hiện có về '<...>'." Không có câu "hãy hỏi dược sĩ/bác sĩ" (AGENTS.md §3 yêu cầu). Xem B nếu đã ghi.
- Tái hiện: POST /v1/agent/chat message="xyzzy thuốc không tồn tại abc" -> Đã tái hiện: Có.
### C-02  [Low]  Chuỗi truy vấn bị bỏ dấu khi phản hồi lại người dùng
- `'xyzzy ton abc'` thay vì `'xyzzy thuốc không tồn tại abc'` (dùng chuỗi đã chuẩn hoá để echo). Chỉ thẩm mỹ.
### C-03  [Low]  `tool_call.arguments` là `""` thay vì `{}`
- Dart chịu được; nên thống nhất kiểu object để hợp đồng sạch.

## Smoke máy ảo (emulator-5554, `com.medassist.app.dev`)
Mở sẵn ở Trang chủ (không gặp khoá PIN, không thử PIN). Đã duyệt bằng uiautomator dump: Trang chủ (lịch hôm nay + "Lời khuyên cho bạn" có trích dẫn "Nguồn: ..."), Nhắc thuốc (Đã uống/Bỏ qua/Hoãn 15 phút), Hồ sơ, Quyền riêng tư (PIN, AI đám mây mặc định có nút Thu hồi đồng ý, Xoá tài khoản). Không phát hiện crash/lỗi hiển thị. Lưu ý: app trong máy ảo đang trỏ server demo (cổng 8000?) — không kiểm chứng.

## Đếm
Critical 0 / High 0 / Medium 0 / Low 3

## Chưa kiểm tra được vì
- Chưa duyệt Tin tức, Cơ sở y tế, Tiến độ, Trợ lý chat trên máy ảo (giới hạn thời gian/phạm vi tối thiểu).
- Không chạy LLM thật (MEDASSIST_LLM) nên nhánh `generated_by=claude` và SSE từ LLM chưa kiểm.
- Không xác minh luồng khoá PIN (app đang mở khoá sẵn).
