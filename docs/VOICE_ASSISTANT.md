# TRỢ LÝ GIỌNG NÓI (voice AI assistant) — thiết kế & điều kiện bật

> **Trạng thái:** kiến trúc đã sẵn sàng, **chưa bật** (chờ chủ dự án duyệt gói + quyền micro).
> Mục tiêu: người dùng (nhất là người lớn tuổi) **nói** câu hỏi về thuốc và **nghe** câu trả lời,
> dùng lại đúng luồng chat có trích dẫn đang chạy (`/v1/agent/chat`, SSE).

## 1. Đã làm sẵn trong code

| Thành phần | File | Việc |
|---|---|---|
| Cổng vào/ra giọng nói | `lib/domain/ports/voice_assistant.dart` | `VoiceInput` (nghe → sự kiện `VoicePartial/VoiceFinal/VoiceError/VoiceListening`) và `VoiceOutput` (đọc to). Thuần Dart, không phụ thuộc gói nào. |
| Adapter "chưa bật" | `lib/data/adapters/unavailable_voice.dart` | `isAvailable = false`, không làm gì. |
| Nối vào DI | `lib/core/voice_providers.dart` | `voiceInputProvider`, `voiceOutputProvider` (mặc định là adapter "chưa bật"). |
| Giao diện | `lib/screens/agent_chat_screen.dart` | Nút micro **chỉ hiện khi** `voiceInputProvider.isAvailable == true`; chữ nhận dạng được điền vào ô nhập (không tự gửi). |

Nghĩa là khi được duyệt, chỉ cần **viết thêm một adapter** và đổi một provider — không sửa màn hình, controller hay server.

## 2. Luồng khi bật

```
nút micro ──► VoiceInput.listen() ──► VoicePartial… ──► VoiceFinal(text)
                                                          │ (điền vào ô nhập, người dùng bấm Gửi)
                                                          ▼
                                         ChatController.send ──► SSE /v1/agent/chat
                                                          │
                              TokenDelta… + Citation… + Done
                                                          ▼
                              (tuỳ chọn) VoiceOutput.speak(câu trả lời + disclaimer)
```

Quy tắc an toàn cho phần đọc to:
- **Luôn đọc kèm disclaimer** ("thông tin chỉ mang tính tham khảo…") — giọng nói dễ bị hiểu là lời khuyên chắc chắn.
- Câu bị từ chối (`out_of_scope`, `no_evidence`) đọc đúng thông điệp từ chối, **không** tự diễn giải thêm.
- Không đọc to khi tai nghe không cắm và người dùng chưa bật "đọc to" (tránh lộ thông tin sức khoẻ nơi công cộng).

## 3. Quyết định cần chủ dự án duyệt (AGENTS.md yêu cầu hỏi trước)

| # | Việc | Vì sao phải hỏi |
|---|---|---|
| 1 | Thêm gói nhận dạng giọng nói (vd. `speech_to_text`) và đọc to (vd. `flutter_tts`) | quy tắc: không thêm package khi chưa duyệt |
| 2 | Khai báo quyền `RECORD_AUDIO` (Android) và mô tả micro/nhận dạng giọng nói (iOS) | quy tắc: không mở rộng quyền khi chưa hỏi |
| 3 | Chọn **nơi xử lý âm thanh** (mục 4) | âm thanh giọng nói là dữ liệu sức khoẻ nhạy cảm |
| 4 | Màn đồng ý riêng cho giọng nói + cập nhật `docs/PRIVACY.md` | quy tắc mục 2: có consent + khử định danh + tài liệu trước khi gửi dữ liệu ra ngoài |

## 4. Phương án xử lý âm thanh (chọn trước khi viết adapter thật)

| Phương án | Âm thanh rời máy? | Nhận xét |
|---|---|---|
| **A. Nhận dạng ngay trên máy** (ưu tiên) | Không | Riêng tư nhất. Phụ thuộc máy có gói giọng tiếng Việt ngoại tuyến; chất lượng thay đổi theo thiết bị. **Cần kiểm tra thực tế** trên máy mục tiêu và trên máy ảo trước khi cam kết. |
| B. Bộ nhận dạng mặc định của hệ điều hành | **Có thể** (Android/iOS có thể gửi âm thanh lên máy chủ của hãng nếu không ép chế độ ngoại tuyến) | Chỉ dùng nếu ép được chế độ ngoại tuyến hoặc người dùng đồng ý rõ ràng; phải ghi trong `PRIVACY.md`. |
| C. Dịch vụ STT/TTS đám mây bên thứ ba | Có | Chỉ khi đủ **cả ba**: đồng ý trong UI, khử định danh, tài liệu. Hiện **không** làm. |

Văn bản (đã nhận dạng) đi qua đúng kênh chat hiện tại tới máy chủ của dự án, như khi gõ tay — không thêm bên thứ ba.

## 5. Kế hoạch kiểm thử khi bật

1. Adapter thật có test với bộ nhận dạng giả (phát `VoicePartial/Final/Error`), kiểm tra: dừng khi rời màn hình, không nghe khi chưa có quyền, không tự gửi.
2. Thử trên máy ảo bằng đầu vào âm thanh giả của emulator + trên 1 máy thật tiếng Việt.
3. Kiểm tra từ chối quyền micro: app vẫn dùng bình thường, nút micro ẩn/báo `voiceUnavailable`.
4. Kiểm tra log: không ghi âm thanh hay nội dung nhận dạng (quy tắc D4).

## 6. Việc kế tiếp sau khi duyệt

1. Bạn duyệt gói + quyền + phương án ở mục 3–4.
2. Viết adapter thật (`lib/data/adapters/…_voice.dart`) + màn đồng ý giọng nói.
3. Đổi `voiceInputProvider` / `voiceOutputProvider`, cập nhật `docs/PRIVACY.md`, chạy lại kiểm thử trên máy ảo.
