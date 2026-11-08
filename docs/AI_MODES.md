# Chế độ trả lời của MedAssist: có / chưa bật AI đám mây

| | Chưa bật AI đám mây | Đã bật AI đám mây |
|---|---|---|
| Dữ liệu rời máy chủ MedAssist? | Không | Có, tới Anthropic |
| Thứ được gửi | Không gì | Câu hỏi đã khử định danh + đoạn trích nguồn; tên thuốc/bệnh nền chỉ khi có đồng ý hồ sơ |
| Cách trả lời | Trích nguyên văn có mã [R1] | Claude diễn đạt lại tự nhiên, nhưng bị kiểm tra bám nguồn |
| Điều kiện | Không | Khoá API trên server + công tắc đồng ý trong app + cờ `MEDASSIST_LLM=claude` |
| Khi lỗi | n/a | Lặng lẽ rơi về trích nguyên văn, log mã lý do (vd `http_401`), không log nội dung |
| Tốc độ / chi phí | Nhanh, miễn phí | Thêm một lượt gọi mạng; tốn phí theo token |

## Không đổi giữa hai chế độ

- Nguồn trích dẫn (mã `[R#]`, `source`, cờ `is_authoritative`).
- Từ chối chẩn đoán, liều dùng, tương tác chưa có dữ liệu kiểm chứng.
- Cảnh báo cấp cứu (không gọi LLM).
- Disclaimer.

## Cách app hiển thị

Sự kiện SSE cuối: `event: done` / `data: {"sessionId": "...", "mode": "extractive" | "cloud"}`.
`mode = "cloud"` CHỈ khi bản viết lại của Claude đã qua kiểm tra bám nguồn ở lượt đó; mọi
trường hợp khác (mock/ollama, chưa đồng ý, bị từ chối, bị loại, cấp cứu, từ chối, no_data)
là `"extractive"`. App hiện một chip nhỏ theo `mode`.

## Quy tắc bám nguồn cho bản Claude viết lại (chủ dự án duyệt 2026-10-30)

Dữ liệu mẫu là tiếng Anh nên không so khớp chữ với câu tiếng Việt được. Mỗi dòng của Claude
phải (1) kết thúc bằng mã trích dẫn có thật `[Rn]`, (2) chứa NGUYÊN VĂN đoạn trích của nguồn
đó trong dấu ngoặc kép, (3) chỉ kèm lời giải thích ≤ 14 từ, (4) không có con số mới ngoài
đoạn trích/tên thuốc. Không có câu mở đầu/kết không kèm mã. Vi phạm bất kỳ điều nào -> bỏ bản
Claude, trả lời trích nguyên văn (`extractive`). Log chỉ ghi mã lý do + số đếm, không ghi nội dung.
Thử nghiệm cục bộ với câu hỏi giả đã duyệt: 6/6 lần qua kiểm tra (mode `cloud`).

## Cách kiểm thử

- Extractive: tắt công tắc đồng ý (hoặc không đặt `MEDASSIST_LLM`) -> `done.mode == "extractive"`.
- Cloud: `MEDASSIST_LLM=claude` + khoá trên server + bật công tắc -> hỏi một thuốc có trong kho
  -> `mode == "cloud"` khi câu trả lời qua kiểm tra; 401 hoặc không bám nguồn -> `extractive`.
- Tự động: `pytest tests/test_diet_and_mode.py` (dùng provider giả, không gọi mạng).
