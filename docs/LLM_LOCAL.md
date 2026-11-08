# LLM THẬT CHẠY TRÊN MÁY (Ollama)

Dữ liệu **không rời máy**: Ollama chạy ngay trên máy bạn, và server MedAssist chỉ gọi tới
`127.0.0.1`. Mặc định vẫn là mock; LLM thật chỉ bật khi bạn đặt `MEDASSIST_LLM=ollama`.

## 1. Cài Ollama (bạn tự làm — cần mạng)

1. Tải bộ cài Windows từ trang chính thức của Ollama và cài.
2. Tải một model nhỏ (vài GB, cần mạng một lần):

```powershell
ollama pull qwen2.5:3b
ollama list            # phải thấy qwen2.5:3b
```

Ổ C chỉ còn khoảng 11 GB trống (ổ D còn nhiều): có thể đặt thư mục model sang ổ D bằng biến môi trường
`OLLAMA_MODELS` (ví dụ `D:\ollama\models`) **trước** khi `ollama pull`.

Model khác cũng được (đổi bằng `MEDASSIST_OLLAMA_MODEL`); máy yếu nên chọn model ≤ 3B.

## 2. Chạy server với LLM thật

```powershell
$env:MEDASSIST_LLM = "ollama"
$env:MEDASSIST_OLLAMA_MODEL = "qwen2.5:3b"     # tuỳ chọn
.\tools\demo_server.ps1
```

Hoặc `.\tools\demo_server.ps1 -Llm` (script tự đặt hai biến trên).

## 3. Kiểm tra

- Mở app → Chat → hỏi "Panadol có công dụng gì". Câu trả lời phải kèm mã `[R1]`.
- Hoặc gọi `POST /v1/agent/ask` (Postman) và xem trường `provider`: `ollama` là LLM thật, `mock` là chế độ cũ.
- Ollama không chạy / quá chậm → server **không lỗi**: tự dùng câu trả lời trích nguyên văn (như mock).

## 4. An toàn — vì sao vẫn có thể tin kết quả

| Lớp | Việc làm |
|---|---|
| Trước khi gọi LLM | lọc câu hỏi ngoài phạm vi (liều dùng, tương tác, chẩn đoán…) + phát hiện prompt injection |
| Trong prompt | nguồn là dữ liệu đã escape, không phải lệnh |
| Sau khi LLM sinh | `verify_grounding()`: mọi dòng phải có mã trích dẫn tồn tại và bám nội dung nguồn; không đạt → dùng lại câu trả lời trích nguyên văn |
| Mạng | chỉ loopback, không proxy, không redirect |

## 5. Giới hạn hiện tại (nói thẳng)

- **Đã chạy thử với `qwen2.5:3b` trên máy này (28/10/2026):** Ollama cài ở `D:\ollama\app`, model ở `D:\ollama\models` (biến người dùng `OLLAMA_MODELS`), chỉ nghe trên `127.0.0.1:11434`. Khi model đã nạp, mỗi câu mất khoảng 0,3–1,7 giây; lần đầu mất khoảng 1 phút để nạp model (`demo_server.ps1 -Llm` tự nạp sẵn).
- Kết quả mẫu: "Panadol có tác dụng phụ gì" → LLM trả `- Nausea, allergic reactions [R1]` (đạt kiểm chứng). "Strepsils dùng để làm gì" → model chép lại câu hỏi, không đạt kiểm chứng nên chat tự dùng câu trích nguyên văn.
- Kiểm chứng trích dẫn khá chặt: model 3B hay **dịch** nguồn tiếng Anh sang tiếng Việt hoặc chép lại câu hỏi; lời nhắc trong `llm_ollama.py` yêu cầu chép nguyên văn nên tỉ lệ đạt đã tốt hơn, nhưng vẫn có lượt quay về câu trích nguyên văn. `/v1/agent/ask` thì từ chối (`unsupported_claim`) thay vì quay lại, vì đó là đường kiểm chứng chặt.
- Kiểm chứng so khớp theo từ (≥ 60% từ của dòng nằm trong nguồn) nên vài từ nối như "có thành phần" vẫn lọt; nội dung thực chất vẫn phải lấy từ nguồn.
- Câu hỏi liều dùng/tương tác vẫn bị từ chối (dữ liệu mới là 10 thuốc mẫu, không phải nguồn lâm sàng).
- LLM chạy trên CPU laptop có thể mất vài giây đến vài chục giây mỗi câu (timeout 60 giây).
- LLM đám mây (Claude/OpenAI…) **chưa** làm: cần đủ đồng ý trong UI + khử định danh + ghi PRIVACY (`AGENTS.md` mục 2).
