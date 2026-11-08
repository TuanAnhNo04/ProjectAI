# CHÍNH SÁCH DỮ LIỆU & QUYỀN RIÊNG TƯ — MedAssist

> **Trạng thái:** bản nháp kỹ thuật, mô tả **đúng những gì code hiện làm** (cập nhật 24/10/2026).
> Chưa phải văn bản pháp lý. Trước khi phát hành công khai cần người có chuyên môn pháp lý rà soát.
> Nguồn quy tắc: [`../AGENTS.md`](../AGENTS.md) · các quyết định: [`DECISIONS.md`](DECISIONS.md).

## 1. Nguyên tắc

1. **Dữ liệu sức khoẻ ở lại trên máy mình.** Server do dự án tự chạy; không có bên thứ ba nhận dữ liệu.
2. **Không dữ liệu nào rời khỏi dự án** qua LLM/OCR/analytics/báo lỗi bên ngoài. Hiện **không có** SDK nào như vậy trong app lẫn server.
3. **Thu thập tối thiểu, xoá được.** Mỗi loại dữ liệu bên dưới nêu rõ nơi lưu, thời hạn và cách xoá.
4. **Không bịa nguồn y khoa.** Mọi câu trả lời y khoa phải trích dẫn tài liệu đã nạp; không có thì từ chối.

## 2. Dữ liệu nào được lưu, ở đâu, bao lâu

### 2.1 Trên server (`server/medassist-core`, SQLite)

| Dữ liệu | Lưu gì | Thời hạn | Cách xoá |
|---|---|---|---|
| Tài khoản | tên đăng nhập, email, **hash bcrypt** của mật khẩu (không lưu mật khẩu) | tới khi xoá tài khoản | `POST /v1/auth/delete-account` |
| Lịch sử | đường dẫn ảnh + kết quả tra cứu, gắn `user_id` | theo `MEDASSIST_HISTORY_RETENTION_DAYS` (mặc định **không tự xoá**) | `DELETE /v1/history/{id}`; xoá tài khoản; retention |
| Phiên đăng nhập | **hash** refresh token (không lưu token thô) | 7 ngày (hết hạn) | đăng xuất; xoá tài khoản |
| Hội thoại với agent | các lượt chat gần nhất, **chỉ trong RAM** | tự hết hạn sau 24 giờ; mất khi server khởi động lại | `DELETE /v1/agent/session/{id}`; xoá tài khoản |
| Số liệu sử dụng | số request / độ dài ký tự, **chỉ trong RAM** | mất khi khởi động lại | xoá tài khoản |

Mọi truy vấn lịch sử/hội thoại lọc theo `user_id` lấy từ token (có test hai người dùng A/B chống truy cập chéo).

### 2.2 Trên thiết bị (app Flutter)

| Dữ liệu | Nơi lưu | Mã hoá | Cách xoá |
|---|---|---|---|
| Token đăng nhập | `flutter_secure_storage` (Keystore/Keychain) | có | đăng xuất |
| Cache lịch sử, kết quả tìm kiếm, kết quả nhận diện (để xem khi mất mạng) | `shared_preferences` | **không** | **tự xoá khi đăng xuất hoặc khi server trả 401** |
| Hồ sơ sức khoẻ (dị ứng, bệnh nền, thuốc đang dùng), lịch nhắc, nhật ký uống thuốc | SQLite cục bộ (`health_profile.db`) | **không** | nút xoá trong màn Hồ sơ (`deleteAll`) |

- Hồ sơ sức khoẻ **chỉ nằm trên máy** và **không** được Google Auto Backup đẩy lên đám mây (`android:allowBackup="false"`).
- Ảnh hộp thuốc/đơn thuốc: ở chế độ `ANALYZER=ondevice` ảnh **không rời máy** (OCR bằng ML Kit trên máy). Ở chế độ mặc định `remote` (và `hybrid` khi không nhận diện được trên máy) ảnh được gửi lên server cấu hình bởi `API_BASE_URL` — xem mục 3. `POST /v1/analyze` của `medassist-core` chưa bật (trả 501) nên hiện chỉ server Flask cũ/mock nhận ảnh.

## 3. Dữ liệu rời khỏi máy (tới server của dự án)

| Khi nào | Gửi gì | Ghi chú |
|---|---|---|
| Đăng ký / đăng nhập | tên đăng nhập, mật khẩu, email | mật khẩu chỉ đi qua đường truyền, server băm ngay |
| Nhận diện ảnh (`remote`/`hybrid`) | **ảnh hộp thuốc/đơn thuốc** (multipart) | chỉ tới server của `API_BASE_URL`; mock server chỉ chạy localhost và không dùng với dữ liệu thật |
| Tìm thuốc | từ khoá tìm kiếm | không ghi vào log |
| Chat với agent | nội dung câu hỏi **và danh sách "thuốc đang dùng"** từ hồ sơ | chỉ dùng trong lượt đó để cảnh báo; server **không lưu** danh sách thuốc, chỉ lưu lượt chat trong RAM 24 giờ |

Lưu ý riêng: khi người dùng có hồ sơ và dùng **Lời khuyên**, tên thuốc đang dùng và bệnh nền được gửi tới **máy chủ MedAssist** để xử lý yêu cầu đó (không lưu, không ghi log). Việc này **tách biệt** với sự đồng ý gửi tới đám mây Claude (mục 6.1). Khử định danh trước khi gửi ra Claude là **nỗ lực tối đa, không tuyệt đối**; có thể bỏ sót một số dạng định danh.

Bản release bắt buộc **HTTPS**; cleartext chỉ được phép ở bản debug tới `localhost`/`10.0.2.2`.

## 4. Nhật ký (log)

- **Log truy cập** chỉ có: `trace_id`, `route` (dạng template, không có query string), `latency_ms`, `error_code`.
- **Audit log** (hành động nhạy cảm) chỉ có: `trace_id`, `action`, `outcome` — với `action` thuộc danh sách đóng (đăng nhập thành công/thất bại, xoá tài khoản, xoá lịch sử/phiên, tái sử dụng refresh token, dọn retention). **Không** có tên đăng nhập, email, `user_id`.
- **Không bao giờ** ghi: mật khẩu, token, header `Authorization`, email, tên thuốc, nội dung chat, đường dẫn ảnh.

## 5. Quyền của người dùng

| Quyền | Cách thực hiện hôm nay |
|---|---|
| Xoá dữ liệu trên server | `POST /v1/auth/delete-account` (cần token **và** mật khẩu): xoá tài khoản, lịch sử, phiên, hội thoại |
| Xoá dữ liệu trên máy | đăng xuất (xoá cache); nút xoá hồ sơ sức khoẻ |
| Xoá một mục / một cuộc hội thoại | `DELETE /v1/history/{id}`, `DELETE /v1/agent/session/{id}` |
| Không bị gửi dữ liệu cho bên thứ ba | mặc định — không có tích hợp bên ngoài nào |

### 5.1 Bộ nhớ cá nhân (triệu chứng, bệnh nền)

Bộ nhớ về triệu chứng và bệnh nền của bạn được lưu **CHỈ trên điện thoại**, có mã hoá. Server chỉ nhận ngữ cảnh tối thiểu cho từng lượt hỏi (ví dụ danh sách triệu chứng gần đây để nhận biết tái phát), không lưu và không ghi log. Các trường này không bao giờ được đưa vào prompt của LLM đám mây.

## 6. Điều kiện bắt buộc trước khi gửi dữ liệu ra bên ngoài (LLM / OCR đám mây)

Hiện tại LLM mặc định là **mock chạy offline**, và OCR chạy **trên máy** (ML Kit).

**LLM cục bộ (Ollama) — tuỳ chọn, bật bằng `MEDASSIST_LLM=ollama`:** mô hình chạy ngay trên máy chạy server và `app/llm_ollama.py` chỉ nhận địa chỉ `127.0.0.1`/`localhost`/`::1` (không qua proxy, không theo redirect), nên **không có dữ liệu nào rời máy** và không cần bên nhận thứ ba. Câu hỏi và các đoạn nguồn được đưa vào prompt; không ghi log nội dung. Câu LLM sinh ra vẫn phải qua kiểm chứng trích dẫn (`verify_grounding`), không đạt thì quay về câu trả lời trích nguyên văn. Hướng dẫn: `docs/LLM_LOCAL.md`.

Nếu sau này thêm dịch vụ **bên ngoài máy** (LLM/OCR đám mây), phải có **đủ cả ba** trước khi viết adapter gửi dữ liệu thật:

1. **Đồng ý rõ ràng trong UI** của người dùng, tách riêng cho từng loại dữ liệu. Theo
   Nghị định 356/2025/NĐ-CP (hiệu lực 01/01/2026, thay Nghị định 13/2023/NĐ-CP) về bảo vệ
   dữ liệu cá nhân, đồng ý phải **rõ ràng và kiểm chứng được** — phải lưu lại **thời điểm
   + phiên bản văn bản đã đồng ý**, không chỉ một cờ boolean.
2. **Khử định danh** trước khi gửi (bỏ tên, email, số điện thoại, ID). ⚠️ Regex là **lưới
   an toàn giảm thiểu rủi ro, không phải đảm bảo tuyệt đối** — tên người tiếng Việt không
   bắt hết được bằng regex thuần (tên là từ thường). Xem `app/deidentify.py`.
3. **Cập nhật tài liệu này** nêu tên bên nhận, dữ liệu gửi và thời gian họ lưu.

### 6.1 Claude (Anthropic) — nhà cung cấp đám mây ĐÃ CÓ, bị khoá bởi hai cổng

Cập nhật 29/10/2026. Claude là provider tuỳ chọn; **mặc định vẫn là mock offline**.

**Bên nhận:** Anthropic (Claude Messages API, `https://api.anthropic.com/v1/messages`),
gọi **chỉ từ server** (khoá API nằm ở biến môi trường server, không bao giờ trong app).

**Hai cổng, thiếu một là không gửi gì:**

1. **Cờ server:** `MEDASSIST_LLM=claude` **và** có `ANTHROPIC_API_KEY`. Thiếu khoá ->
   `get_llm_provider()` rơi về mock.
2. **Đồng ý theo từng request của người dùng:** body phải có `cloud_ai_consent=true`.
   App chỉ gửi `true` sau khi người dùng bật công tắc ở màn Quyền riêng tư; app lưu
   **thời điểm + phiên bản văn bản** đã đồng ý (`CloudAiConsentStore`). Thiếu cờ -> lượt
   đó dùng mock, kể cả khi server đã bật Claude
   (`app/orchestrator.py::_select_provider_for_request`).

**Dữ liệu được gửi (chỉ khi qua cả hai cổng):**

| Luồng | Gửi đi |
|---|---|
| `/v1/agent/chat` (diễn đạt lại câu trả lời) | câu hỏi **đã khử định danh** (`app/deidentify.py`) + các đoạn trích nguồn có mã trích dẫn từ danh mục thuốc |
| `/v1/agent/chat` (đối chiếu triệu chứng/tác dụng phụ) | như trên + câu nháp có tên thuốc khớp + bệnh nền trong `health_context.conditions` (đã khử định danh, đặt trong vùng DỮ LIỆU của prompt) |
| `/v1/agent/lifestyle` (viết lại câu chữ gợi ý) | các gợi ý đã dựng sẵn (có tên thuốc + đoạn trích tác dụng phụ) + `recent_topics` **đã khử định danh** |

Không gửi: tên đăng nhập, email, token, `user_id`, ảnh, lịch sử tìm kiếm, danh sách dị ứng.
Mọi văn bản người dùng nằm trong vùng DỮ LIỆU (escape `<`/`>`), không phải chỉ dẫn.

**Kiểm soát đầu ra:** câu do Claude viết phải qua `verify_grounding` (chat) hoặc kiểm tra
mã trích dẫn/không có con số mới (lifestyle); không đạt -> dùng bản luật trích nguyên văn.
Luồng **khẩn cấp** (khó thở, đau ngực...) không bao giờ gọi LLM.

**Server stateless với dữ liệu sức khoẻ:** `current_medications`, `health_context`,
`recent_topics` chỉ dùng trong lượt request, **không** lưu CSDL và **không** ghi log. Phiên
chat chỉ trong RAM (mất khi restart, xoá được qua `DELETE /v1/agent/session/{id}`). Server
lưu lâu dài chỉ: tài khoản (mật khẩu băm bcrypt) + danh mục thuốc.

**Lưu trữ phía Anthropic:** ⚠️ Chủ dự án cần **tự kiểm tra điều khoản lưu giữ dữ liệu hiện
hành của Anthropic** (thời gian lưu, có dùng để huấn luyện không, ZDR) trên trang chính sách
chính thức trước khi bật cho người dùng thật, và cập nhật mục này với kết quả.

**Chuyển dữ liệu ra nước ngoài:** máy chủ Anthropic (và Railway, xem `docs/DEPLOY_RAILWAY.md`)
đặt ngoài Việt Nam — cần đánh giá nghĩa vụ chuyển dữ liệu cá nhân ra nước ngoài theo Nghị
định 356/2025/NĐ-CP trước khi phát hành thật.

**Giới hạn:** khử định danh bằng regex là lưới an toàn, không đảm bảo tuyệt đối (xem mục 6.2).

### 6.2 Thành phần kỹ thuật

- `server/medassist-core/app/deidentify.py`: khử định danh bằng regex (email, SĐT, CCCD,
  ngày tháng, URL/IP, tên có danh xưng dẫn trước). Có test với dữ liệu giả.
- `server/medassist-core/app/llm_claude.py`: adapter HTTP (header `anthropic-version:
  2026-10-21`, model mặc định `claude-sonnet-5-5`, `output_config.effort`), mọi lỗi ->
  từ chối, không log nội dung.
- `server/medassist-core/app/safety.py`, `app/lifestyle.py`: luật xác định chạy **trước**
  và **độc lập** với LLM.

## 7. Khoảng hở đã biết (chưa xử lý — nói thẳng)

1. ~~App chưa có màn xoá tài khoản~~ → **đã có** (Menu ⋮ → Quyền riêng tư → Xoá tài khoản và dữ liệu; cần mật khẩu), đã kiểm chứng trên máy ảo.
2. ~~Chưa có màn đồng ý~~ → **đã có** màn đồng ý ở lần mở đầu tiên (và thu hồi được trong Quyền riêng tư).
3. **SQLite cục bộ và cache `shared_preferences` chưa mã hoá.** Máy bị root/mất máy không khoá màn hình thì có thể đọc được.
4. **CSDL phía server chưa mã hoá khi lưu**; bản chạy thật cần mã hoá ổ đĩa hoặc chuyển sang Postgres có mã hoá.
5. **Retention mặc định tắt** và việc dọn refresh token hết hạn (`purge_expired_refresh_tokens`) chưa được lên lịch.
6. **Access token còn hiệu lực tối đa 15 phút sau khi xoá tài khoản** (không có danh sách thu hồi access token); nó không còn đọc được dữ liệu vì dữ liệu đã bị xoá.
7. **Nhắc uống thuốc**: quyền `POST_NOTIFICATIONS` do plugin tự khai báo; app xin quyền lúc chạy khi tạo nhắc đầu tiên. **Giờ nhắc chưa chính xác**: hiện lặp mỗi 24 giờ kể từ lúc tạo, chưa nổ đúng giờ:phút đã chọn — cần gói `timezone` (xem `docs/RELEASE.md`).
9. **Access log của uvicorn ghi cả query string** (`?name=panadol`). Đã tắt `uvicorn.access` ngay trong `app/main.py`; log truy cập an toàn riêng vẫn bật. Nếu dùng reverse proxy (nginx…) phải tắt/che query string ở đó.
8. **Không có báo lỗi crash.** Đây là lựa chọn cố ý: SDK như Crashlytics/Sentry gửi dữ liệu ra ngoài nên cần chủ dự án duyệt và cập nhật mục 3.
