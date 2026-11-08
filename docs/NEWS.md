# TIN TỨC THUỐC & Y TẾ (Day 31)

Mục Tin tức trong app lấy dữ liệu qua **server của chính dự án** (`GET /v1/news`), không
bao giờ gọi RSS trực tiếp từ thiết bị người dùng (`AGENTS.md` mục 2).

## Mặc định: TẮT (bài mẫu)

Chưa đặt biến môi trường nào thì server **không gọi mạng** và trả 4 bài mẫu tự soạn
(nhắc nhở chung, không có số liệu/liều dùng cụ thể), kèm `is_sample: true`. App hiện rõ
banner "Đang hiện nội dung mẫu — chưa bật nguồn tin thật." — không bao giờ để người dùng
tưởng đó là tin thật.

## Bật nguồn tin thật (chủ dự án tự quyết định)

**Đang dùng cho demo (chủ dự án duyệt 30/10/2026): chỉ FDA Recalls/Safety.**

```powershell
.\tools\demo_server.ps1 -News      # bật đúng FDA Recalls; không có -News = bài mẫu
```

Muốn nguồn khác thì đặt biến môi trường thủ công (mỗi nguồn mới phải được chủ dự án duyệt URL):

```powershell
$env:MEDASSIST_NEWS_FEEDS = "FDA|https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
.\tools\demo_server.ps1
```

Nhiều nguồn cách nhau bằng dấu phẩy: `"Tên 1|URL 1,Tên 2|URL 2"`.

### Bảng nguồn đã kiểm chứng (28/10/2026) — xem kỹ trước khi bật

| Nguồn | URL | RSS? | Điều khoản | Khuyến nghị |
|---|---|---|---|---|
| **FDA Press Releases** | `fda.gov/.../press-releases/rss.xml` | ✅ | Public domain, dùng tự do | ✅ An toàn để bật |
| **FDA Recalls/Safety** | `fda.gov/.../recalls/rss.xml` | ✅ | Public domain | ✅ An toàn để bật |
| VnExpress Sức khỏe | `vnexpress.net/rss/suc-khoe.rss` | ✅ | Chỉ cho "cá nhân và tổ chức **phi lợi nhuận**", bắt buộc ghi nguồn, báo có quyền yêu cầu ngừng bất cứ lúc nào | ⚠️ Đồ án học tập: tạm chấp nhận được (ghi rõ nguồn, không đăng lại toàn văn). Phát hành thương mại: **phải xin phép trước** |
| Tuổi Trẻ Sức khỏe | `tuoitre.vn/rss/suc-khoe.rss` | ✅ (ngày giờ sai chuẩn RFC822, đã xử lý trong code) | Như trên | Như trên |
| Dân trí Sức khỏe | `dantri.com.vn/rss/suc-khoe.rss` | ✅ | Như trên, có thêm "Cấm sao chép dưới mọi hình thức nếu không có văn bản chấp thuận" | ⚠️ Cẩn trọng hơn — câu "cấm sao chép" rõ ràng nhất trong các báo |
| Sức khỏe & Đời sống | `suckhoedoisong.vn/rss/home.rss` | ✅ | Không tìm thấy trang điều khoản | ❌ **Không dùng feed "chuyên mục"**: đã đo thật, feed `/rss/y-te.rss` trả nhầm **trang chủ** (tin hình sự, xổ số). Bộ lọc trong `app/news.py` chặn được hầu hết nhưng không nên tin tưởng tuyệt đối |
| MedlinePlus | `medlineplus.gov/feeds/*.xml` | ✅ | RSS ghi rõ **"for personal use only"** | ❌ **Không dùng trong app** |
| Bộ Y tế, Cục Quản lý Dược | — | ❌ không có | — | Không có lựa chọn |
| WHO | `who.int/rss-feeds/...` | ❓ | Không xác minh được (server trả 403 khi tôi thử) | Thử trên máy thật trước khi tin vào URL này |

**Khuyến nghị cho đồ án:** bật FDA (an toàn tuyệt đối) + tối đa một báo Việt Nam, và ghi rõ
trong phần giới thiệu rằng đây là demo học thuật, không phải sản phẩm thương mại.

## Vì sao kiến trúc như vậy

1. **App không gọi RSS trực tiếp.** Một địa chỉ duy nhất (server của ta) — giữ đúng
   nguyên tắc "không gọi mạng ngoài localhost/server của mình".
2. **Gỡ nguồn không cần phát hành lại app.** Báo yêu cầu dừng → sửa biến môi trường,
   khởi động lại server.
3. **Lọc tập trung ở một chỗ** (`app/news.py::is_relevant`): loại bỏ tin không phải y tế
   (đã bắt được trường hợp thật: feed "y tế" trả nhầm trang chủ với tin xổ số/hình sự),
   quảng cáo thuốc trá hình (từ khoá "hotline", "giá chỉ", số điện thoại trong trích đoạn),
   và link tiếp thị (`utm_`, `affiliate`).
4. **Chỉ trích dẫn, không đăng lại.** Tiêu đề + tối đa 200 ký tự + tên nguồn + link bài
   gốc. Không lưu, không hiện toàn văn — đúng tinh thần điều khoản của báo Việt Nam dù
   họ chưa cho phép rõ ràng ứng dụng bên thứ ba.
5. **Chống SSRF.** `_is_safe_url()` chặn `localhost`, IP nội bộ, IP metadata cloud — ai
   đặt được biến môi trường cũng không khiến server tự gọi vào chính nó hay mạng nội bộ.

## Mở bài gốc trong app

Bấm vào bài -> hộp xác nhận "Rời ứng dụng để đọc bài?" (Huỷ / Chép link / Mở trình duyệt).
Mở bằng kênh native `medassist/device` có sẵn, **không** dùng gói `url_launcher`. Chỉ mở link
`https` thuộc allowlist (`lib/widgets/external_link.dart::safeExternalUri`).

Feed FDA ghi link `http://`: server **nâng lên `https://`** rồi mới kiểm tra allowlist
(`app/news.py::_upgrade_http`). App không bao giờ nhận link `http`; link ngoài allowlist vẫn bị xoá.
Đã kiểm tra trên máy ảo 30/10/2026: 20 bài FDA, cả 20 có link https, bấm mở được Chrome.

## Giới hạn đã biết

- Không ghi log ai đọc bài nào (thói quen đọc tin y tế cũng là dữ liệu sức khoẻ).
- Cache 15 phút trong RAM server (`MEDASSIST_NEWS_TTL`, giây) — mất khi khởi động lại.
- Bộ lọc dựa trên từ khoá, không hoàn hảo; một bài quảng cáo tinh vi có thể lọt qua.
- Ảnh trong tin lấy trực tiếp từ server báo gốc — nếu báo đổi/xoá ảnh, app hiện icon thay thế.

## Cổng tin tức đã kiểm chứng (allowlist)

Mọi link người dùng thấy trong app phải đến từ nguồn **đã được chủ dự án kiểm chứng**. Đây là
một "cổng" nằm trong code (`app/news.py`), không phải hạ tầng đám mây.

- Biến môi trường `MEDASSIST_NEWS_ALLOWED_DOMAINS` (tên miền đăng ký, ngăn cách bằng dấu
  phẩy). Mặc định chỉ có cơ quan y tế công: `fda.gov, who.int, cdc.gov, nih.gov, moh.gov.vn,
  dav.gov.vn`. **Chủ dự án phải tự kiểm chứng và chỉnh danh sách này** trước khi phát hành.
- Khớp theo nhãn tên miền: host bằng hoặc là subdomain của một mục (không phân biệt hoa
  thường). `evilfda.gov` và `fda.gov.evil.com` bị từ chối.
- Ba điểm chặn: (a) feed cấu hình có host ngoài allowlist bị bỏ qua và chỉ ghi **tên** vào
  `failed_sources`; (b) link bài phải là `https`, không userinfo, cổng 443/vắng và nằm trong
  allowlist, nếu không `url` bị xoá (bài giữ dạng chỉ-văn-bản, app coi là không có link); ảnh
  ngoài allowlist cũng bị bỏ; (c) mọi chặng chuyển hướng (kể cả URL cuối) phải còn trong
  allowlist, nếu không kết quả của feed đó bị bỏ.
- Cách thêm một tên miền: thêm vào `MEDASSIST_NEWS_ALLOWED_DOMAINS` (ví dụ
  `fda.gov,who.int,example-health.org`), rồi thêm feed vào `MEDASSIST_NEWS_FEEDS`, khởi động lại.
  Biến rỗng/không hợp lệ -> quay về danh sách mặc định.
- API trả thêm `verified_sources` (danh sách tên miền đang áp dụng) để app hiện
  "Nguồn đã kiểm chứng: fda.gov, who.int…".
- **Bài mẫu không có link** (`url` rỗng, `is_sample=true`).
- Cô lập hạ tầng thật (ví dụ Railway private networking, chặn egress) là việc **triển khai**,
  không được làm trong code này.
