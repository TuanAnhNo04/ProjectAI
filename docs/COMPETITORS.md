# Phân tích đối thủ — app nhắc thuốc & quét đơn/hộp thuốc (OCR)

> Lập ngày 30/10/2026. Nguồn: tìm kiếm web công khai (chỉ đọc), đã được chủ dự án cho phép trong chat.
> **Độ tin cậy:** phần lớn nguồn là trang tổng hợp/blog so sánh, một số mang tính quảng cáo. Mỗi nhận định ghi
> mức tin cậy; mục "chưa kiểm chứng" cần xem lại trực tiếp trên App Store/Google Play trước khi đưa vào báo cáo/thuyết trình.
> Phạm vi: nhóm **nhắc thuốc quốc tế** và **quét đơn/hộp thuốc (OCR)**. Chưa gồm app thuốc Việt Nam và app tra cứu (Drugs.com, Epocrates...).

## 1. Tóm tắt một trang

| App | Mạnh nhất | Yếu nhất (theo nguồn) |
|---|---|---|
| **Medisafe** | Lịch phức tạp, người thân theo dõi (Medfriend), cảnh báo tương tác, 90+ chỉ số sức khoẻ | Từ 01/2026 bản miễn phí chỉ còn 2 thuốc, lịch sử bị khoá sau gói trả phí; báo cáo khó dùng |
| **MyTherapy** | Miễn phí/không quảng cáo (theo nhà phát triển), quét mã vạch, báo cáo PDF cho bác sĩ, theo dõi chỉ số | Nhập tay không có hàm lượng; thông báo có thể không ổn định sau cập nhật (nguồn yếu) |
| **Dosecast** | Chạy offline, không giới hạn số thuốc, hoãn nhắc linh hoạt | Giao diện cũ, ít cập nhật, đồng bộ nhiều máy phải trả phí, không có chế độ tối |
| **App quét đơn/OCR nói chung** | Tự điền tên thuốc/liều/tần suất, giảm gõ tay | Sai ký tự có thể ghi nhầm thuốc/liều; chữ viết tay kém hơn nhiều chữ in |

Không có kết quả về **Round Health** trong lần tìm này nên không đưa vào.

## 2. Chi tiết từng app

### 2.1 Medisafe
- **Tính năng:** nhắc thuốc cá nhân hoá, theo dõi tuân thủ và báo cáo, cảnh báo tương tác thuốc, hơn 90 chỉ số sức khoẻ, người thân/người chăm sóc (Medfriend), tự thêm lịch hẹn vào lịch điện thoại.
- **Điểm mạnh (theo nguồn):** lịch linh hoạt cho phác đồ phức tạp; hỗ trợ người chăm sóc; có cảnh báo tương tác. Điểm App Store khoảng 4,7/5 (~99.000 lượt).
- **Điểm yếu (theo nguồn):** từ 01/2026 phần lớn tính năng cốt lõi cần đăng ký trả phí, bản miễn phí giới hạn 2 thuốc; dữ liệu lịch sử nằm sau tường trả phí; báo cáo khó dùng; phụ thuộc tích hợp hồ sơ y tế kiểu Mỹ nên khó mở rộng quốc tế. Giá nguồn nêu không thống nhất (4,99 USD/tháng hay 39,99 USD/năm) — **chưa kiểm chứng**.
- **Chưa xác nhận:** Medisafe có quét mã vạch/ảnh hay không (không tìm thấy mô tả).
- Nguồn: [Medical News Today](https://www.medicalnewstoday.com/articles/pill-reminder/), [IntuitionLabs](https://intuitionlabs.ai/software/patient-education-engagement/medication-adherence-apps/medisafe), [Pillo so sánh](https://pillo.care/blog/pillo-vs-medisafe-comparison), [Healthify](https://healthify.nz/apps/m/medisafe-meds-pill-reminder-app).

### 2.2 MyTherapy
- **Tính năng:** nhắc thuốc, nhắc tái đơn/mua thuốc, theo dõi hơn 20 chỉ số (đường huyết, nhịp tim...), báo cáo PDF gửi bác sĩ/dược sĩ, tìm thuốc trong cơ sở dữ liệu hoặc **quét mã vạch hộp thuốc** (nếu không quét được thì nhập tay).
- **Điểm mạnh:** dễ dùng (Parkinson's UK Tech Guide), miễn phí và không quảng cáo theo lời nhà phát triển, điểm App Store khoảng 4,8.
- **Điểm yếu:** nhập tay không ghi được hàm lượng (một bài đánh giá bên thứ ba); trang tổng hợp AI nêu thông báo không ổn định sau cập nhật và quảng cáo/ép đăng ký — **nguồn độ tin cậy thấp, mâu thuẫn với lời nhà phát triển**.
- Nguồn: [App Store](https://apps.apple.com/gb/app/pill-reminder-mytherapy/id662170995), [Parkinson's UK](https://techguide.parkinsons.org.uk/catalogue/my-therapy), [Hướng dẫn thêm thuốc](https://mytherapy.uservoice.com/knowledgebase/articles/1166296-how-do-i-enter-a-new-medication).

### 2.3 Dosecast
- **Tính năng:** nhắc thuốc có hoãn trước/sau, tự tắt tiếng theo giấc ngủ, không giới hạn số thuốc, lịch lặp, chạy offline; đồng bộ nhiều máy là gói Pro.
- **Điểm yếu:** giao diện cũ, ít cập nhật; popup nhắc che các thuốc sắp tới; chưa có chế độ tối; điểm khoảng 4,0 (~225 đánh giá). Một nguồn nói có người chăm sóc theo dõi từ xa, nguồn khác nói không có — **mâu thuẫn, không dùng**.
- Nguồn: [App Store](https://apps.apple.com/us/app/id365191644), [Griswold Labs](https://griswoldlabs.pages.dev/blog/free-pill-reminder-apps-2026).

### 2.4 Quét đơn/hộp thuốc (OCR)
- **Cách làm phổ biến:** quét mã vạch (MyTherapy) hoặc chụp ảnh nhãn/đơn rồi OCR để tự điền tên, liều, tần suất.
- **Điểm yếu đã có bằng chứng:**
  - Ảnh mờ, lệch, thiếu sáng làm OCR sai; nhãn chai cong khó đọc; mỗi nhà thuốc một mẫu nhãn khác nhau; **một ký tự sai có thể ghi nhầm thuốc/liều/hạn dùng** (các bằng sáng chế mô tả vấn đề này).
  - Nghiên cứu PresRx (2016, chỉ 8 người thử): tự điền đúng 95% tên thuốc, 98% liều, 96% tần suất — mẫu rất nhỏ.
  - Nghiên cứu tiếng Việt (Google Vision + GPT-3.5, 97 đơn in): 90,2% trích đúng cặp khoá–giá trị, nhưng chỉ **25,7% trên 11 đơn viết tay**. Mẫu nhỏ.
  - Một app trong nguồn tự ghi trong điều khoản: OCR có thể đọc sai, người dùng phải kiểm tra từng kết quả.
- **Không tìm thấy** số liệu độ chính xác công khai của ML Kit trên đơn thuốc viết tay. Google đang nghiên cứu đọc chữ bác sĩ trong Google Lens nhưng vẫn giữ dược sĩ trong vòng kiểm tra.
- Nguồn: [Bằng sáng chế OCR bề mặt cong](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11495019), [Nghiên cứu PresRx](https://journals.sagepub.com/doi/full/10.1089/tmj.2016.0100), [Nghiên cứu tiếng Việt](https://keep.hcmiu.edu.vn/handle/123456789/6544), [Google và chữ bác sĩ](https://www.fiercebiotech.com/medtech/google-aims-translate-hand-scribbled-doctors-notes-and-prescriptions-using-ai).

## 3. Điểm mạnh của đối thủ → áp dụng vào MedAssist

| # | Học từ | Áp dụng cho MedAssist | Ghi chú theo AGENTS.md |
|---|---|---|---|
| M1 | Medisafe: người thân theo dõi | Chế độ "người thân" xem tuân thủ, **chia sẻ tuỳ chọn, mặc định tắt** | Cần server + đồng ý riêng + cập nhật `PRIVACY.md`; hiện dữ liệu sức khoẻ chỉ ở máy |
| M2 | MyTherapy: báo cáo PDF cho bác sĩ | Xuất báo cáo tuân thủ (đã có `adherence_report.dart`) ra PDF, tạo **trên máy** | Cần gói tạo PDF → phải xin duyệt trước khi thêm gói |
| M3 | MyTherapy: quét mã vạch | Thêm quét mã vạch hộp thuốc làm đường vào nhanh | Cần bảng ánh xạ mã vạch → thuốc có nguồn; không bịa. Có thể dùng ML Kit sẵn nếu đã có |
| M4 | MyTherapy: nhắc tái đơn | Đã có cảnh báo sắp hết thuốc; thêm "nhắc đi mua/tái khám" | Làm được hoàn toàn trên máy |
| M5 | Dosecast: chạy offline | Giữ nguyên nguyên tắc dữ liệu cá nhân ở máy; cho tra cứu offline phần danh mục đã nạp | Lợi thế riêng tư cần nêu khi thuyết trình |
| M6 | Dosecast: hoãn nhắc linh hoạt | Đã có "Hoãn"; thêm hoãn theo các mức 10/30/60 phút và tự tắt tiếng giờ ngủ | Không cần quyền mới |
| M7 | Medisafe: tích hợp lịch | Thêm lịch tái khám vào lịch điện thoại (người dùng bấm) | Cần quyền lịch → **hỏi trước khi mở rộng quyền** |

## 4. Điểm yếu của đối thủ → MedAssist cải thiện/nâng cấp

| # | Điểm yếu | MedAssist đã có hoặc nên làm | Trạng thái |
|---|---|---|---|
| Y1 | Tường trả phí, giới hạn số thuốc (Medisafe) | Không giới hạn số thuốc, không khoá lịch sử; nêu rõ trên slide so sánh | Đã có sẵn, chỉ cần thể hiện |
| Y2 | Cảnh báo tương tác không rõ nguồn | Luôn từ chối khi chưa có nguồn kiểm chứng và nói rõ lý do; chỉ mở khi có nguồn + dược sĩ duyệt | Đã có (`check_interaction` luôn từ chối) |
| Y3 | OCR sai thì ghi nhầm thuốc/liều | **Không tự lưu**: bộ đọc theo luật đề xuất, người dùng xác nhận mới lưu; thêm nhãn "độ chắc chắn thấp" và tô đậm trường nghi ngờ | Một phần đã có; phần nhãn độ chắc chắn nên làm |
| Y4 | Chữ viết tay đọc rất kém (25,7%) | Nói thẳng trong app: đơn viết tay → nhập/kiểm tra tay; thêm bước "đối chiếu với tên thuốc trong danh mục DAV" để bắt lỗi chính tả | **Nên làm** — danh mục DAV 52.201 thuốc đã sẵn để đối chiếu |
| Y5 | Nhập tay thiếu hàm lượng (MyTherapy) | Ô hàm lượng/đơn vị riêng, lấy gợi ý từ danh mục DAV (cột `Strength`) | Nên làm |
| Y6 | Bán dữ liệu/quảng cáo/ép đăng ký (nguồn yếu) | Dữ liệu sức khoẻ chỉ ở máy, mã hoá bằng Keystore, PIN, chặn chụp màn hình, xoá khi đổi tài khoản | Đã có |
| Y7 | Giao diện cũ, thiếu chế độ tối (Dosecast) | Thêm chế độ tối + cỡ chữ lớn cho người lớn tuổi | Nên làm; kiểm tra `app_theme.dart` đã có chưa |
| Y8 | Báo cáo khó dùng (Medisafe) | Báo cáo tuân thủ 1 màn hình, đọc được ngay, có xuất PDF | Một phần đã có |
| Y9 | Thông báo không ổn định sau cập nhật | Kiểm tra thông báo sau khi khởi động lại máy; hướng dẫn tắt tối ưu pin; nút trên thông báo hiện chỉ hoạt động khi app còn chạy | **Hạn chế đã biết (HANDOFF mục 6)** |
| Y10 | Không có cơ chế "nói không biết" | Câu trả lời luôn kèm nguồn; không có nguồn → "chưa đủ bằng chứng, hỏi dược sĩ/bác sĩ" | Đã có — **điểm khác biệt lớn nhất** |

## 5. Khoảng trống cần nói thật khi so sánh
- MedAssist **chưa** có: người thân theo dõi, xuất PDF, quét mã vạch, đồng bộ nhiều máy, tích hợp lịch.
- Các đối thủ đã có lượng người dùng lớn và được dùng thật lâu; MedAssist là đồ án, chưa thử với dược sĩ hay đơn thật.
- Dữ liệu công dụng/tác dụng phụ của MedAssist còn là mẫu; sổ DAV mới chỉ có tên, hoạt chất, hàm lượng, nhà sản xuất.

## 6. Việc đề xuất (chủ dự án duyệt tất cả ngày 30/10/2026)
1. ✅ Y4 + Y5: nút "Đối chiếu danh mục thuốc" trên màn xác nhận đơn — chỉ gửi tên thuốc khi người dùng bấm
   (`/v1/search`, công khai, không lưu); hiện tối đa 3 gợi ý kèm "hoạt chất · hàm lượng", bấm "Dùng tên này"
   để thay tên. Không có trong danh mục -> khoá, phải tick "Tôi đã kiểm tra".
   Hàm lượng chỉ **hiển thị** từ danh mục, chưa lưu thành trường riêng (cần đổi schema SQLite).
2. ✅ Y3: nhãn "Độ chắc chắn thấp" (thiếu liều, tên quá ngắn, ký tự lạ/chữ-số lẫn nhau, không có trong danh mục)
   — `lib/domain/ocr_confidence.dart`.
3. ✅ Y7: Hồ sơ > Hiển thị: Giao diện (Theo máy/Sáng/Tối) + Cỡ chữ (Chuẩn/Lớn/Rất lớn), lưu trên máy.
4. ✅ M6: hoãn 10/15/30/60 phút trong app (nút trên thông báo vẫn 15 phút).
   ⏸ M4 nhắc tái khám: cần bảng dữ liệu mới -> chờ kế hoạch đổi schema được duyệt.
   (Cảnh báo sắp hết thuốc / nhắc mua thêm đã có sẵn.)
5. ⏸ M2: xuất PDF — cần chọn gói cụ thể và chủ dự án duyệt tên gói.
6. ⏸ M1, M3, M7: cần quyền/dữ liệu/server mới -> chờ duyệt cụ thể.

## 7. Việc cần kiểm chứng trước khi dùng trong báo cáo
- Giá và giới hạn bản miễn phí hiện tại của Medisafe (trên App Store/Google Play).
- Medisafe có quét mã vạch/ảnh không.
- MyTherapy có thật sự không quảng cáo/không có tính năng trả phí không (nguồn mâu thuẫn).
- Round Health và các app thuốc Việt Nam (chưa nghiên cứu).
