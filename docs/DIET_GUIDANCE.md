# Gợi ý ăn uống theo bệnh nền

`POST /v1/agent/lifestyle` nhận `conditions` (app chỉ gửi khi người dùng đã đồng ý dùng hồ sơ).
Với mỗi bệnh nền khớp một dòng trong CSV, server trả một gợi ý `kind: "diet"` kèm trích dẫn
(`D1`, `D2`...: `source`, `excerpt`, `is_authoritative`). Bệnh nền không khớp dòng nào thì KHÔNG
có gợi ý, và được liệt kê trong `unmatched_conditions` để app báo "chưa có tài liệu cho: ...".
Tối đa 5 gợi ý; thứ tự: thuốc có trích dẫn, ăn uống, chung.

## Dữ liệu mẫu, cần thay

`data/diet_guidance.dev.csv` chỉ là MẪU (5 bệnh, `is_authoritative=false`, nguồn ghi rõ
"chưa kiểm chứng"). Trước khi dùng thật hãy thay bằng tài liệu chính thống và trỏ tới file đó:

```
MEDASSIST_DIET_CSV=/duong/dan/file.csv
```

File thay thế có thể đặt `is_authoritative=true`. Thiếu file thì không có gợi ý ăn uống, không lỗi.

## Định dạng CSV (UTF-8, có dòng tiêu đề)

`id,condition_keywords,title,advice,source,is_authoritative`

- `condition_keywords`: các thuật ngữ cách nhau bằng `|`; khớp không phân biệt dấu/hoa thường,
  theo từ trọn vẹn. Từ khoá đầu tiên là nhãn hiển thị trong trích dẫn.
- `source`: tên tài liệu thật đã nạp; không bịa nguồn.

Thêm bệnh mới = thêm một dòng; không cần sửa code.

## Quy tắc an toàn

- Không con số, không liều, không khẳng định thức ăn-thuốc, không khuyên đổi/ngưng thuốc.
- Mỗi lời khuyên kết bằng ý "hỏi bác sĩ/chuyên gia dinh dưỡng về chế độ phù hợp với bạn".
- Không khớp thì không có gợi ý (không bao giờ dùng lời khuyên chung thay thế).
- Chuỗi bệnh nền là dữ liệu không tin cậy: chỉ dùng để so khớp, không bị log, không lưu.
- AI đám mây (nếu bật) không được nhận hay sửa gợi ý ăn uống; nội dung và trích dẫn giữ nguyên bản CSV.
