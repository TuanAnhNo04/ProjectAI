/// JSON **thật** theo đúng contract của Flask server mà app đang gọi.
///
/// Nguồn: tên trường lấy từ code cũ **đang chạy được**
/// (`lib/screens/*.dart` dùng `item['Medicine_Name']`, `data['medicine_info']`,
/// `data['history']`, `item['search_results']`) — đây là hợp đồng thật.
///
/// ⚠️ Khi server Flask còn chạy, hãy thay bằng body thật:
/// ```powershell
/// curl.exe -s "http://10.0.2.2:5000/search-medicine?name=panadol" > search.json
/// ```
/// rồi dán vào đây. Các giá trị `null` / `""` / số thay chuỗi bên dưới là để
/// **cố ý** mô phỏng dữ liệu CSV thực tế (thiếu ô, ô rỗng, cột review là số).
library;

/// `GET /search-medicine?name=panadol` — thuốc 2 có `Uses = null`,
/// `Side_effects = ""` và review là **số** chứ không phải chuỗi `"93%"`.
const String searchMedicineBody = '''
{
  "results": [
    {
      "Medicine_Name": "Panadol 500mg",
      "Composition": "Paracetamol 500mg",
      "Uses": "Pain relief, fever reduction",
      "Side_effects": "Nausea, allergic reactions",
      "Manufacturer": "GSK",
      "Excellent Review %": "93%",
      "Average Review %": "5%",
      "Poor Review %": "2%"
    },
    {
      "Medicine_Name": "Efferalgan",
      "Composition": "Paracetamol 1g",
      "Uses": null,
      "Side_effects": "",
      "Manufacturer": "UPSA",
      "Excellent Review %": 88,
      "Average Review %": null,
      "Poor Review %": null
    }
  ]
}
''';

/// `POST /process-image` (multipart field `image`).
const String processImageBody = '''
{
  "medicine_info": [
    {
      "Medicine_Name": "Panadol 500mg",
      "Composition": "Paracetamol 500mg",
      "Uses": "Pain relief, fever reduction",
      "Side_effects": "Nausea, allergic reactions",
      "Manufacturer": "GSK",
      "Excellent Review %": "93%",
      "Average Review %": "5%",
      "Poor Review %": "2%"
    }
  ]
}
''';

/// `POST /process-image` — ảnh chụp không đọc được tên thuốc nào.
const String processImageEmptyBody = '{"medicine_info": []}';

/// `GET /history` — mục thứ 2 **không có `id`** (server trả lỗi) và
/// `search_results` rỗng → đúng loại bản ghi mà UI cũ hiện "Invalid ID".
const String historyBody = '''
{
  "history": [
    {
      "id": 1,
      "filepath": "C:/data/user/0/cache/photo_1.jpg",
      "search_results": [
        {
          "Medicine_Name": "Panadol 500mg",
          "Composition": "Paracetamol 500mg",
          "Uses": "Pain relief",
          "Side_effects": "Nausea",
          "Manufacturer": "GSK",
          "Excellent Review %": "93%",
          "Average Review %": "5%",
          "Poor Review %": "2%"
        }
      ]
    },
    {
      "id": 2,
      "filepath": "/data/user/0/cache/photo_2.jpg",
      "search_results": []
    },
    {
      "filepath": "/data/user/0/cache/photo_loi.jpg",
      "search_results": []
    }
  ]
}
''';

/// `POST /login` — bản có token + user lồng nhau.
const String loginWithTokenBody = '''
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiJ9.demo",
  "user": {
    "id": 7,
    "username": "tuananh",
    "email": "tuan@example.com"
  }
}
''';

/// `POST /login` — bản cũ: chỉ trả message, **không có token**.
const String loginLegacyBody = '{"message": "Login successful"}';

/// `POST /register`.
const String registerBody = '{"success": true, "message": "User registered"}';

/// `POST /login` sai mật khẩu — lỗi do server, không phải parse lỗi.
const String loginWrongPasswordBody = '{"message": "Sai tai khoan hoac mat khau"}';