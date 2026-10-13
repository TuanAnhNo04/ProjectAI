/// Tiện ích đọc JSON "kiểu an toàn" cho tầng `data`.
///
/// **Vì sao không `json['x'] as String`?** Server Flask đọc dữ liệu từ file CSV
/// nên **thiếu khoá** và **sai kiểu** là chuyện thường ngày:
/// `null`, `""`, `"93%"`, `93`, `null` cho cột không có số liệu...
/// Một phép `as` sai kiểu làm **sập cả màn hình**; ở đây chỉ mất đúng 1 trường.
///
/// **Hai hàm duy nhất được ném** là `asJsonMap`/`asJsonList`: khi body sai hình
/// dạng thì adapter không thể làm gì ngoài báo lỗi — đúng như mô tả sẵn có của
/// `AppExceptionKind.parsing`. Nhờ vậy adapter bọc trong `guard()` là xong.
library;

import 'dart:convert';

import 'package:image_processing_app/core/network/api_exception.dart';

/// Lỗi body không đúng hình dạng mong đợi (không phải lỗi mạng).
AppException parsingError(String detail) => AppException(
      kind: AppExceptionKind.parsing,
      message: 'Dữ liệu từ máy chủ không đúng định dạng mong đợi.',
      detail: detail,
    );

/// Ép về `Map<String, dynamic>`; **ném** nếu không phải object.
Map<String, dynamic> asJsonMap(Object? value, String field) {
  if (value is Map) {
    return value.map((key, v) => MapEntry('$key', v));
  }
  throw parsingError('$field phải là object, nhận được ${value.runtimeType}');
}

/// Ép về list; `null` thành list rỗng, sai kiểu thì **ném**.
///
/// `null -> []` là cố ý: server hay trả `{"search_results": null}` cho lần
/// quét ảnh không tìm thấy thuốc nào, đó là "không có kết quả" chứ không phải lỗi.
List<Object?> asJsonList(Object? value, String field) {
  if (value == null) return const <Object?>[];
  if (value is List) return value;
  throw parsingError(
      '$field phải là danh sách, nhận được ${value.runtimeType}');
}

/// String đã trim; `null`/số/bool đều quy về chuỗi, không bao giờ ném.
String asString(Object? value, {String fallback = ''}) {
  if (value == null) return fallback;
  if (value is String) return value.trim();
  return '$value'.trim();
}

/// String hoặc `null` — chuỗi rỗng cũng thành `null` (server hay trồng `"  "`).
String? asStringOrNull(Object? value) {
  final text = asString(value);
  return text.isEmpty ? null : text;
}

int? asIntOrNull(Object? value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  if (value is String) return int.tryParse(value.trim());
  return null;
}

double? asDoubleOrNull(Object? value) {
  if (value is num) {
    final d = value.toDouble();
    return d.isFinite ? d : null;
  }
  if (value is String) return double.tryParse(value.trim());
  return null;
}

/// Đọc cột "Excellent Review %" mà server có thể trả `"93%"`, `93`, `" 93 %"`
/// hoặc `null` — về **một số 0–100**, hoặc `null` nếu không có số liệu.
double? asPercent(Object? value) {
  final parsed = value is String
      ? double.tryParse(value.replaceAll('%', '').trim())
      : asDoubleOrNull(value);
  if (parsed == null || !parsed.isFinite) return null;
  return parsed.clamp(0, 100).toDouble();
}

/// Ngược của [asPercent] — phát lại đúng kiểu server để `toJson` round-trip.
String? formatPercent(double? value) =>
    value == null ? null : '${value.round()}%';

/// Ngày giờ ISO-8601 (FastAPI trả chuẩn này); `null`/`""`/sai định dạng → `null`.
DateTime? asDateTimeOrNull(Object? value) {
  final text = asStringOrNull(value);
  if (text == null) return null;
  return DateTime.tryParse(text)?.toLocal();
}

/// Đọc body text của response rồi ép thành object — thay cho cặp
/// `jsonDecode(body)` + ép kiểu thủ công trong từng màn hình.
Map<String, dynamic> decodeJsonObject(String body) {
  try {
    return asJsonMap(jsonDecode(body), 'Phản hồi');
  } on FormatException catch (e) {
    throw parsingError('Body không phải JSON hợp lệ: ${e.message}');
  }
}

/// Sentinel cho `copyWith`: phân biệt **"không truyền"** với **"truyền `null`"**.
///
/// `copyWith({String? x})` không phân biệt được hai trường hợp đó — dùng
/// `Object? x = notProvided` là cách duy nhất để `copyWith(x: null)` thật sự
/// **xoá** field thay vì giữ nguyên. Dùng chung một sentinel cho mọi model.
const Object notProvided = Object();

/// Chọn giá trị mới cho `copyWith`, giữ giá trị cũ nếu không truyền gì.
///
/// [current] phải khai báo `T?` chứ không phải `T`. Nếu khai báo `T current`,
/// Dart suy luận `T` từ **ngữ cảnh kiểu trả về** trước (`T?` ← `String?` → suy ra
/// `T = String`) rồi từ chối nhận giá trị hiện có kiểu `String?`. Khai báo
/// `T? current` khiến suy luận bám đúng tham số.
T? pickChanged<T>(Object? candidate, T? current) =>
    identical(candidate, notProvided) ? current : candidate as T?;
