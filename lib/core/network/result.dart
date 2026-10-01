import 'package:image_processing_app/core/network/api_exception.dart';

/// Kết quả của một phép tính có thể thất bại, **không cần thêm dependency**.
///
/// `sealed` (Dart 3) giúp trình biên dịch bắt buộc xử lý đủ nhánh:
/// ```dart
/// final res = await api.getJson('/history', parser: HistoryItem.listFrom);
/// return switch (res) {
///   Ok(:final value) => value,
///   Err(:final error) => throw error, // hoặc hiển thị error.message
/// };
/// ```
sealed class Result<T> {
  const Result();

  bool get isOk;
  bool get isErr => !isOk;

  /// Giá trị nếu thành công, `null` nếu thất bại.
  T? get valueOrNull;

  /// Lỗi nếu thất bại, `null` nếu thành công.
  AppException? get errorOrNull;

  /// Rẽ nhánh tường minh — ưu tiên dùng cái này hoặc `switch` pattern.
  R map<R>({
    required R Function(T value) ok,
    required R Function(AppException error) err,
  });

  /// Biến đổi giá trị thành công, giữ nguyên lỗi.
  Result<R> mapValue<R>(R Function(T value) transform) => switch (this) {
        Ok<T>(:final value) => Ok<R>(transform(value)),
        Err<T>(:final error) => Err<R>(error),
      };
}

/// Thành công.
final class Ok<T> extends Result<T> {
  const Ok(this.value);

  final T value;

  @override
  bool get isOk => true;

  @override
  T? get valueOrNull => value;

  @override
  AppException? get errorOrNull => null;

  @override
  R map<R>({
    required R Function(T value) ok,
    required R Function(AppException error) err,
  }) =>
      ok(value);

  @override
  String toString() => 'Ok($value)';
}

/// Thất bại — luôn mang [AppException] đã có thông báo tiếng Việt.
///
/// [partial] cho phép luồng "vẫn hiển thị dữ liệu cũ + âm thầm báo lỗi"
/// (dùng ở Day 12 — cache offline).
final class Err<T> extends Result<T> {
  const Err(this.error, {T? partial}) : _partial = partial;

  final AppException error;

  final T? _partial;

  @override
  bool get isOk => false;

  @override
  AppException? get errorOrNull => error;

  @override
  T? get valueOrNull => _partial;

  @override
  R map<R>({
    required R Function(T value) ok,
    required R Function(AppException error) err,
  }) =>
      err(error);

  @override
  String toString() => 'Err(${error.kind.name}: ${error.message})';
}

/// Chạy [body] và bọc ngoại lệ thành `Result` — đặt ở ranh giới adapter để
/// tầng UI/domain không bao giờ phải `try/catch`.
Future<Result<T>> guard<T>(Future<T> Function() body) async {
  try {
    return Ok<T>(await body());
  } on AppException catch (e) {
    return Err<T>(e);
  } on Object catch (e) {
    return Err<T>(AppException.from(e));
  }
}
