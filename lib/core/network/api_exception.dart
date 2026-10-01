import 'dart:convert';

import 'package:dio/dio.dart';

/// Nhóm lỗi mà tầng UI cần phân biệt để quyết định cách hiển thị.
enum AppExceptionKind {
  /// Người dùng/hệ thống chủ động huỷ request — **không** nên báo lỗi.
  canceled,

  /// Không kết nối được trong thời gian cho phép.
  connectionTimeout,

  /// Gửi dữ liệu (upload ảnh) quá lâu.
  sendTimeout,

  /// Server nhận request nhưng trả lời quá lâu.
  receiveTimeout,

  /// Không có mạng, sai host, DNS lỗi, server chưa bật, cleartext bị chặn.
  network,

  /// 401 — token sai/hết hạn hoặc sai thông tin đăng nhập.
  unauthorized,

  /// 403 — đăng nhập đúng nhưng không đủ quyền.
  forbidden,

  /// 404 — sai đường dẫn hoặc bản ghi không còn tồn tại.
  notFound,

  /// 400/422 — dữ liệu gửi lên không hợp lệ.
  validation,

  /// 409 — trùng dữ liệu (vd tên đăng ký đã tồn tại).
  conflict,

  /// 5xx — lỗi phía server.
  server,

  /// Server trả về 2xx nhưng thân trả về không đúng định dạng mong đợi.
  parsing,

  /// Những thứ còn lại: lỗi lập trình, bất thường không phân loại.
  unknown,
}

/// Ngoại lệ mạng duy nhất mà app ném ra, kèm **thông báo tiếng Việt** đã sẵn
/// sàng để đưa vào `SnackBar`.
///
/// [message] luôn ngắn và thân thiện; [detail] giữ mô tả kỹ thuật (nếu server
/// gửi) để log/debug — **không** hiển thị cho người dùng.
class AppException implements Exception {
  const AppException({
    required this.kind,
    required this.message,
    this.statusCode,
    this.detail,
    this.uri,
    this.cause,
  });

  /// Biến bất kỳ ngoại lệ nào (kể cả `DioException`) thành `AppException`.
  factory AppException.from(Object error) => switch (error) {
        final AppException e => e,
        final DioException e => AppException.fromDio(e),
        _ => AppException(
            kind: AppExceptionKind.unknown,
            message: 'Ứng dụng gặp sự cố, vui lòng thử lại.',
            cause: error,
          ),
      };

  /// Ánh xạ `DioException` -> thông điệp phù hợp với từng loại lỗi.
  factory AppException.fromDio(DioException e) {
    final uri = e.requestOptions.uri;
    final status = e.response?.statusCode;
    if (status != null) {
      return AppException(
        kind: AppException.fromStatus(status),
        message: messageForStatus(status),
        statusCode: status,
        detail: serverMessage(e.response?.data),
        uri: uri,
        cause: e,
      );
    }

    final kind = switch (e.type) {
      DioExceptionType.connectionTimeout => AppExceptionKind.connectionTimeout,
      DioExceptionType.sendTimeout => AppExceptionKind.sendTimeout,
      DioExceptionType.receiveTimeout => AppExceptionKind.receiveTimeout,
      DioExceptionType.badCertificate => AppExceptionKind.network,
      DioExceptionType.cancel => AppExceptionKind.canceled,
      DioExceptionType.connectionError => AppExceptionKind.network,
      _ => AppExceptionKind.unknown,
    };
    return AppException(
      kind: kind,
      message: messageForKind(kind, uri),
      detail: e.error?.toString(),
      uri: uri,
      cause: e,
    );
  }

  final AppExceptionKind kind;

  /// Câu chữ hiển thị cho người dùng.
  final String message;

  /// Mã HTTP, `null` nếu lỗi xảy ra trước khi có phản hồi.
  final int? statusCode;

  /// Mô tả kỹ thuật (message do server trả / exception gốc) — chỉ để log.
  final String? detail;

  /// URI đã gọi, giúp đoán nhanh sai `API_BASE_URL`.
  final Uri? uri;

  final Object? cause;

  /// 401 -> tầng lưu token phải xoá token và đưa người dùng về Login (Day 11).
  bool get isUnauthorized => kind == AppExceptionKind.unauthorized;

  /// Thử lại vẫn có cơ hội thành công (mất mạng chập chờn, 5xx, timeout).
  bool get isRetryable => switch (kind) {
        AppExceptionKind.network ||
        AppExceptionKind.connectionTimeout ||
        AppExceptionKind.sendTimeout ||
        AppExceptionKind.receiveTimeout ||
        AppExceptionKind.server =>
          true,
        _ => false,
      };

  /// Không nên hiện lỗi (request bị huỷ là chuyện bình thường).
  bool get isSilent => kind == AppExceptionKind.canceled;

  static AppExceptionKind fromStatus(int status) {
    if (status == 401) return AppExceptionKind.unauthorized;
    if (status == 403) return AppExceptionKind.forbidden;
    if (status == 404) return AppExceptionKind.notFound;
    if (status == 409) return AppExceptionKind.conflict;
    if (status == 400 || status == 422) return AppExceptionKind.validation;
    if (status >= 500) return AppExceptionKind.server;
    return AppExceptionKind.unknown;
  }

  /// Message mặc định theo mã HTTP. Nếu server gửi `message`/`error` trong body
  /// thì [serverMessage] được ưu tiên hơn.
  static String messageForStatus(int status) => switch (fromStatus(status)) {
        AppExceptionKind.unauthorized =>
          'Phiên đăng nhập không hợp lệ hoặc đã hết hạn. Vui lòng đăng nhập lại.',
        AppExceptionKind.forbidden =>
          'Bạn không có quyền thực hiện thao tác này.',
        AppExceptionKind.notFound => 'Không tìm thấy dữ liệu bạn yêu cầu.',
        AppExceptionKind.conflict => 'Dữ liệu đã tồn tại trong hệ thống.',
        AppExceptionKind.validation => 'Thông tin vừa nhập chưa hợp lệ.',
        AppExceptionKind.server =>
          'Máy chủ đang gặp lỗi, vui lòng thử lại sau.',
        _ => 'Có lỗi xảy ra (mã $status).',
      };

  /// Message mặc định khi chưa có mã HTTP.
  static String messageForKind(AppExceptionKind kind, [Uri? uri]) {
    final host = uri == null || uri.host.isEmpty ? 'máy chủ' : uri.host;
    return switch (kind) {
      AppExceptionKind.network =>
        'Không kết nối được tới $host. Kiểm tra máy chủ đã chạy và thiết bị có mạng.',
      AppExceptionKind.connectionTimeout ||
      AppExceptionKind.sendTimeout =>
        'Mất quá lâu để gửi dữ liệu tới $host. Kiểm tra mạng rồi thử lại.',
      AppExceptionKind.receiveTimeout =>
        '$host phản hồi quá chậm. Vui lòng thử lại.',
      AppExceptionKind.canceled => 'Yêu cầu đã bị huỷ.',
      _ => 'Ứng dụng gặp sự cố, vui lòng thử lại.',
    };
  }

  /// Múc thông báo kỹ thuật từ thân phản hồi.
  ///
  /// Flask cũ trả `{"message": "..."}` hoặc HTML lỗi; FastAPI (Day 15) trả
  /// `{"detail": ...}`. HTML (bắt đầu bằng `<`) bị bỏ qua vì vô nghĩa với
  /// người dùng.
  static String? serverMessage(Object? data) {
    final raw = switch (data) {
      final Map<Object?, Object?> map => _firstText(map, _messageKeys),
      final String s => _looksLikeJson(s) ? _fromJsonString(s) : null,
      _ => null,
    };
    if (raw == null) return null;
    final flat = raw.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (flat.isEmpty) return null;
    return flat.length <= 240 ? flat : '${flat.substring(0, 240)}…';
  }

  static const List<String> _messageKeys = [
    'message',
    'error',
    'detail',
    'msg',
    'error_description',
  ];

  static String? _firstText(Map<Object?, Object?> map, List<String> keys) {
    for (final key in keys) {
      final value = map[key];
      if (value is String && value.trim().isNotEmpty) return value;
      if (value is List && value.isNotEmpty) {
        final parts = value
            .map((e) => e is Map ? '${e['msg'] ?? e['message'] ?? e}' : '$e')
            .join('; ');
        if (parts.trim().isNotEmpty) return parts;
      }
    }
    return null;
  }

  static bool _looksLikeJson(String s) {
    final t = s.trimLeft();
    return t.startsWith('{') || t.startsWith('[');
  }

  static String? _fromJsonString(String s) {
    try {
      final decoded = jsonDecode(s);
      if (decoded is Map<Object?, Object?>) {
        return _firstText(decoded, _messageKeys);
      }
      return null;
    } on FormatException {
      return null;
    }
  }

  @override
  String toString() => 'AppException(${kind.name}'
      '${statusCode == null ? '' : ', $statusCode'}): '
      '${detail == null ? message : '$message | $detail'}';
}
