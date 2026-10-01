import 'dart:developer' as developer;

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import 'package:image_processing_app/core/config/app_config.dart';
import 'package:image_processing_app/core/network/api_exception.dart';

/// Cung cấp token cho interceptor. Day 11 sẽ truyền implementation đọc từ
/// `flutter_secure_storage`; để `null` nghĩa là app chưa dùng auth.
typedef TokenProvider = Future<String?> Function();

/// Được gọi khi server trả 401: xoá token + điều hướng về Login (Day 11).
typedef UnauthorizedCallback = void Function();

/// Mô tả một file cần upload (field `image` mà Flask đang chờ).
class UploadFile {
  const UploadFile({required this.path, this.field = 'image', this.filename});

  /// Đường dẫn tuyệt đối trên thiết bị (từ `image_picker`).
  final String path;

  /// Tên form field — Flask cũ đọc `request.files['image']`.
  final String field;

  /// Tên file gửi đi; `null` -> suy ra từ [path].
  final String? filename;
}

/// Client HTTP duy nhất của app, bọc `Dio`.
///
/// Trách nhiệm:
/// 1. Dựng URL từ [AppConfig.baseUrl] — không màn hình nào tự nối chuỗi nữa.
/// 2. Timeout mặc định + header chung.
/// 3. Interceptor: gắn token (Day 11), log khi debug, báo 401.
/// 4. Đổi mọi lỗi thành [AppException] (ném ra) để `guard()` bọc thành `Result`.
class ApiClient {
  ApiClient({
    Dio? dio,
    this.tokenProvider,
    this.onUnauthorized,
    bool? logTraffic,
    List<String>? publicPaths,
  })  : dio = dio ?? Dio(),
        logTraffic = logTraffic ?? kDebugMode,
        publicPaths = List<String>.unmodifiable(
          publicPaths ?? const ['/login', '/register', '/health'],
        ) {
    final options = this.dio.options;
    options
      ..connectTimeout ??= AppConfig.connectTimeout
      ..receiveTimeout ??= AppConfig.receiveTimeout
      ..headers = <String, dynamic>{
        ...options.headers,
        'Accept': 'application/json'
      };

    // Fail-fast: --dart-define sai chính tả sẽ được chỉ ra ngay lúc khởi động
    // thay vì thất bại mơ hồ ở request đầu tiên.
    for (final problem in AppConfig.validate()) {
      developer.log(problem, name: 'AppConfig');
      assert(false, 'Cấu hình API sai: $problem');
    }

    this.dio.interceptors.add(
          AuthInterceptor(
            tokenProvider: tokenProvider,
            onUnauthorized: onUnauthorized,
            publicPaths: this.publicPaths,
            logTraffic: this.logTraffic,
          ),
        );
  }

  final Dio dio;
  final TokenProvider? tokenProvider;
  final UnauthorizedCallback? onUnauthorized;

  /// Mặc định chỉ bật khi debug, và **luôn che** `Authorization` + mật khẩu.
  final bool logTraffic;

  /// Các đường dẫn không cần token.
  final List<String> publicPaths;

  /// Đóng kết nối — gọi khi provider bị dispose (Day 9).
  void close() => dio.close(force: true);

  /// URL tuyệt đối cho [path] tương đối, có encode `query`.
  /// Truyền URL tuyệt đối (endpoint của bên thứ 3 ở P3) thì giữ nguyên.
  Uri url(String path, [Map<String, Object?>? query]) {
    final absolute = path.startsWith('http://') || path.startsWith('https://');
    return absolute
        ? Uri.parse(path)
        : AppConfig.uri(path, queryParameters: query);
  }

  /// GET thô — ném [AppException]. Tầng UI gọi qua `guard()` (xem `result.dart`).
  Future<Response<dynamic>> get(
    String path, {
    Map<String, Object?>? query,
    Map<String, dynamic>? headers,
    CancelToken? cancelToken,
  }) =>
      request<dynamic>(
        'GET',
        path,
        query: query,
        headers: headers,
        cancelToken: cancelToken,
      );

  /// POST JSON thô — ném [AppException].
  Future<Response<dynamic>> post(
    String path, {
    Object? data,
    Map<String, Object?>? query,
    Map<String, dynamic>? headers,
    CancelToken? cancelToken,
  }) =>
      request<dynamic>(
        'POST',
        path,
        data: data,
        query: query,
        headers: headers,
        cancelToken: cancelToken,
      );

  /// DELETE thô — ném [AppException].
  Future<Response<dynamic>> delete(
    String path, {
    Map<String, Object?>? query,
    Map<String, dynamic>? headers,
    CancelToken? cancelToken,
  }) =>
      request<dynamic>(
        'DELETE',
        path,
        query: query,
        headers: headers,
        cancelToken: cancelToken,
      );

  /// Upload `multipart/form-data`, tương thích `request.files['image']`.
  Future<Response<dynamic>> upload(
    String path, {
    required List<UploadFile> files,
    Map<String, String> fields = const {},
    Map<String, dynamic>? headers,
    ProgressCallback? onSendProgress,
    CancelToken? cancelToken,
  }) async {
    final form = FormData();
    for (final file in files) {
      form.files.add(
        MapEntry(
          file.field,
          await MultipartFile.fromFile(
            file.path,
            filename: file.filename ?? basename(file.path),
          ),
        ),
      );
    }
    for (final entry in fields.entries) {
      form.fields.add(MapEntry(entry.key, entry.value));
    }
    return request<dynamic>(
      'POST',
      path,
      data: form,
      headers: headers,
      cancelToken: cancelToken,
      onSendProgress: onSendProgress,
    );
  }

  /// Lõi của mọi request: dựng URL, gọi Dio, đổi mọi lỗi thành [AppException].
  Future<Response<T>> request<T>(
    String method,
    String path, {
    Object? data,
    Map<String, Object?>? query,
    Map<String, dynamic>? headers,
    ResponseType responseType = ResponseType.json,
    Duration? receiveTimeout,
    CancelToken? cancelToken,
    ProgressCallback? onSendProgress,
    ProgressCallback? onReceiveProgress,
  }) async {
    final target = url(path, query);
    final options = Options(
      method: method,
      responseType: responseType,
      headers: headers,
      extra: <String, Object?>{'path': path},
      receiveTimeout: receiveTimeout,
    );
    try {
      return await dio.request<T>(
        target.toString(),
        data: data,
        options: options,
        cancelToken: cancelToken,
        onSendProgress: onSendProgress,
        onReceiveProgress: onReceiveProgress,
      );
    } on DioException catch (e) {
      throw AppException.fromDio(e);
    } on AppException {
      rethrow;
    } catch (e) {
      throw AppException(
        kind: AppExceptionKind.unknown,
        message: 'Không thực hiện được yêu cầu.',
        detail: '$e',
        uri: target,
        cause: e,
      );
    }
  }

  /// Tên file từ đường dẫn, hỗ trợ cả `\` (Windows) lẫn `/` (thiết bị).
  @visibleForTesting
  static String basename(String path) {
    final normalized = path.replaceAll(r'\', '/');
    final i = normalized.lastIndexOf('/');
    return i < 0 ? normalized : normalized.substring(i + 1);
  }
}

/// Interceptor duy nhất: gắn token, log khi debug, phát hiện 401.
///
/// **Bảo mật:** không bao giờ log `Authorization` hay mật khẩu — danh sách
/// khoá nhạy cảm nằm ở [sensitiveKeys].
class AuthInterceptor extends Interceptor {
  AuthInterceptor({
    this.tokenProvider,
    this.onUnauthorized,
    required this.publicPaths,
    required this.logTraffic,
  });

  final TokenProvider? tokenProvider;
  final UnauthorizedCallback? onUnauthorized;
  final List<String> publicPaths;
  final bool logTraffic;

  static const String startedAtKey = 'apiStartedAt';

  /// Khoá (header hoặc field) không được xuất hiện trong log.
  static const Set<String> sensitiveKeys = {
    'authorization',
    'cookie',
    'password',
    'confirm_password',
    'old_password',
    'new_password',
    'token',
    'access_token',
    'refresh_token',
    'secret',
    'api_key',
  };

  @override
  Future<void> onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    options.extra[startedAtKey] = DateTime.now().microsecondsSinceEpoch;
    if (tokenProvider != null && !isPublic(options)) {
      final alreadySet =
          options.headers.keys.any((k) => k.toLowerCase() == 'authorization');
      if (!alreadySet) {
        final token = await tokenProvider!();
        if (token != null && token.isNotEmpty) {
          options.headers['Authorization'] = 'Bearer $token';
        }
      }
    }
    if (logTraffic) {
      developer.log(
        '→ ${options.method} ${options.uri}'
        '${_bodySuffix(options.data)}'
        '${_headerSuffix(options.headers)}',
        name: 'API',
      );
    }
    handler.next(options);
  }

  @override
  void onResponse(
    Response<dynamic> response,
    ResponseInterceptorHandler handler,
  ) {
    if (logTraffic) {
      developer.log(
        '← ${response.statusCode} ${_elapsed(response.requestOptions)} '
        '${response.requestOptions.method} ${_pathOf(response.requestOptions)}',
        name: 'API',
      );
    }
    handler.next(response);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    if (logTraffic) {
      developer.log(
        '← LỖI ${err.response?.statusCode ?? err.type.name} '
        '${err.requestOptions.method} ${_pathOf(err.requestOptions)} '
        '${_elapsed(err.requestOptions)}',
        name: 'API',
      );
    }
    // 401 ở endpoint có bảo vệ -> báo tầng auth xoá token (Day 11).
    if (err.response?.statusCode == 401 &&
        onUnauthorized != null &&
        !isPublic(err.requestOptions)) {
      onUnauthorized!();
    }
    handler.next(err);
  }

  /// Endpoint công khai: login/register — không gửi token cũ theo.
  bool isPublic(RequestOptions options) {
    final path = _pathOf(options);
    return publicPaths.any((p) => path == p || path.startsWith('$p/'));
  }

  static String _pathOf(RequestOptions options) {
    final raw = (options.extra['path'] as String?) ?? options.path;
    if (raw.startsWith('http://') || raw.startsWith('https://')) {
      final parsed = Uri.tryParse(raw);
      if (parsed != null) return parsed.path;
    }
    return raw;
  }

  String _elapsed(RequestOptions options) {
    final started = options.extra[startedAtKey];
    if (started is! int) return '';
    return '(${(DateTime.now().microsecondsSinceEpoch - started) ~/ 1000}ms)';
  }

  static String _bodySuffix(Object? data) {
    if (data == null) return '';
    return ' body=${redactBody(data)}';
  }

  static String _headerSuffix(Map<String, dynamic> headers) =>
      headers.isEmpty ? '' : ' headers=${redactHeaders(headers)}';

  /// Che giá trị của các khoá nhạy cảm trong header.
  @visibleForTesting
  static Map<String, String> redactHeaders(Map<String, dynamic> headers) => {
        for (final e in headers.entries)
          e.key: sensitiveKeys.contains(e.key.toLowerCase())
              ? '***'
              : '${e.value}',
      };

  /// Che mật khẩu/token trong body, tóm tắt `FormData` (không in nhị phân).
  @visibleForTesting
  static Object? redactBody(Object? body) {
    if (body is FormData) {
      return '<multipart ${body.fields.length} fields, '
          '${body.files.length} files>';
    }
    if (body is Map) {
      return <String, Object?>{
        for (final e in body.entries)
          '${e.key}': sensitiveKeys.contains('${e.key}'.toLowerCase())
              ? '***'
              : redactBody(e.value),
      };
    }
    if (body is List) return body.map(redactBody).toList();
    if (body is String && body.length > 400) {
      return '${body.substring(0, 400)}…';
    }
    return body;
  }
}
