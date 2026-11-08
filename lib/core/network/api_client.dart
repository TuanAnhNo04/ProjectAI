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

/// Làm mới phiên khi access token hết hạn. Trả `true` nếu đã có token mới.
///
/// Phải **single-flight** ở phía cài đặt: server xoay refresh token nên dùng
/// lại token cũ bị coi là đánh cắp (xem `RemoteAuthRepository.refreshSession`).
typedef TokenRefresher = Future<bool> Function();

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
    this.tokenRefresher,
    this.retryDelays = const [
      Duration(milliseconds: 400),
      Duration(milliseconds: 1200),
    ],
    bool? logTraffic,
    List<String>? publicPaths,
  })  : dio = dio ?? Dio(),
        logTraffic = logTraffic ?? kDebugMode,
        publicPaths = List<String>.unmodifiable(
          publicPaths ??
              const [
                '/login',
                '/register',
                '/health',
                '/v1/auth/login',
                '/v1/auth/register',
                '/v1/auth/refresh',
                '/v1/auth/logout',
                '/v1/health',
                '/v1/news',
              ],
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
            tokenRefresher: tokenRefresher,
            dio: this.dio,
            publicPaths: this.publicPaths,
            logTraffic: this.logTraffic,
          ),
        );
  }

  final Dio dio;
  final TokenProvider? tokenProvider;
  final UnauthorizedCallback? onUnauthorized;

  /// Có thì 401 ở endpoint có bảo vệ được thử làm mới phiên + gọi lại 1 lần
  /// trước khi coi là hết phiên.
  final TokenRefresher? tokenRefresher;

  /// Độ trễ giữa các lần thử lại của **GET** khi mất kết nối/server tạm lỗi.
  /// Rỗng = không thử lại. Chỉ GET (idempotent): POST/DELETE thử lại có thể
  /// tạo/xoá hai lần.
  final List<Duration> retryDelays;

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
  ///
  /// [recoverUnauthorized] = `false` cho thao tác mà 401 nghĩa là "sai mật khẩu
  /// xác nhận" chứ không phải "hết phiên" (vd xoá tài khoản): không làm mới phiên,
  /// không đăng xuất người dùng.
  Future<Response<dynamic>> post(
    String path, {
    Object? data,
    Map<String, Object?>? query,
    Map<String, dynamic>? headers,
    CancelToken? cancelToken,
    bool recoverUnauthorized = true,
  }) =>
      request<dynamic>(
        'POST',
        path,
        data: data,
        query: query,
        headers: headers,
        cancelToken: cancelToken,
        recoverUnauthorized: recoverUnauthorized,
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
    bool recoverUnauthorized = true,
  }) async {
    final target = url(path, query);
    final options = Options(
      method: method,
      responseType: responseType,
      headers: headers,
      extra: <String, Object?>{
        'path': path,
        if (!recoverUnauthorized) AuthInterceptor.noRecoveryKey: true,
      },
      receiveTimeout: receiveTimeout,
    );
    try {
      for (var attempt = 0;; attempt++) {
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
          final canRetry =
              method == 'GET' && attempt < retryDelays.length && isTransient(e);
          if (!canRetry) rethrow;
          await Future<void>.delayed(retryDelays[attempt]);
        }
      }
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

  /// Lỗi tạm thời đáng thử lại: mất kết nối, hết thời gian chờ kết nối, hoặc
  /// server báo 502/503/504. **Không** thử lại 4xx/500 hay request bị huỷ.
  @visibleForTesting
  static bool isTransient(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionError:
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
        return true;
      case DioExceptionType.badResponse:
        final code = e.response?.statusCode;
        return code == 502 || code == 503 || code == 504;
      default:
        return false;
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
    this.tokenRefresher,
    this.dio,
    required this.publicPaths,
    required this.logTraffic,
  });

  final TokenProvider? tokenProvider;
  final UnauthorizedCallback? onUnauthorized;
  final TokenRefresher? tokenRefresher;

  /// Cần để gọi lại request sau khi làm mới phiên.
  final Dio? dio;
  final List<String> publicPaths;
  final bool logTraffic;

  static const String startedAtKey = 'apiStartedAt';

  /// Đánh dấu request đã được gọi lại sau làm mới — chặn vòng lặp vô hạn.
  static const String retriedKey = 'apiRetriedAfterRefresh';

  /// Request đặt cờ này thì 401 chỉ là lỗi thường (không làm mới, không đăng xuất).
  static const String noRecoveryKey = 'apiNoUnauthorizedRecovery';

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
  Future<void> onError(
    DioException err,
    ErrorInterceptorHandler handler,
  ) async {
    if (logTraffic) {
      developer.log(
        '← LỖI ${err.response?.statusCode ?? err.type.name} '
        '${err.requestOptions.method} ${_pathOf(err.requestOptions)} '
        '${_elapsed(err.requestOptions)}',
        name: 'API',
      );
    }
    // 401 ở endpoint có bảo vệ: thử làm mới phiên trước, hết cách mới coi là hết
    // phiên và báo tầng auth xoá token (Day 11).
    if (err.response?.statusCode == 401 &&
        !isPublic(err.requestOptions) &&
        err.requestOptions.extra[noRecoveryKey] != true) {
      final options = err.requestOptions;
      final refresher = tokenRefresher;
      final client = dio;
      if (refresher != null &&
          client != null &&
          options.extra[retriedKey] != true &&
          await refresher()) {
        // FormData chỉ gửi được 1 lần -> không gọi lại được; phiên đã được làm
        // mới nên KHÔNG đăng xuất, để người dùng thử lại thao tác.
        if (options.data is FormData) {
          handler.next(err);
          return;
        }
        final token = await tokenProvider?.call();
        options.extra[retriedKey] = true;
        if (token != null && token.isNotEmpty) {
          options.headers['Authorization'] = 'Bearer $token';
        }
        try {
          handler.resolve(await client.fetch<dynamic>(options));
        } on DioException catch (retryError) {
          handler.next(retryError);
        }
        return;
      }
      onUnauthorized?.call();
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
