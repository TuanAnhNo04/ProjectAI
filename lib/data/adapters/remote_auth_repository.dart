import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/session_refresher.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/user.dart';
import 'package:image_processing_app/domain/ports/auth_repository.dart';

/// Nơi lưu token của phiên đăng nhập.
///
/// Cố ý là một cổng nhỏ tách khỏi adapter: Day 8 dùng bản **in-memory** đủ để
/// chạy toàn luồng cũ, Day 11 thay bằng bản đọc `flutter_secure_storage` mà
/// **không sửa** `RemoteAuthRepository`. Cũng nhờ vậy adapter không import
/// Flutter và test chạy được trên Dart VM thuần.
abstract interface class TokenStore {
  /// Token hiện tại, `null` nếu chưa đăng nhập. Không bao giờ ném.
  Future<String?> read();

  /// Ghi token mới (`null` để xoá).
  Future<void> write(String? token);

  /// Refresh token (chỉ `medassist-core` có), `null` nếu không có. Không ném.
  Future<String?> readRefresh();

  /// Ghi refresh token mới (`null` để xoá).
  Future<void> writeRefresh(String? token);

  /// Xoá **cả** access lẫn refresh token — an toàn khi gọi lại nhiều lần.
  Future<void> clear();
}

/// Bản [TokenStore] đơn giản nhất: giữ token trong biến của tiến trình.
///
/// ⚠️ Chỉ dùng cho Day 8 (chạy được luồng cũ, kể cả khi tắt app thì mất token).
/// Day 11 thay bằng lưu trữ bền vững + mã hoá.
class InMemoryTokenStore implements TokenStore {
  String? _token;
  String? _refresh;

  @override
  Future<String?> read() async => _token;

  @override
  Future<void> write(String? token) async => _token = token;

  @override
  Future<String?> readRefresh() async => _refresh;

  @override
  Future<void> writeRefresh(String? token) async => _refresh = token;

  @override
  Future<void> clear() async {
    _token = null;
    _refresh = null;
  }
}

/// Adapter đăng nhập gọi **Flask hiện tại** (`/login`, `/register`).
///
/// Body gửi lên dựng từ [User.toCredentialsJson]: mật khẩu là tham số rồi biến
/// mất, không được nhét vào bất kỳ object nào sống lâu.
class RemoteAuthRepository implements AuthRepository {
  RemoteAuthRepository(
    this._api, {
    TokenStore? store,
    String loginPath = '/login',
    String registerPath = '/register',
    String logoutPath = '/logout',
    String refreshPath = '/v1/auth/refresh',
    String deleteAccountPath = '/v1/auth/delete-account',
    String? mePath = '/v1/auth/me',
    SessionRefresher? refresher,
  })  : _store = store ?? InMemoryTokenStore(),
        _loginPath = loginPath,
        _registerPath = registerPath,
        _logoutPath = logoutPath,
        _refreshPath = refreshPath,
        _deleteAccountPath = deleteAccountPath,
        _mePath = mePath,
        _refresherOverride = refresher;

  final ApiClient _api;
  final TokenStore _store;
  final String _loginPath;
  final String _registerPath;
  final String _logoutPath;
  final String _refreshPath;
  final String _deleteAccountPath;
  final String? _mePath;
  final SessionRefresher? _refresherOverride;

  /// Dùng chung với `ApiClient` (qua providers) để single-flight; mặc định tự tạo trên
  /// chính `_api` — đủ cho test và cho nối tay không qua Riverpod.
  late final SessionRefresher _refresher = _refresherOverride ??
      SessionRefresher(_api, _store, refreshPath: _refreshPath);

  @override
  Future<AuthSession> login(String username, String password) async {
    try {
      final response = await _api.post(
        _loginPath,
        data: User(username: username).toCredentialsJson(password: password),
      );
      return await _accept(response.data, _loginPath);
    } on AppException catch (e) {
      if (e.kind != AppExceptionKind.unauthorized) rethrow;
      // ⚠️ Thông điệp mặc định của 401 là "phiên hết hạn, đăng nhập lại" — SAI ngữ cảnh ở
      // màn đăng nhập (người dùng chưa có phiên nào): sai mật khẩu hoặc tài khoản không tồn tại.
      throw AppException(
        kind: e.kind,
        message: 'Tên đăng nhập hoặc mật khẩu không đúng.',
        statusCode: e.statusCode,
        detail: e.detail,
        uri: e.uri,
        cause: e.cause,
      );
    }
  }

  @override
  Future<AuthSession> register(
    String username,
    String password,
    String email,
  ) async {
    final response = await _api.post(
      _registerPath,
      data: User(username: username, email: email)
          .toCredentialsJson(password: password),
    );
    return _accept(response.data, _registerPath);
  }

  /// Xoá token cục bộ rồi **cố** gọi `/logout` phía server.
  ///
  /// Không bao giờ ném: hàm này còn được gọi tự động khi gặp 401 (Day 11), lúc
  /// đó server có thể cũng đang lỗi. Token cục bộ đã xoá là điều quan trọng
  /// nhất; gọi server chỉ là dọn dẹp thêm. An toàn khi gọi lại nhiều lần.
  @override
  Future<void> logout() async {
    final refresh = await _store.readRefresh();
    await _store.clear();
    try {
      await _api.post(
        _logoutPath,
        data:
            refresh == null ? null : <String, String>{'refresh_token': refresh},
      );
    } on AppException {
      // Server cũ chưa có /logout (404) hoặc đang mất mạng — bỏ qua.
    }
  }

  @override
  Future<String?> currentToken() => _store.read();

  @override
  Future<User?> currentUser() async {
    final path = _mePath;
    if (path == null) return null;
    try {
      final response = await _api.get(path);
      return User.fromJson(asJsonMap(response.data, path));
    } on Object {
      return null;
    }
  }

  @override
  Future<bool> refreshSession() => _refresher.refresh();

  @override
  Future<void> deleteAccount(String password) async {
    try {
      await _api.post(
        _deleteAccountPath,
        data: <String, String>{'password': password},
        // 401 ở đây = sai mật khẩu, không phải hết phiên: đừng đăng xuất người dùng.
        recoverUnauthorized: false,
      );
    } on AppException catch (e) {
      if (e.kind != AppExceptionKind.unauthorized) rethrow;
      // Thông điệp mặc định của 401 là "phiên hết hạn, đăng nhập lại" — sai ngữ cảnh.
      throw AppException(
        kind: e.kind,
        message: 'Mật khẩu không đúng.',
        statusCode: e.statusCode,
        detail: e.detail,
        uri: e.uri,
        cause: e.cause,
      );
    }
    await _store.clear();
  }

  /// Đọc [AuthSession], lưu token nếu server trả, và coi `{"success": false}`
  /// (kèm HTTP 200) là đăng nhập thất bại — một số bản Flask trả kiểu này.
  Future<AuthSession> _accept(Object? data, String path) async {
    final json = asJsonMap(data, path);
    if (json['success'] == false) {
      throw AppException(
        kind: AppExceptionKind.unauthorized,
        message: AppException.messageForStatus(401),
        detail: AppException.serverMessage(json),
      );
    }

    final session = AuthSession.fromJson(json);
    await _store.write(session.token);
    await _store.writeRefresh(session.refreshToken);
    return session;
  }
}
