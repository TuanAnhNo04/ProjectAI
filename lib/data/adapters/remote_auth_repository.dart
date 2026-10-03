import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
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

  /// Xoá token — an toàn khi gọi lại nhiều lần.
  Future<void> clear();
}

/// Bản [TokenStore] đơn giản nhất: giữ token trong biến của tiến trình.
///
/// ⚠️ Chỉ dùng cho Day 8 (chạy được luồng cũ, kể cả khi tắt app thì mất token).
/// Day 11 thay bằng lưu trữ bền vững + mã hoá.
class InMemoryTokenStore implements TokenStore {
  String? _token;

  @override
  Future<String?> read() async => _token;

  @override
  Future<void> write(String? token) async => _token = token;

  @override
  Future<void> clear() async => _token = null;
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
  })  : _store = store ?? InMemoryTokenStore(),
        _loginPath = loginPath,
        _registerPath = registerPath,
        _logoutPath = logoutPath;

  final ApiClient _api;
  final TokenStore _store;
  final String _loginPath;
  final String _registerPath;
  final String _logoutPath;

  @override
  Future<AuthSession> login(String username, String password) async {
    final response = await _api.post(
      _loginPath,
      data: User(username: username).toCredentialsJson(password: password),
    );
    return _accept(response.data, _loginPath);
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
    await _store.clear();
    try {
      await _api.post(_logoutPath);
    } on AppException {
      // Server cũ chưa có /logout (404) hoặc đang mất mạng — bỏ qua.
    }
  }

  @override
  Future<String?> currentToken() => _store.read();

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
    return session;
  }
}
