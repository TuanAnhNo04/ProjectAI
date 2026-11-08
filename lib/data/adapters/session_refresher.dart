import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/user.dart';

/// Làm mới phiên bằng refresh token đã lưu (xoay cặp token).
///
/// Tách khỏi [RemoteAuthRepository] vì **vòng phụ thuộc**: `ApiClient` cần làm mới phiên
/// khi gặp 401, còn repository cần `ApiClient` để gọi mạng. Nếu `ApiClient` đọc repository
/// qua Riverpod thì ném `CircularDependencyError` (đã gặp khi chạy thật trên máy ảo).
/// Giải pháp: refresher dùng một [ApiClient] **riêng, không có refresher** (nên không đệ
/// quy) và chỉ phụ thuộc vào kho token.
///
/// **Single-flight**: nhiều request cùng gặp 401 chỉ làm mới MỘT lần — server xoay refresh
/// token, dùng lại token cũ bị coi là bị đánh cắp và thu hồi cả chuỗi phiên. Vì vậy cả
/// `ApiClient` lẫn repository phải dùng chung **một** instance.
class SessionRefresher {
  SessionRefresher(this._api, this._store, {required String refreshPath})
      : _refreshPath = refreshPath;

  final ApiClient _api;
  final TokenStore _store;
  final String _refreshPath;

  Future<bool>? _inFlight;

  /// `true` khi đã có cặp token mới (đã lưu); `false` khi không làm mới được. Không ném.
  Future<bool> refresh() =>
      _inFlight ??= _doRefresh().whenComplete(() => _inFlight = null);

  Future<bool> _doRefresh() async {
    final refresh = await _store.readRefresh();
    if (refresh == null || refresh.isEmpty) return false;
    try {
      final response = await _api.post(
        _refreshPath,
        data: <String, String>{'refresh_token': refresh},
      );
      final session =
          AuthSession.fromJson(asJsonMap(response.data, _refreshPath));
      if (!session.isAuthenticated) return false;
      await _store.write(session.token);
      await _store.writeRefresh(session.refreshToken);
      return true;
    } on AppException {
      return false;
    } on Object {
      return false;
    }
  }
}
