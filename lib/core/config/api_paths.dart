import 'package:image_processing_app/core/config/app_config.dart';

/// Đường dẫn endpoint theo backend đang dùng.
///
/// App nói chuyện được với hai backend (chọn bằng `--dart-define=BACKEND=...`):
/// - `core` (mặc định): `medassist-core` FastAPI, mọi thứ dưới `/v1`;
/// - `flask`: server Flask cũ / `tools/mock_server` (`/login`, `/search-medicine`…).
///
/// Gom đường dẫn vào đây để adapter không phải biết backend nào đang chạy.
class ApiPaths {
  const ApiPaths({
    required this.login,
    required this.register,
    required this.logout,
    required this.refresh,
    required this.deleteAccount,
    required this.search,
    required this.history,
    required this.analyze,
    this.catalog,
    this.me,
    this.news,
    this.lifestyle,
  });

  static const ApiPaths flask = ApiPaths(
    login: '/login',
    register: '/register',
    logout: '/logout',
    refresh: '/v1/auth/refresh',
    deleteAccount: '/v1/auth/delete-account',
    search: '/search-medicine',
    history: '/history',
    analyze: '/process-image',
  );

  static const ApiPaths core = ApiPaths(
    login: '/v1/auth/login',
    register: '/v1/auth/register',
    logout: '/v1/auth/logout',
    refresh: '/v1/auth/refresh',
    deleteAccount: '/v1/auth/delete-account',
    search: '/v1/search',
    history: '/v1/history',
    analyze: '/v1/analyze',
    catalog: '/v1/medicines',
    me: '/v1/auth/me',
    news: '/v1/news',
    lifestyle: '/v1/agent/lifestyle',
  );

  /// Bộ đường dẫn theo [AppConfig.backend].
  static ApiPaths get current => forBackend(AppConfig.backend);

  static ApiPaths forBackend(String backend) =>
      backend == 'flask' ? flask : core;

  final String login;
  final String register;
  final String logout;
  final String refresh;
  final String deleteAccount;
  final String search;
  final String history;
  final String analyze;

  /// Danh mục thuốc cho OCR trên máy; `null` nếu backend không có.
  final String? catalog;

  /// Thông tin người đang đăng nhập; `null` nếu backend không có.
  final String? me;

  /// Tin tức thuốc/y tế; `null` nếu backend không có (vd Flask cũ).
  final String? news;

  /// Lời khuyên lối sống (cần đăng nhập); `null` nếu backend không có.
  final String? lifestyle;
}
