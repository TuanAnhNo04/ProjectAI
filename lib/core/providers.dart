import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/config/api_paths.dart';
import 'package:image_processing_app/core/config/app_config.dart';
import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/device_channel.dart';
import 'package:image_processing_app/data/adapters/hybrid_analyzer.dart';
import 'package:image_processing_app/data/adapters/ondevice_analyzer.dart';
import 'package:image_processing_app/data/adapters/remote_analyzer.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_catalog.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_repository.dart';
import 'package:image_processing_app/data/adapters/remote_lifestyle_repository.dart';
import 'package:image_processing_app/data/adapters/remote_news_repository.dart';
import 'package:image_processing_app/data/adapters/secure_token_store.dart';
import 'package:image_processing_app/data/adapters/session_refresher.dart';
import 'package:image_processing_app/data/adapters/sse_agent_client.dart';
import 'package:image_processing_app/data/local/app_lock_store.dart';
import 'package:image_processing_app/data/local/assistant_memory_store.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/local/cloud_ai_consent_store.dart';
import 'package:image_processing_app/data/local/consent_store.dart';
import 'package:image_processing_app/data/local/health_database.dart';
import 'package:image_processing_app/data/local/last_user_store.dart';
import 'package:image_processing_app/data/local/notification_service.dart';
import 'package:image_processing_app/data/models/chat_message.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/medical_facility.dart';
import 'package:image_processing_app/data/models/prescription.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/data/models/news_article.dart';
import 'package:image_processing_app/data/models/user.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';
import 'package:image_processing_app/domain/ports/auth_repository.dart';
import 'package:image_processing_app/domain/ports/medicine_analyzer.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';
import 'package:image_processing_app/domain/ports/lifestyle_repository.dart';
import 'package:image_processing_app/domain/ports/news_repository.dart';
import 'package:image_processing_app/screens/chat_controller.dart';

/// Day 9 — điểm lắp ráp duy nhất của app (composition root).
///
/// Trước Day 9 mỗi màn hình tự `new` adapter của riêng nó, nên không test được
/// và không đổi được backend. Giờ toàn bộ phụ thuộc được khai báo **một lần** ở
/// đây; màn hình chỉ `watch` và không biết Flask tồn tại.
///
/// Đổi backend = sửa đúng một chỗ: thay dòng `Remote…` bên dưới bằng adapter
/// khác (ví dụ `OnDeviceAnalyzer` ở Day 26) mà **không sửa màn hình nào**.

/// Báo "phiên hết hạn" — khai báo ở `main.dart` để [ApiClient] gọi khi gặp 401.
///
/// Dùng callback thay vì import trực tiếp `Navigator`: tầng `core` không được
/// biết màn hình nào đang mở (Day 9 đã cấm điều này với tầng `domain`).
final unauthorizedNotifierProvider = Provider<void Function()>(
  (ref) => () {},
  name: 'unauthorizedNotifier',
);

/// Nơi lưu token — Day 11 dùng [SecureTokenStore] (kho bí mật của hệ điều
/// hành). Web không mã hoá nên rơi về bộ nhớ tạm cho tới khi có phương án khác.
final tokenStoreProvider = Provider<TokenStore>(
  (ref) => SecureTokenStore.isSecurelyAvailable
      ? SecureTokenStore()
      : InMemoryTokenStore(),
  name: 'tokenStore',
);

/// Client mạng duy nhất. Đóng kết nối khi provider bị dispose để không rò
/// socket khi hot restart / test chạy lại nhiều lần.
final Provider<ApiClient> apiClientProvider = Provider<ApiClient>(
  (ref) {
    final client = ApiClient(
      tokenProvider: () => ref.read(tokenStoreProvider).read(),
      // Access token sống ngắn (15 phút ở `medassist-core`): hết hạn thì làm mới
      // bằng refresh token rồi gọi lại, thay vì đá người dùng ra màn Login.
      // KHÔNG đọc `authRepositoryProvider` ở đây: nó phụ thuộc `apiClientProvider` nên
      // gây `CircularDependencyError` khi refresh chạy thật (xem SessionRefresher).
      tokenRefresher: () => ref.read(sessionRefresherProvider).refresh(),
      // 401 ở endpoint có bảo vệ -> xoá token cục bộ ngay, để lần gọi sau
      // không gửi "Bearer <token hết hạn>" nữa.
      onUnauthorized: ref.read(unauthorizedNotifierProvider),
    );
    ref.onDispose(client.close);
    return client;
  },
  name: 'apiClient',
);

/// Bộ làm mới phiên dùng chung (single-flight) cho `ApiClient` và `AuthRepository`.
///
/// Có `ApiClient` **riêng, không có refresher** để việc gọi `/refresh` không tự đệ quy.
final sessionRefresherProvider = Provider<SessionRefresher>(
  (ref) {
    final plain = ApiClient(retryDelays: const []);
    ref.onDispose(plain.close);
    return SessionRefresher(
      plain,
      ref.watch(tokenStoreProvider),
      refreshPath: ApiPaths.current.refresh,
    );
  },
  name: 'sessionRefresher',
);

final Provider<AuthRepository> authRepositoryProvider =
    Provider<AuthRepository>(
  (ref) {
    final paths = ApiPaths.current;
    return RemoteAuthRepository(
      ref.watch(apiClientProvider),
      store: ref.watch(tokenStoreProvider),
      loginPath: paths.login,
      registerPath: paths.register,
      logoutPath: paths.logout,
      refreshPath: paths.refresh,
      deleteAccountPath: paths.deleteAccount,
      mePath: paths.me,
      refresher: ref.watch(sessionRefresherProvider),
    );
  },
  name: 'authRepository',
);

final medicineRepositoryProvider = Provider<MedicineRepository>(
  (ref) {
    final paths = ApiPaths.current;
    return RemoteMedicineRepository(
      ref.watch(apiClientProvider),
      searchPath: paths.search,
      historyPath: paths.history,
    );
  },
  name: 'medicineRepository',
);

/// Feature flag Day 26: `--dart-define=ANALYZER=hybrid` | `ondevice` | `remote`.
///
/// Không đặt thì: backend `flask` -> `remote` (server cũ có `/process-image`);
/// backend `core` -> `ondevice`, vì `medassist-core` chưa bật `/v1/analyze` (501).
const String _analyzerFlag = String.fromEnvironment('ANALYZER');

String get _analyzerMode => _analyzerFlag.isNotEmpty
    ? _analyzerFlag
    : (AppConfig.backend == 'flask' ? 'remote' : 'ondevice');

/// Danh sách thuốc nạp tay — dùng khi không tải được danh mục từ server.
final medicineDatabaseProvider = StateProvider<List<Medicine>>(
  (ref) => const [],
  name: 'medicineDatabase',
);

/// Danh mục thuốc cho OCR trên máy: tải từ server, lưu cache để lần sau dùng
/// được khi **mất mạng**; không có cả hai thì rỗng (OCR không khớp được gì).
///
/// Là dữ liệu công khai, không gắn người dùng — khác lịch sử.
final medicineCatalogProvider = FutureProvider<List<Medicine>>(
  (ref) async {
    final path = ApiPaths.current.catalog;
    if (path == null) return const <Medicine>[];
    final cache = ref.watch(offlineCacheProvider);
    try {
      final list = await RemoteMedicineCatalog(
        ref.watch(apiClientProvider),
        path: path,
      ).fetchAll();
      await cache.save(CacheKeys.catalog, <String, dynamic>{
        'results': list.map((m) => m.toJson()).toList(),
      });
      return list;
    } on Object {
      final raw = await cache.load(CacheKeys.catalog);
      if (raw == null) return const <Medicine>[];
      return Medicine.listFromJson(raw['results'], 'results');
    }
  },
  name: 'medicineCatalog',
);

/// Cổng tin tức — `null` nếu backend hiện tại không có endpoint này (vd Flask cũ).
final newsRepositoryProvider = Provider<NewsRepository?>(
  (ref) {
    final path = ApiPaths.current.news;
    if (path == null) return null;
    return RemoteNewsRepository(ref.watch(apiClientProvider), path: path);
  },
  name: 'newsRepository',
);

/// Tin tức thuốc/y tế: tải từ server, lưu cache để đọc lại khi **mất mạng**.
///
/// Giống [medicineCatalogProvider]: dữ liệu công khai, không gắn người dùng, nên
/// cache không phải lo lộ bí mật khi đăng xuất.
final newsProvider = FutureProvider<NewsSnapshot>(
  (ref) async {
    final repo = ref.watch(newsRepositoryProvider);
    final cache = ref.watch(offlineCacheProvider);
    if (repo == null) {
      return const NewsSnapshot(articles: [], isSample: true);
    }
    try {
      final snapshot = await repo.fetchNews();
      await cache.save(CacheKeys.news, <String, dynamic>{
        'articles': snapshot.articles.map((a) => a.toJson()).toList(),
        'is_sample': snapshot.isSample,
        'failed_sources': snapshot.failedSources,
        'disclaimer': snapshot.disclaimer,
        'verified_sources': snapshot.verifiedSources,
      });
      return snapshot;
    } on Object {
      final raw = await cache.load(CacheKeys.news);
      if (raw == null) rethrow; // chưa có cache -> để màn hình hiện lỗi thật
      return NewsSnapshot.fromJson(raw);
    }
  },
  name: 'news',
);

/// Cổng lời khuyên lối sống — `null` nếu backend không có (Flask cũ).
final lifestyleRepositoryProvider = Provider<LifestyleRepository?>(
  (ref) {
    final path = ApiPaths.current.lifestyle;
    if (path == null) return null;
    return RemoteLifestyleRepository(ref.watch(apiClientProvider), path: path);
  },
  name: 'lifestyleRepository',
);

/// Lời khuyên cho bạn. Request chỉ gồm tên thuốc, bệnh nền (nếu hồ sơ có consent)
/// và cờ đồng ý AI đám mây; KHÔNG gửi lịch sử chat. Cache để xem khi mất mạng
/// (cache bị xoá khi đăng xuất). Làm mới thủ công: `ref.invalidate(lifestyleProvider)`.
final lifestyleProvider = FutureProvider<LifestyleSnapshot>(
  (ref) async {
    final repo = ref.watch(lifestyleRepositoryProvider);
    if (repo == null) return const LifestyleSnapshot(tips: []);
    final cache = ref.watch(offlineCacheProvider);
    final reminders =
        ref.watch(reminderControllerProvider).valueOrNull ?? const [];
    final profile = ref.watch(healthProfileControllerProvider).valueOrNull;
    final consent =
        ref.watch(cloudAiConsentControllerProvider).valueOrNull?.granted ??
            false;
    final request = LifestyleRequest.build(
      reminderMedicineNames: [for (final r in reminders) r.medicineName],
      profile: profile,
      cloudAiConsent: consent,
    );
    try {
      final snapshot = await repo.fetchTips(request);
      await cache.save(CacheKeys.lifestyle, snapshot.toJson());
      return snapshot;
    } on Object {
      final raw = await cache.load(CacheKeys.lifestyle);
      if (raw == null) rethrow;
      return LifestyleSnapshot.fromJson(raw);
    }
  },
  name: 'lifestyle',
);

final analyzerProvider = Provider<MedicineAnalyzer>(
  (ref) {
    final api = ref.watch(apiClientProvider);
    final remote = RemoteAnalyzer(api, path: ApiPaths.current.analyze);
    if (_analyzerMode == 'remote') return remote;

    final catalog = ref.watch(medicineCatalogProvider).valueOrNull;
    final db = (catalog != null && catalog.isNotEmpty)
        ? catalog
        : ref.watch(medicineDatabaseProvider);
    final local = OnDeviceAnalyzer(medicineDatabase: db);
    ref.onDispose(local.dispose);

    if (_analyzerMode == 'ondevice') return local;

    // hybrid (mặc định khi không phải remote/ondevice)
    return HybridAnalyzer(
      local: local,
      remote: remote,
      cache: ref.watch(offlineCacheProvider),
    );
  },
  name: 'analyzer',
);

/// Nhãn tầng nhận diện đang chạy (`'remote-flask'`, `'ondevice-mlkit'`, `'hybrid'`).
final analyzerCapabilityProvider = Provider<String>(
  (ref) => ref.watch(analyzerProvider).capability,
  name: 'analyzerCapability',
);

/// Cache offline (Day 12) — `shared_preferences`, **không** chứa bí mật.
///
/// Bọc trong `Provider` riêng để test override bằng `InMemoryCacheStore` mà
/// không chạm đĩa.
final cacheStoreProvider = Provider<CacheStore>(
  (ref) => PreferencesCacheStore(),
  name: 'cacheStore',
);

final offlineCacheProvider = Provider<OfflineCache>(
  (ref) => CacheBackedOfflineCache(ref.watch(cacheStoreProvider)),
  name: 'offlineCache',
);

/// Trả về **ngay** (không `await`) để state đầu tiên là `AsyncData` chứ không
/// phải `AsyncLoading`.
///
/// Quan trọng: nếu `build()` là `async`, ngay khung hình đầu tiên state là
/// `loading` -> nút bấm hiện spinner trước khi người dùng kịp làm gì. Dùng
/// [SynchronousFuture] giữ `isLoading == false` lúc mở màn.
///
/// Chỉ dùng cho giá trị **đã có sẵn**; muốn khởi tạo bằng dữ liệu bất đồng bộ
/// thì khai báo `build()` là `async` và chấp nhận 1 vòng `loading` ban đầu.
Future<T> _ready<T>(T value) => SynchronousFuture<T>(value);

/// Ràng buộc mật khẩu tối thiểu (Day 11).
///
/// 6 ký tự là mức tối thiểu của server; UI và controller dùng **chung** hằng số
/// này để không bao giờ lệch nhau.
const int kMinPasswordLength = 6;

/// Lý do form đăng nhập/đăng ký không hợp lệ.
///
/// Tách khỏi chuỗi thông điệp để **đa ngôn ngữ**: tầng `core` trả mã, màn hình
/// dịch mã sang ngôn ngữ đang chạy (Day 13).
enum CredentialIssue {
  usernameRequired,
  passwordRequired,
  passwordTooShort,
  emailRequired,
  emailInvalid;

  /// Bản tiếng Việt — dùng khi không có `BuildContext` (test, code nền tảng).
  String get vietnamese => switch (this) {
        CredentialIssue.usernameRequired => 'Vui lòng nhập tên đăng nhập.',
        CredentialIssue.passwordRequired => 'Vui lòng nhập mật khẩu.',
        CredentialIssue.passwordTooShort =>
          'Mật khẩu phải có ít nhất $kMinPasswordLength ký tự.',
        CredentialIssue.emailRequired => 'Vui lòng nhập email.',
        CredentialIssue.emailInvalid => 'Email không hợp lệ.',
      };
}

/// Kiểm tra dữ liệu form trước khi gọi mạng.
///
/// Trả về thông điệp tiếng Việt đầu tiên bị sai, hoặc `null` nếu hợp lệ.
/// Không gửi request thất bại chỉ để server trả lại lỗi tương tự — tiết kiệm
/// mạng và tránh đưa mật khẩu đi vô ích.
///
/// Tầng `core` không import Flutter nên không dựng được `AppLocalizations`; vì
/// vậy hàm này trả về **mã lỗi** ([CredentialIssue]). Màn hình dịch mã đó sang
/// ngôn ngữ đang chạy. [validateCredentials] là bản trả về chuỗi tiếng Việt,
/// dùng cho test và cho code không có `BuildContext`.
CredentialIssue? checkCredentials({
  required String username,
  required String password,
  String? email,
  bool requireEmail = false,
}) {
  if (username.trim().isEmpty) return CredentialIssue.usernameRequired;
  if (password.isEmpty) return CredentialIssue.passwordRequired;
  if (password.length < kMinPasswordLength) {
    return CredentialIssue.passwordTooShort;
  }
  if (requireEmail) {
    final mail = (email ?? '').trim();
    if (mail.isEmpty) return CredentialIssue.emailRequired;
    // Cùng regex với `User.hasValidEmail` (Day 6) để không lệch quy tắc.
    if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(mail)) {
      return CredentialIssue.emailInvalid;
    }
  }
  return null;
}

/// Dịch mã lỗi sang ngôn ngữ đang chạy — dùng ở màn hình (Day 13).
extension CredentialIssueMessage on CredentialIssue {
  String localize(AppLocalizations l10n) => switch (this) {
        CredentialIssue.usernameRequired => l10n.validationUsernameRequired,
        CredentialIssue.passwordRequired => l10n.validationPasswordRequired,
        CredentialIssue.passwordTooShort =>
          l10n.validationPasswordTooShort(kMinPasswordLength),
        CredentialIssue.emailRequired => l10n.validationEmailRequired,
        CredentialIssue.emailInvalid => l10n.validationEmailInvalid,
      };
}

/// Kiểm tra form, trả **chuỗi tiếng Việt** (kèm quy tắc dùng trong test).
///
/// Dùng khi không có `BuildContext`; màn hình nên dùng [checkCredentials] rồi
/// [CredentialIssueMessage.localize] để hỗ trợ đa ngôn ngữ.
String? validateCredentials({
  required String username,
  required String password,
  String? email,
  bool requireEmail = false,
}) {
  final issue = checkCredentials(
    username: username,
    password: password,
    email: email,
    requireEmail: requireEmail,
  );
  return issue?.vietnamese;
}

/// Phiên đăng nhập hiện tại; `null` = chưa đăng nhập.
///
/// Dùng [AsyncNotifier] thay vì `StateNotifier` vì lỗi mạng đã được đóng gói
/// sẵn trong `AsyncValue.error` — UI đọc `state.error` là ra thông điệp tiếng
/// Việt của `AppException`, không cần `try/catch` ở màn hình.
class AuthController extends AsyncNotifier<AuthSession?> {
  /// State đầu tiên **luôn** là `AsyncData(null)` — dùng [SynchronousFuture]
  /// để không có khung hình `loading` nào lúc mở app (xem [_ready]).
  ///
  /// Việc khôi phục token chạy ở [restoreSession], do màn Login gọi sau khi dựng
  /// xong. Lý do: nếu `build()` await kho bí mật thì nút "Đăng nhập" sẽ hiện
  /// spinner và bị khoá trong lúc đọc đĩa, trông như app bị treo.
  @override
  Future<AuthSession?> build() => _ready(null);

  /// Khôi phục phiên đã lưu (Day 11). Gọi 1 lần khi app mở.
  ///
  /// Chỉ khôi phục khi kho **còn** token; không có token thì giữ `null` và màn
  /// Login hiện ra như cũ. Không gọi `/me` vì server Flask cũ chưa có endpoint
  /// đó — token hết hạn sẽ bị phát hiện ở request kế tiếp và xử lý qua 401.
  Future<void> restoreSession() async {
    final token = await ref.read(authRepositoryProvider).currentToken();
    if (token == null || token.isEmpty) return;
    state =
        AsyncData(AuthSession(user: const User(username: ''), token: token));

    // Lấy tên người dùng (và xác nhận phiên còn hợp lệ: token hết hạn thì tầng mạng tự
    // làm mới, không được thì đăng xuất). Sau `await` phiên có thể đã bị xoá -> kiểm tra lại.
    final user = await _repo.currentUser();
    final latestToken = await _repo.currentToken();
    if (user != null && latestToken != null && latestToken.isNotEmpty) {
      if (state.valueOrNull?.isAuthenticated ?? false) {
        await _syncUser(user);
        state = AsyncData(AuthSession(user: user, token: latestToken));
      }
    }
  }

  AuthRepository get _repo => ref.read(authRepositoryProvider);

  /// Đăng nhập. Trả về `true` khi thành công để màn hình điều hướng —
  /// `false` khi thất bại (lỗi đã nằm trong [state]).
  ///
  /// [formError] là thông báo validate phía UI (rỗi/quá ngắn/email sai): có
  /// thì **không gọi mạng**, lỗi nằm ngay trong [state].
  Future<bool> login(
    String username,
    String password, {
    String? formError,
  }) async {
    final invalid = formError;
    if (invalid != null) {
      state = AsyncError(
        AppException(kind: AppExceptionKind.validation, message: invalid),
        StackTrace.current,
      );
      return false;
    }
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      final session = await _repo.login(username, password);
      await _syncUser(session.user);
      return session;
    });
    return !state.hasError;
  }

  /// Đăng ký. `true` khi server trả phiên hợp lệ.
  Future<bool> register(
    String username,
    String password,
    String email, {
    String? formError,
  }) async {
    final invalid = formError;
    if (invalid != null) {
      state = AsyncError(
        AppException(kind: AppExceptionKind.validation, message: invalid),
        StackTrace.current,
      );
      return false;
    }
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      final session = await _repo.register(username, password, email);
      await _syncUser(session.user);
      return session;
    });
    return !state.hasError;
  }

  Future<void> logout() async {
    await _repo.logout();
    await _wipeLocalCache();
    await _revokeCloudAi();
    state = const AsyncData(null);
  }

  /// Xoá tài khoản và **toàn bộ** dữ liệu: server (lịch sử, phiên, hội thoại),
  /// cache ngoại tuyến, hồ sơ sức khoẻ + lịch nhắc trên máy (quyền được quên).
  ///
  /// Ném [AppException] khi sai mật khẩu — khi đó **không** xoá gì, kể cả cục bộ.
  Future<void> deleteAccount(String password) async {
    if (password.isEmpty) {
      throw const AppException(
        kind: AppExceptionKind.validation,
        message: 'Vui lòng nhập mật khẩu.',
      );
    }
    await _repo.deleteAccount(password);
    await _wipeLocalCache();
    await _revokeCloudAi();
    await ref.read(appLockStoreProvider).clearPin();
    await ref.read(lastUserStoreProvider).clear();
    await _wipeVault();
    try {
      await ref.read(healthDatabaseProvider).deleteAll();
      await ref.read(notificationServiceProvider).cancelAll();
    } on Object {
      // Dữ liệu trên server đã xoá; lỗi dọn cục bộ không được che kết quả đó.
    }
    ref.invalidate(healthProfileControllerProvider);
    state = const AsyncData(null);
  }

  /// Được gọi khi `ApiClient` gặp 401: xoá token và đưa state về chưa đăng
  /// nhập để UI điều hướng về màn Login.
  Future<void> handleUnauthorized() async {
    await _repo.logout();
    await _wipeLocalCache();
    state = const AsyncData(null);
  }

  /// Thiết bị dùng chung: nếu người đăng nhập KHÁC người đã nhớ thì xoá dữ liệu
  /// sức khoẻ cục bộ của người trước (DB, nhắc, đơn, tuân thủ, cache lối sống,
  /// đồng ý AI đám mây) rồi nhớ id mới. Cùng người -> giữ nguyên mọi thứ.
  /// Chưa từng nhớ id (bản cũ) -> chỉ ghi nhớ, không xoá.
  Future<void> _syncUser(User user) async {
    final id = user.id?.toString() ??
        (user.username.isNotEmpty ? user.username : null);
    if (id == null) return;
    final store = ref.read(lastUserStoreProvider);
    final previous = await store.read();
    if (previous != null && previous != id) {
      await _wipeLocalCache();
      await _revokeCloudAi();
      await _wipeVault();
      try {
        await ref.read(healthDatabaseProvider).deleteAll();
        await ref.read(notificationServiceProvider).cancelAll();
      } on Object {
        // Không chặn đăng nhập; không log nội dung.
      }
      ref.invalidate(healthProfileControllerProvider);
      ref.invalidate(reminderControllerProvider);
      ref.invalidate(prescriptionControllerProvider);
      ref.invalidate(adherenceControllerProvider);
      ref.invalidate(lifestyleProvider);
    }
    if (previous != id) await store.write(id);
  }

  /// Xoá két mã hoá (hồ sơ sức khoẻ + nhật ký triệu chứng) và tắt trí nhớ trợ lý.
  Future<void> _wipeVault() async {
    try {
      await ref.read(healthVaultProvider).wipe();
    } on Object {
      // Không chặn đăng xuất/xoá tài khoản; không log nội dung.
    }
    try {
      await ref.read(assistantMemoryStoreProvider).write(false);
    } on Object {
      // Bỏ qua.
    }
    ref.invalidate(symptomEpisodesProvider);
    ref.invalidate(assistantMemoryControllerProvider);
  }

  /// Thu hồi đồng ý AI đám mây khi rời tài khoản (đồng ý gắn với người dùng cũ).
  Future<void> _revokeCloudAi() async {
    try {
      await ref.read(cloudAiConsentStoreProvider).revoke();
      ref.invalidate(cloudAiConsentControllerProvider);
    } on Object {
      // Bỏ qua: đăng xuất phải luôn hoàn tất.
    }
  }

  /// Xoá cache ngoại tuyến (lịch sử + kết quả tìm kiếm) của người dùng vừa rời đi.
  ///
  /// ⚠️ Không có bước này thì người đăng nhập kế tiếp trên cùng máy **đọc được lịch sử
  /// của người trước** khi mất mạng — cache không gắn với tài khoản nào. Lỗi xoá cache
  /// không được chặn việc đăng xuất.
  Future<void> _wipeLocalCache() async {
    try {
      await ref.read(cacheStoreProvider).clear();
    } on Object {
      // Bỏ qua: đăng xuất phải luôn hoàn tất.
    }
  }
}

/// Nơi lưu việc đồng ý chính sách dữ liệu. Test override bằng bản trong RAM.
final consentStoreProvider = Provider<ConsentStore>(
  (ref) => PreferencesConsentStore(),
  name: 'consentStore',
);

/// Trạng thái đồng ý chính sách dữ liệu (`docs/PRIVACY.md`).
///
/// `build()` đọc kho nên khung hình đầu là `loading`; `ConsentGate` hiện spinner
/// trong lúc đó chứ không nháy màn đồng ý.
class ConsentController extends AsyncNotifier<bool> {
  @override
  Future<bool> build() => ref.read(consentStoreProvider).hasAccepted();

  Future<void> accept() async {
    await ref.read(consentStoreProvider).accept();
    state = const AsyncData(true);
  }

  Future<void> revoke() async {
    await ref.read(consentStoreProvider).revoke();
    state = const AsyncData(false);
  }
}

// --- Đồng ý AI đám mây (tách riêng khỏi consent hồ sơ) ----------------------

final cloudAiConsentStoreProvider = Provider<CloudAiConsentStore>(
  (ref) => PreferencesCloudAiConsentStore(),
  name: 'cloudAiConsentStore',
);

extension CloudAiConsentX on CloudAiConsent? {
  /// `true` nếu đồng ý còn hiệu lực.
  bool get granted => this != null;
}

class CloudAiConsentController extends AsyncNotifier<CloudAiConsent?> {
  @override
  Future<CloudAiConsent?> build() =>
      ref.read(cloudAiConsentStoreProvider).read();

  Future<void> grant() async {
    final store = ref.read(cloudAiConsentStoreProvider);
    await store.grant();
    state = AsyncData(await store.read());
  }

  Future<void> revoke() async {
    await ref.read(cloudAiConsentStoreProvider).revoke();
    state = const AsyncData(null);
  }
}

final cloudAiConsentControllerProvider =
    AsyncNotifierProvider<CloudAiConsentController, CloudAiConsent?>(
  CloudAiConsentController.new,
);

final consentControllerProvider =
    AsyncNotifierProvider<ConsentController, bool>(ConsentController.new);

final authControllerProvider =
    AsyncNotifierProvider<AuthController, AuthSession?>(AuthController.new);

/// Cổng cache **tối giản** cho dữ liệu không nhạy cảm (Day 12).
///
/// Chỉ gồm 2 việc mà UI thật sự cần:
/// 1. `load()` — lấy dữ liệu đã lưu để hiển thị khi **mất mạng**;
/// 2. `cachedAt()` — mốc thời gian để dựng banner "cập nhật lúc ...".
///
/// ⚠️ **Không** dùng port này cho token/mật khẩu/email: token đi qua
/// `AuthRepository` + kho bí mật của hệ điều hành (Day 11), không phải ở đây.
abstract interface class OfflineCache {
  /// Dữ liệu cache theo [key], `null` nếu chưa có hoặc cache hỏng.
  Future<Map<String, dynamic>?> load(String key);

  /// Lưu [data] kèm mốc thời gian hiện tại.
  Future<void> save(String key, Map<String, dynamic> data);

  /// Khi nào [key] được ghi lần cuối; `null` nếu chưa có.
  Future<DateTime?> cachedAt(String key);

  /// Xoá một mục (vd: sau khi đăng xuất).
  Future<void> invalidate(String key);
}

/// Adapter mặc định: bọc [CacheStore] của tầng `data`.
///
/// Giữ port này ở `domain` nghĩa là UI chỉ biết "có cache/không", không biết
/// cache nằm trên đĩa hay trong RAM — cùng tinh thần với các port của Day 7.
class CacheBackedOfflineCache implements OfflineCache {
  CacheBackedOfflineCache(this._store);

  final CacheStore _store;

  @override
  Future<Map<String, dynamic>?> load(String key) async {
    final value = await _store.read(key);
    return value is Map<String, dynamic> ? value : null;
  }

  @override
  Future<void> save(String key, Map<String, dynamic> data) =>
      _store.write(key, data);

  @override
  Future<DateTime?> cachedAt(String key) async {
    if (_store case final PreferencesCacheStore prefs) {
      return prefs.cachedAt(key);
    }
    if (_store case final InMemoryCacheStore memory) {
      return memory.cachedAtOf(key);
    }
    return null;
  }

  @override
  Future<void> invalidate(String key) => _store.remove(key);
}

/// Kết quả tìm thuốc kèm cờ "cũ" cho banner ngoại tuyến (Day 12).
class SearchSnapshot {
  const SearchSnapshot(
      {required this.response, required this.isStale, this.cachedAt});

  final MedicineSearchResponse response;

  /// `true` = lấy từ cache vì không gọi được server.
  final bool isStale;

  final DateTime? cachedAt;

  List<Medicine> get results => response.results;
}

/// Kết quả tìm thuốc. `null` = chưa tìm (không phải "không có kết quả").
class SearchController extends AsyncNotifier<SearchSnapshot?> {
  @override
  Future<SearchSnapshot?> build() => _ready(null);

  OfflineCache get _cache => ref.read(offlineCacheProvider);

  Future<bool> search(String name) async {
    final trimmed = name.trim();
    if (trimmed.isEmpty) return false; // UI tự hiện cảnh báo, không gọi mạng.

    final cacheKey = CacheKeys.search(trimmed);
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      try {
        final response =
            await ref.read(medicineRepositoryProvider).search(trimmed);
        unawaited(_saveCache(cacheKey, response));
        return SearchSnapshot(response: response, isStale: false);
      } on AppException {
        // Mất mạng -> tìm trong cache cùng từ khoá.
        final cached = await _cache.load(cacheKey);
        if (cached == null) rethrow;
        return SearchSnapshot(
          response: MedicineSearchResponse.fromJson(cached),
          isStale: true,
          cachedAt: await _cache.cachedAt(cacheKey),
        );
      }
    });
    return !state.hasError;
  }

  Future<void> _saveCache(String key, MedicineSearchResponse response) async {
    try {
      await _cache.save(key, <String, dynamic>{
        MedicineSearchResponse.keyResults:
            response.results.map((m) => m.toJson()).toList(),
        MedicineSearchResponse.keyQuery: response.query,
      });
    } on Object {
      // Cache là tiện ích: lỗi ghi không được làm hỏng luồng chính.
    }
  }

  /// Xoá cảnh báo "Vui lòng nhập tên thuốc!" mà không phá kết quả đang hiện.
  void clearError() {
    if (state.hasError) state = const AsyncData(null);
  }
}

final searchControllerProvider =
    AsyncNotifierProvider<SearchController, SearchSnapshot?>(
  SearchController.new,
);

/// Lịch sử kèm trạng thái "cũ" để UI dựng banner ngoại tuyến (Day 12).
class HistorySnapshot {
  const HistorySnapshot({
    required this.items,
    required this.isStale,
    this.cachedAt,
  });

  final List<HistoryItem> items;

  /// `true` = dữ liệu lấy từ cache vì không gọi được server.
  final bool isStale;

  /// Lúc dữ liệu cache được ghi; `null` khi [isStale] = `false`.
  final DateTime? cachedAt;

  bool get isEmpty => items.isEmpty;

  HistorySnapshot copyWithItems(List<HistoryItem> next) => HistorySnapshot(
        items: next,
        isStale: isStale,
        cachedAt: cachedAt,
      );
}

/// Lịch sử quét ảnh; `null` = chưa tải.
class HistoryController extends AsyncNotifier<HistorySnapshot?> {
  @override
  Future<HistorySnapshot?> build() => _ready(null);

  OfflineCache get _cache => ref.read(offlineCacheProvider);

  /// Gọi server; nếu thất bại thì lấy cache để **không trắng màn hình**.
  Future<void> load() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() async {
      try {
        final items = await ref.read(medicineRepositoryProvider).history();
        // Có mạng -> lưu cache cho lần offline sau. Cache **không** chặn
        // UI: ghi lỗi thì lần offline chỉ mất cache, không mất dữ liệu.
        unawaited(_saveCache(items));
        return HistorySnapshot(items: items, isStale: false);
      } on AppException {
        // Mất mạng -> hiện dữ liệu cũ kèm cảnh báo thay vì màn trắng.
        final items = await _readCache();
        if (items.isEmpty) rethrow;
        return HistorySnapshot(
          items: items,
          isStale: true,
          cachedAt: await _cache.cachedAt(CacheKeys.history),
        );
      }
    });
  }

  /// Cache dùng **đúng shape của server** (`{"history": [...]}`) nên đọc lại
  /// bằng chính `HistoryResponse.fromJson` — không tự chế lại format.
  Future<void> _saveCache(List<HistoryItem> items) async {
    try {
      await _cache.save(
        CacheKeys.history,
        <String, dynamic>{
          HistoryResponse.keyHistory:
              items.map((item) => item.toJson()).toList(),
        },
      );
    } on Object {
      // Bỏ qua: cache là tiện ích, lỗi ghi không được làm hỏng mạch chính.
    }
  }

  Future<List<HistoryItem>> _readCache() async {
    try {
      final raw = await _cache.load(CacheKeys.history);
      if (raw == null) return const <HistoryItem>[];
      return HistoryResponse.fromJson(raw).items;
    } on Object {
      return const <HistoryItem>[];
    }
  }

  /// Xoá một bản ghi rồi cập nhật danh sách ngay trên máy.
  Future<bool> delete(int id) async {
    final previous = state.valueOrNull;
    state = await AsyncValue.guard(() async {
      await ref.read(medicineRepositoryProvider).deleteHistory(id);
      final kept = (previous?.items ?? const <HistoryItem>[])
          .where((item) => item.id != id)
          .toList(growable: false);
      // Xoá thành công -> cập nhật cache để lần offline khỏi hiện bản ghi đã xoá.
      unawaited(_saveCache(kept));
      return previous?.copyWithItems(kept) ??
          HistorySnapshot(items: kept, isStale: false);
    });
    if (state.hasError) {
      // Không xoá được -> trả danh sách cũ về UI để không mất dữ liệu đang xem.
      state = AsyncData(previous);
    }
    return !state.hasError;
  }
}

final historyControllerProvider =
    AsyncNotifierProvider<HistoryController, HistorySnapshot?>(
  HistoryController.new,
);

/// Kết quả nhận diện từ ảnh; `null` = chưa phân tích.
class AnalysisController extends AsyncNotifier<AnalysisResult?> {
  @override
  Future<AnalysisResult?> build() => _ready(null);

  Future<bool> analyze(String imagePath) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(
      () => ref.read(analyzerProvider).analyze(imagePath),
    );
    return !state.hasError;
  }

  /// Xoá kết quả cũ khi người dùng chọn ảnh khác.
  void reset() => state = const AsyncData(null);
}

final analysisControllerProvider =
    AsyncNotifierProvider<AnalysisController, AnalysisResult?>(
  AnalysisController.new,
);

// --- Agent chat (Day 25) ---------------------------------------------------

final agentClientProvider = Provider<AgentClient>(
  (ref) => SseAgentClient(ref.watch(apiClientProvider)),
  name: 'agentClient',
);

final chatControllerProvider =
    AutoDisposeAsyncNotifierProvider<ChatController, List<ChatMessage>>(
  ChatController.new,
);

// --- Health profile & reminders (Day 27) ------------------------------------

final healthDatabaseProvider = Provider<HealthDatabase>(
  (ref) => HealthDatabase.instance,
  name: 'healthDatabase',
);

final notificationServiceProvider = Provider<NotificationService>(
  (ref) => NotificationService.instance,
  name: 'notificationService',
);

/// Két mã hoá trên máy cho hồ sơ sức khoẻ + nhật ký triệu chứng. Test override
/// bằng `InMemoryHealthVault`.
final healthVaultProvider = Provider<HealthVault>(
  (ref) => SecureHealthVault(),
  name: 'healthVault',
);

/// Hồ sơ sức khoẻ nằm trong [HealthVault] (mã hoá). SQLite chỉ còn bảng cũ để
/// **di chuyển một lần**; nhắc/tuân thủ/đơn thuốc vẫn ở SQLite chưa mã hoá
/// (khuyến nghị SQLCipher sau này, cần chủ dự án duyệt).
///
/// Kho không đọc được -> state là lỗi [VaultUnreadable] (fail-closed, không ghi đè).
class HealthProfileController extends AsyncNotifier<HealthProfile?> {
  @override
  Future<HealthProfile?> build() => _ready(null);

  HealthDatabase get _db => ref.read(healthDatabaseProvider);
  HealthVault get _vault => ref.read(healthVaultProvider);

  Future<HealthProfile?> load() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_loadAndMigrate);
    return state.valueOrNull;
  }

  Future<HealthProfile?> _loadAndMigrate() async {
    final fromVault = await _vault.readProfile();
    HealthProfile? legacy;
    try {
      legacy = await _db.getProfile();
    } on Object {
      legacy =
          null; // không có/không đọc được SQLite cũ -> không còn gì để chuyển
    }
    if (fromVault != null) {
      // Lần chuyển trước dừng giữa chừng: két là nguồn đúng, xoá bản cũ rõ ràng.
      if (legacy != null) await _deleteLegacy();
      return fromVault;
    }
    if (legacy == null) return null;
    await _vault.writeProfile(legacy);
    final back = await _vault.readProfile();
    if (back == null) {
      return legacy; // chưa xác minh được -> giữ bản cũ, thử lại sau
    }
    await _deleteLegacy();
    return back;
  }

  Future<void> _deleteLegacy() async {
    try {
      await _db.deleteProfile();
    } on Object {
      // Thử lại ở lần tải sau.
    }
  }

  Future<HealthProfile> _persist(HealthProfile p) async {
    await _vault.writeProfile(p);
    return p;
  }

  Future<void> giveConsent() async {
    final existing = state.valueOrNull ?? const HealthProfile();
    final updated = existing.copyWith(consentGivenAt: DateTime.now());
    state = await AsyncValue.guard(() => _persist(updated));
  }

  Future<void> save(HealthProfile profile) async {
    final withConsent = profile.copyWith(
      consentGivenAt:
          profile.consentGivenAt ?? state.valueOrNull?.consentGivenAt,
    );
    state = await AsyncValue.guard(() => _persist(withConsent));
  }

  /// Xoá hồ sơ + nhật ký triệu chứng (mất consent thì trí nhớ cũng phải mất).
  Future<void> deleteProfile() async {
    await _vault.wipe();
    try {
      await _db.deleteProfile();
    } on Object {
      // Bản cũ có thể không còn.
    }
    await ref.read(assistantMemoryStoreProvider).write(false);
    ref.invalidate(symptomEpisodesProvider);
    ref.invalidate(assistantMemoryControllerProvider);
    await ref.read(notificationServiceProvider).cancelAll();
    state = const AsyncData(null);
  }
}

final healthProfileControllerProvider =
    AsyncNotifierProvider<HealthProfileController, HealthProfile?>(
  HealthProfileController.new,
);

// --- Trí nhớ triệu chứng của trợ lý (mặc định TẮT, chỉ trên máy) -------------

final assistantMemoryStoreProvider = Provider<AssistantMemoryStore>(
  (ref) => PreferencesAssistantMemoryStore(),
  name: 'assistantMemoryStore',
);

/// Công tắc trí nhớ. Chỉ bật được khi hồ sơ sức khoẻ đã có consent.
class AssistantMemoryController extends AsyncNotifier<bool> {
  @override
  Future<bool> build() => ref.read(assistantMemoryStoreProvider).read();

  /// Trả `true` nếu trạng thái mới được áp dụng; bật khi chưa consent -> `false`.
  Future<bool> setEnabled(bool on) async {
    if (on) {
      final consent =
          ref.read(healthProfileControllerProvider).valueOrNull?.hasConsent ??
              false;
      if (!consent) return false;
    }
    await ref.read(assistantMemoryStoreProvider).write(on);
    state = AsyncData(on);
    return true;
  }
}

final assistantMemoryControllerProvider =
    AsyncNotifierProvider<AssistantMemoryController, bool>(
  AssistantMemoryController.new,
);

/// Nhật ký triệu chứng trong két (mới nhất trước). Lỗi [VaultUnreadable] được
/// giữ nguyên trong state để UI báo, không bao giờ coi là danh sách rỗng.
class SymptomEpisodesController extends AsyncNotifier<List<SymptomEpisode>> {
  @override
  Future<List<SymptomEpisode>> build() =>
      ref.read(healthVaultProvider).readEpisodes();

  Future<void> reload() async {
    state = await AsyncValue.guard(
        () => ref.read(healthVaultProvider).readEpisodes());
  }

  Future<void> add(SymptomEpisode e) async {
    final vault = ref.read(healthVaultProvider);
    await vault.appendEpisode(e);
    await reload();
  }

  Future<void> remove(DateTime at) async {
    final vault = ref.read(healthVaultProvider);
    await vault.deleteEpisode(at);
    await reload();
  }

  Future<void> clear() async {
    await ref.read(healthVaultProvider).clearEpisodes();
    await reload();
  }
}

final symptomEpisodesProvider =
    AsyncNotifierProvider<SymptomEpisodesController, List<SymptomEpisode>>(
  SymptomEpisodesController.new,
);

/// Trí nhớ đang HOẠT ĐỘNG = công tắc bật VÀ hồ sơ có consent.
bool isAssistantMemoryActive(Ref ref) {
  final on = ref.read(assistantMemoryControllerProvider).valueOrNull ?? false;
  final consent =
      ref.read(healthProfileControllerProvider).valueOrNull?.hasConsent ??
          false;
  return on && consent;
}

class ReminderController extends AsyncNotifier<List<MedicationReminder>> {
  @override
  Future<List<MedicationReminder>> build() => _ready(const []);

  HealthDatabase get _db => ref.read(healthDatabaseProvider);
  NotificationService get _notifications =>
      ref.read(notificationServiceProvider);

  Future<void> load() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() => _db.getReminders());
  }

  Future<void> add(MedicationReminder reminder) async {
    state = await AsyncValue.guard(() async {
      final saved = await _db.insertReminder(reminder);
      // Android 13+: không xin quyền lúc chạy thì thông báo bị chặn âm thầm (kênh
      // `importance=NONE`) dù báo thức đã đặt — phát hiện khi chạy trên máy ảo.
      await _notifications.requestPermission();
      await _notifications.scheduleReminder(saved);
      return _db.getReminders();
    });
  }

  Future<void> toggleActive(MedicationReminder reminder, bool active) async {
    final updated = reminder.copyWith(isActive: active);
    await _db.updateReminder(updated);
    if (active) {
      await _notifications.scheduleReminder(updated);
    } else {
      await _notifications.cancelReminder(reminder.id!);
    }
    state = await AsyncValue.guard(() => _db.getReminders());
  }

  /// Trừ `pillsPerDose` khỏi `pillsLeft` (nếu đang theo dõi) sau khi uống.
  Future<void> consumePills(MedicationReminder reminder) async {
    final id = reminder.id;
    if (reminder.pillsLeft == null || id == null) return;
    // Chống gọi đôi đồng thời cho cùng một nhắc.
    if (!_consuming.add(id)) return;
    try {
      // Đọc lại số viên mới nhất từ DB (không tin bản [reminder] đã cũ).
      final current = (await _db.getReminders())
          .where((r) => r.id == id)
          .map((r) => r.pillsLeft)
          .firstOrNull;
      final left = current ?? reminder.pillsLeft!;
      final next = pillsAfterDose(left, reminder.pillsPerDose);
      await _db.updateReminder(reminder.copyWith(pillsLeft: () => next));
      state = await AsyncValue.guard(() => _db.getReminders());
    } finally {
      _consuming.remove(id);
    }
  }

  final Set<int> _consuming = {};

  Future<void> remove(int id) async {
    await _notifications.cancelReminder(id);
    await _db.deleteReminder(id);
    state = await AsyncValue.guard(() => _db.getReminders());
  }
}

final reminderControllerProvider =
    AsyncNotifierProvider<ReminderController, List<MedicationReminder>>(
  ReminderController.new,
);

// --- Đơn thuốc quét được → lịch nhắc (người dùng xác nhận trước khi lưu) ----

/// Lưu đơn thuốc đã xác nhận (chỉ dòng thuốc đã tách, không OCR thô/ảnh) và tạo
/// lịch nhắc tương ứng. Không ghi log tên thuốc/chẩn đoán.
class PrescriptionController extends AsyncNotifier<List<SavedPrescription>> {
  @override
  Future<List<SavedPrescription>> build() => _ready(const []);

  HealthDatabase get _db => ref.read(healthDatabaseProvider);

  Future<void> load() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() => _db.getPrescriptions());
  }

  /// Lưu đơn + tạo một [MedicationReminder] cho mỗi (thuốc, giờ).
  ///
  /// [mealLabel] đổi `meal_relation` thành nhãn hiển thị (đã dịch) để ghép vào
  /// `dosage` = "<liều> · <bữa ăn>". Tên thuốc và (tuỳ chọn) chẩn đoán chỉ được
  /// thêm vào hồ sơ sức khoẻ khi hồ sơ đã có consent.
  /// Trả về số lịch nhắc đã tạo.
  Future<int> saveWithReminders({
    required List<PrescriptionItemRecord> items,
    required String Function(String mealRelation) mealLabel,
    String? diagnosis,
    bool addDiagnosisToProfile = false,
  }) async {
    final valid = [
      for (final i in items)
        if (i.medicineName.trim().isNotEmpty && i.slots.isNotEmpty) i,
    ];
    if (valid.isEmpty) return 0;

    final profileCtrl = ref.read(healthProfileControllerProvider.notifier);
    final profile = ref.read(healthProfileControllerProvider).valueOrNull ??
        await profileCtrl.load();
    final consent = profile?.hasConsent ?? false;
    final keepDiagnosis = consent &&
        addDiagnosisToProfile &&
        (diagnosis?.trim().isNotEmpty ?? false);

    await _db.insertPrescription(SavedPrescription(
      createdAt: DateTime.now(),
      diagnosis: keepDiagnosis ? diagnosis!.trim() : null,
      items: valid,
    ));

    final reminders = ref.read(reminderControllerProvider.notifier);
    var count = 0;
    for (final item in valid) {
      final label = mealLabel(item.mealRelation);
      final dosage = [
        if (item.doseText.trim().isNotEmpty) item.doseText.trim(),
        if (item.mealRelation != 'none' && label.isNotEmpty) label,
      ].join(' · ');
      // Tổng số viên chia đều (làm tròn xuống) cho mỗi khung giờ của thuốc.
      final perSlot = item.pillsTotal == null
          ? null
          : item.pillsTotal! ~/ item.slots.length;
      for (final slot in item.slots) {
        final parts = slot.split(':');
        final hour = int.tryParse(parts.first) ?? 8;
        final minute = parts.length > 1 ? int.tryParse(parts[1]) ?? 0 : 0;
        await reminders.add(MedicationReminder(
          medicineName: item.medicineName.trim(),
          hour: hour.clamp(0, 23),
          minute: minute.clamp(0, 59),
          dosage: dosage.isEmpty ? null : dosage,
          pillsLeft: perSlot,
          pillsPerDose: item.pillsPerDose < 1 ? 1 : item.pillsPerDose,
        ));
        count++;
      }
    }

    if (consent && profile != null) {
      final meds = [...profile.currentMedications];
      for (final item in valid) {
        final name = item.medicineName.trim();
        if (!meds.any((m) => m.toLowerCase() == name.toLowerCase())) {
          meds.add(name);
        }
      }
      final conditions = [...profile.conditions];
      if (keepDiagnosis) {
        final d = diagnosis!.trim();
        if (!conditions.any((c) => c.toLowerCase() == d.toLowerCase())) {
          conditions.add(d);
        }
      }
      await profileCtrl.save(
        profile.copyWith(currentMedications: meds, conditions: conditions),
      );
    }

    state = await AsyncValue.guard(() => _db.getPrescriptions());
    return count;
  }
}

final prescriptionControllerProvider =
    AsyncNotifierProvider<PrescriptionController, List<SavedPrescription>>(
  PrescriptionController.new,
);

/// Lịch nhắc sắp hết thuốc: còn <= 3 liều (pillsLeft <= pillsPerDose * 3).
List<MedicationReminder> lowStockOf(Iterable<MedicationReminder> all) => [
      for (final r in all)
        if (r.pillsLeft != null && r.pillsLeft! <= r.pillsPerDose * 3) r,
    ];

final lowStockRemindersProvider = Provider<List<MedicationReminder>>(
  (ref) => lowStockOf(
    ref.watch(reminderControllerProvider).valueOrNull ?? const [],
  ),
  name: 'lowStockReminders',
);

// --- Thiết bị (vị trí/quay số/bản đồ) & cơ sở y tế --------------------------

final deviceChannelProvider = Provider<DeviceChannel>(
  (ref) => DeviceChannel(),
  name: 'deviceChannel',
);

/// Danh sách cơ sở y tế mẫu (asset, chưa kiểm chứng) — đọc từ máy, không gọi mạng.
final medicalFacilitiesProvider = FutureProvider<List<MedicalFacility>>(
  (ref) async => MedicalFacility.listFromJsonString(
    await rootBundle.loadString('assets/data/medical_facilities.json'),
  ),
  name: 'medicalFacilities',
);

/// Theo dõi tuân thủ uống thuốc (streak) — Day 33.
///
/// Dữ liệu lấy từ bảng `adherence_log` đã có sẵn trong [HealthDatabase] từ Day 27
/// nhưng chưa từng được ghi/đọc ở đâu trong app; provider này là chỗ nối thật đầu
/// tiên. Giữ cửa sổ 60 ngày đổ lại — đủ tính streak và vẽ 7 ngày gần nhất, không
/// cần tải toàn bộ lịch sử.
class AdherenceController extends AsyncNotifier<List<AdherenceRecord>> {
  static const _lookbackDays = 60;

  HealthDatabase get _db => ref.read(healthDatabaseProvider);

  Future<void> load() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_fetch);
  }

  @override
  Future<List<AdherenceRecord>> build() {
    // Nút "Đã uống"/"Hoãn" trên thông báo (khi app còn sống) đi qua đây.
    NotificationService.instance.onDoseAction = (id, action) async {
      final list = await _db.getReminders();
      final matches = list.where((r) => r.id == id);
      if (matches.isEmpty) return;
      if (action == NotificationService.actionTaken) {
        await logDose(matches.first);
      } else if (action == NotificationService.actionSnooze) {
        await snooze(matches.first);
      }
    };
    return _ready(const []);
  }

  /// Ghi nhận một lượt cho [reminder] (giờ thiết bị) với [status].
  ///
  /// `taken` quá giờ hẹn >60 phút hôm nay -> lưu `late`. Uống (taken/late) thì trừ
  /// `pillsLeft` đi `pillsPerDose` nếu đang theo dõi số viên.
  Future<void> logDose(MedicationReminder reminder,
      {String status = kDoseTaken}) async {
    final id = reminder.id;
    if (id == null) return;
    final now = DateTime.now();
    final effective =
        status == kDoseTaken ? doseStatusForTaken(reminder, now) : status;
    if (isTakenStatus(effective)) {
      // Chặn ghi trùng (bấm đôi / thông báo + nút): đã uống hôm nay thì bỏ qua.
      final logs = state.valueOrNull ?? await _fetch();
      if (takenIdsToday(logs, now).contains(id)) return;
    }
    await _db.logAdherence(
        AdherenceRecord(reminderId: id, takenAt: now, status: effective));
    if (isTakenStatus(effective)) {
      await ref
          .read(reminderControllerProvider.notifier)
          .consumePills(reminder);
    }
    state = await AsyncValue.guard(_fetch);
  }

  /// Hoãn [minutes] phút (mặc định 15, như nút trên thông báo): đặt thông báo
  /// một-lần và ghi `snoozed`.
  Future<void> snooze(MedicationReminder reminder, {int minutes = 15}) async {
    await ref
        .read(notificationServiceProvider)
        .snooze(reminder, minutes: minutes);
    await logDose(reminder, status: kDoseSnoozed);
  }

  Future<List<AdherenceRecord>> _fetch() => _db.getAdherenceSince(
        DateTime.now().subtract(const Duration(days: _lookbackDays)),
      );
}

final adherenceControllerProvider =
    AsyncNotifierProvider<AdherenceController, List<AdherenceRecord>>(
  AdherenceController.new,
);

bool isTakenStatus(String status) =>
    status == kDoseTaken || status == kDoseLate;

/// `taken` -> `late` nếu quá giờ hẹn hôm nay hơn 60 phút.
///
/// Chọn giờ hẹn **gần nhất** (theo khoảng cách tuyệt đối) trong hôm qua/hôm nay/
/// ngày mai: liều 23:30 ghi lúc 00:45 hôm sau -> gần hẹn hôm qua -> `late`;
/// liều 13:00 ghi lúc 08:00 -> gần hẹn hôm nay (chưa tới) -> `taken` (sớm).
String doseStatusForTaken(MedicationReminder r, DateTime now) {
  DateTime? nearest;
  for (final dayOffset in const [-1, 0, 1]) {
    final due =
        DateTime(now.year, now.month, now.day + dayOffset, r.hour, r.minute);
    if (nearest == null ||
        now.difference(due).abs() < now.difference(nearest).abs()) {
      nearest = due;
    }
  }
  return now.difference(nearest!) > const Duration(minutes: 60)
      ? kDoseLate
      : kDoseTaken;
}

/// Số viên còn lại sau một liều, không âm. Hàm thuần để test.
int pillsAfterDose(int left, int perDose) =>
    (left - perDose).clamp(0, left < 0 ? 0 : left).toInt();

bool _sameDay(DateTime d, DateTime t) =>
    d.year == t.year && d.month == t.month && d.day == t.day;

/// Id nhắc đã uống (taken/late) hôm nay. Hàm thuần để test.
Set<int> takenIdsToday(List<AdherenceRecord> logs, DateTime now) => {
      for (final log in logs)
        if (isTakenStatus(log.status) &&
            !log.skipped &&
            _sameDay(log.takenAt, now))
          log.reminderId,
    };

/// Id nhắc đã bấm "Bỏ qua" hôm nay (và chưa uống).
Set<int> skippedIdsToday(List<AdherenceRecord> logs, DateTime now) {
  final taken = takenIdsToday(logs, now);
  return {
    for (final log in logs)
      if (log.status == kDoseSkipped &&
          _sameDay(log.takenAt, now) &&
          !taken.contains(log.reminderId))
        log.reminderId,
  };
}

/// Streak: ngày liên tiếp có >=1 lượt taken/late; chưa uống hôm nay không phá streak.
int adherenceStreakOf(List<AdherenceRecord> logs, DateTime now) {
  final takenDates = <DateTime>{
    for (final log in logs)
      if (isTakenStatus(log.status) && !log.skipped)
        DateTime(log.takenAt.year, log.takenAt.month, log.takenAt.day),
  };
  var day = DateTime(now.year, now.month, now.day);
  if (!takenDates.contains(day)) {
    day = DateTime(day.year, day.month, day.day - 1);
  }
  var streak = 0;
  while (takenDates.contains(day)) {
    streak++;
    day = DateTime(day.year, day.month, day.day - 1);
  }
  return streak;
}

/// Id các nhắc thuốc **đã** uống hôm nay (taken hoặc late).
final takenTodayProvider = Provider<Set<int>>((ref) {
  final logs = ref.watch(adherenceControllerProvider).valueOrNull ?? const [];
  return takenIdsToday(logs, DateTime.now());
});

/// Id nhắc bị bỏ qua hôm nay.
final skippedTodayProvider = Provider<Set<int>>((ref) {
  final logs = ref.watch(adherenceControllerProvider).valueOrNull ?? const [];
  return skippedIdsToday(logs, DateTime.now());
});

final adherenceStreakProvider = Provider<int>((ref) {
  final logs = ref.watch(adherenceControllerProvider).valueOrNull ?? const [];
  return adherenceStreakOf(logs, DateTime.now());
});

// --- Khoá ứng dụng bằng mã PIN (Day 33) -------------------------------------

final lastUserStoreProvider = Provider<LastUserStore>(
  (ref) => SecureLastUserStore(),
  name: 'lastUserStore',
);

final appLockStoreProvider = Provider<AppLockStore>(
  (ref) => SecureAppLockStore(),
  name: 'appLockStore',
);
