import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_analyzer.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_repository.dart';
import 'package:image_processing_app/data/adapters/secure_token_store.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/data/models/user.dart';
import 'package:image_processing_app/domain/ports/auth_repository.dart';
import 'package:image_processing_app/domain/ports/medicine_analyzer.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';

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
final apiClientProvider = Provider<ApiClient>(
  (ref) {
    final client = ApiClient(
      tokenProvider: () => ref.read(tokenStoreProvider).read(),
      // 401 ở endpoint có bảo vệ -> xoá token cục bộ ngay, để lần gọi sau
      // không gửi "Bearer <token hết hạn>" nữa.
      onUnauthorized: ref.read(unauthorizedNotifierProvider),
    );
    ref.onDispose(client.close);
    return client;
  },
  name: 'apiClient',
);

final authRepositoryProvider = Provider<AuthRepository>(
  (ref) => RemoteAuthRepository(
    ref.watch(apiClientProvider),
    store: ref.watch(tokenStoreProvider),
  ),
  name: 'authRepository',
);

final medicineRepositoryProvider = Provider<MedicineRepository>(
  (ref) => RemoteMedicineRepository(ref.watch(apiClientProvider)),
  name: 'medicineRepository',
);

final analyzerProvider = Provider<MedicineAnalyzer>(
  (ref) => RemoteAnalyzer(ref.watch(apiClientProvider)),
  name: 'analyzer',
);

/// Nhãn tầng nhận diện đang chạy (`'remote-flask'`, Day 26 sẽ thêm nữa).
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
    state = await AsyncValue.guard(() => _repo.login(username, password));
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
    state = await AsyncValue.guard(
      () => _repo.register(username, password, email),
    );
    return !state.hasError;
  }

  Future<void> logout() async {
    await _repo.logout();
    state = const AsyncData(null);
  }

  /// Được gọi khi `ApiClient` gặp 401: xoá token và đưa state về chưa đăng
  /// nhập để UI điều hướng về màn Login.
  Future<void> handleUnauthorized() async {
    await _repo.logout();
    state = const AsyncData(null);
  }
}

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
