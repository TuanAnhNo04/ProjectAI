/// Kiểm tra **hợp đồng sống** giữa adapter Dart THẬT và `medassist-core` THẬT.
///
/// Nằm ngoài `test/` vì cần server đang chạy (CI không có). Bắt mọi lệch hợp đồng
/// mà fixture chép tay không bắt được: sai tên trường, sai mã HTTP, token lồng.
///
/// ```powershell
/// # cửa sổ 1 (khoá GIẢ, chỉ localhost, dữ liệu mẫu)
/// $env:MEDASSIST_SECRET_KEY = "fake-local-dev-key-0123456789-abcdefghijklmnop"
/// $env:MEDASSIST_DB = "$env:TEMP\medassist-live.db"
/// cd server\medassist-core ; python -m app.seed
/// python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
/// # cửa sổ 2
/// flutter test tools/medassist_core/core_live_check_test.dart `
///   --dart-define=API_BASE_URL=http://127.0.0.1:8000 --dart-define=BACKEND=core
/// ```
///
/// ⚠️ Các test dùng chung một server và có thứ tự (đăng ký → ... → xoá tài khoản).
/// Tên đăng nhập có hậu tố ngẫu nhiên để chạy lại nhiều lần không bị trùng.
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/config/api_paths.dart';
import 'package:image_processing_app/core/config/app_config.dart';
import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_catalog.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_repository.dart';
import 'package:image_processing_app/data/adapters/sse_agent_client.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';

const String _password = 'matkhau-gia-123';

void main() {
  final paths = ApiPaths.current;
  final stamp = DateTime.now().millisecondsSinceEpoch.toRadixString(36);
  final username = 'live_$stamp';

  final store = InMemoryTokenStore();
  late final RemoteAuthRepository auth;
  late final ApiClient api;
  api = ApiClient(
    tokenProvider: store.read,
    tokenRefresher: () => auth.refreshSession(),
  );
  auth = RemoteAuthRepository(
    api,
    store: store,
    loginPath: paths.login,
    registerPath: paths.register,
    logoutPath: paths.logout,
    refreshPath: paths.refresh,
    deleteAccountPath: paths.deleteAccount,
    mePath: paths.me,
  );
  final meds = RemoteMedicineRepository(
    api,
    searchPath: paths.search,
    historyPath: paths.history,
  );

  test('0. cấu hình: đang trỏ backend core', () {
    expect(AppConfig.backend, 'core');
    expect(paths.login, '/v1/auth/login');
  });

  test('1. đăng ký -> có access + refresh token', () async {
    final session = await auth.register(
      username,
      _password,
      '$username@example.invalid',
    );
    expect(session.isAuthenticated, isTrue);
    expect(session.refreshToken, isNotNull);
    expect(await store.readRefresh(), session.refreshToken);
  });

  test('1b. /me trả đúng tên người dùng (để hiện lời chào)', () async {
    final user = await auth.currentUser();
    expect(user?.username, username);
  });

  test('2. tìm thuốc: đọc đúng tên trường của core', () async {
    final result = await meds.search('panadol');
    expect(result.results, isNotEmpty);
    expect(result.results.first.name, isNotEmpty);
  });

  test('3. danh mục thuốc cho OCR: tải đủ, tên không rỗng', () async {
    final all =
        await RemoteMedicineCatalog(api, path: paths.catalog!).fetchAll();
    expect(all, isNotEmpty);
    expect(all.every((m) => m.name.isNotEmpty), isTrue);
  });

  test('4. lịch sử (cần token): rỗng với tài khoản mới', () async {
    final items = await meds.history();
    expect(items, isEmpty);
  });

  test('5. access token HẾT HẠN -> tự làm mới và gọi lại thành công', () async {
    final oldRefresh = await store.readRefresh();
    await store.write('token-het-han-gia'); // giả lập access token hết hạn

    final items = await meds.history(); // 401 -> refresh -> retry

    expect(items, isEmpty);
    expect(await store.read(), isNot('token-het-han-gia'));
    expect(await store.readRefresh(), isNot(oldRefresh)); // đã xoay
  });

  test('6. chat SSE: luồng sự kiện kết thúc bằng Done, không Failure',
      () async {
    final events = await SseAgentClient(api)
        .chat(sessionId: 'live-$stamp', message: 'Panadol là thuốc gì?')
        .toList();

    expect(
      events.whereType<Failure>(),
      isEmpty,
      reason: events
          .whereType<Failure>()
          .map((f) => '${f.error.message} | cause=${f.error.cause}')
          .join('\n'),
    );
    expect(events.last, isA<Done>());
  });

  test('7. xoá tài khoản sai mật khẩu -> báo lỗi, vẫn đăng nhập', () async {
    await expectLater(
      auth.deleteAccount('sai-mat-khau'),
      throwsA(isA<AppException>()),
    );
    expect(await store.read(), isNotNull);
    expect(await meds.history(), isEmpty); // phiên còn nguyên
  });

  test('8. xoá tài khoản đúng mật khẩu -> không đăng nhập lại được', () async {
    await auth.deleteAccount(_password);
    expect(await store.read(), isNull);

    await expectLater(
      auth.login(username, _password),
      throwsA(
        isA<AppException>()
            .having((e) => e.kind, 'kind', AppExceptionKind.unauthorized),
      ),
    );
  });
}
