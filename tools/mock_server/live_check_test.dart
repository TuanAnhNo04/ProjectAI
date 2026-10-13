/// Kiểm tra **hợp đồng sống** (live contract check) — không phải unit test.
///
/// Vì sao file này nằm ở `tools/mock_server/` chứ không phải `test/`:
/// nó **cần một server Flask đang chạy**, trong khi 200 test trong `test/` là
/// thuần Dart và phải chạy được offline (CI không có server). Đặt ngoài `test/`
/// nên `flutter test` (chạy cả thư mục `test/`) **không** đụng tới nó; muốn chạy
/// phải gọi đích danh đường dẫn.
///
/// Nó lấp đúng lỗ hổng mà fixture JSON không lấp được: `test/data/models/
/// fixtures/server_samples.dart` chỉ là JSON **chép tay**; còn ở đây chính
/// **adapter thật** (`RemoteAuthRepository`, `RemoteAnalyzer`,
/// `RemoteMedicineRepository`) nói chuyện với server thật. Nếu mock trả sai tên
/// trường, sai mã HTTP, hay sai kiểu dữ liệu, file này đỏ ngay — còn fixture thì
/// không bao giờ phát hiện được.
///
/// Cách chạy (2 cửa sổ):
/// ```powershell
/// # cửa sổ 1
/// python tools/mock_server/app.py
/// # cửa sổ 2 — BẮT BUỘC có --dart-define, AppConfig lấy URL lúc compile
/// flutter test tools/mock_server/live_check_test.dart `
///   --dart-define=API_BASE_URL=http://127.0.0.1:5000
/// ```
///
/// ⚠️ Các test trong đây **dùng chung một server** và có thứ tự (luồng 3 tạo
/// history rồi luồng 5 xoá nó). Đừng chạy song song và đừng đổi thứ tự.
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/config/app_config.dart';
import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_analyzer.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_repository.dart';

/// Ảnh PNG 1×1 hợp lệ. Mock không đọc nội dung ảnh, nhưng dùng ảnh **thật** để
/// nếu sau này server kiểm tra định dạng file thì test vẫn đúng ý nghĩa.
const String _png1x1Base64 =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8'
    'z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==';

void main() {
  // Dựng đúng kiểu wiring của app: token nằm ở một `TokenStore` duy nhất, vừa
  // cấp cho `ApiClient` (gắn header) vừa cho `RemoteAuthRepository` (ghi/xoá).
  final store = InMemoryTokenStore();
  final api = ApiClient(tokenProvider: store.read);
  final auth = RemoteAuthRepository(api, store: store);
  final meds = RemoteMedicineRepository(api);
  final analyzer = RemoteAnalyzer(api);

  final tempDir = Directory.systemTemp.createTempSync('medassist_live_');
  late File photo;
  late File emptyPhoto;

  /// Id do luồng 3 tạo ra, để luồng 5 xoá đúng nó.
  int? scannedHistoryId;

  setUpAll(() {
    // Tên file quyết định nhánh của mock (xem README): có "empty" -> không ra
    // thuốc nào. `basename` được `ApiClient.upload` gửi làm filename, nên tên
    // file cục bộ ở đây chính là thứ server nhìn thấy.
    photo = File('${tempDir.path}/scan_thuoc.png')
      ..writeAsBytesSync(base64Decode(_png1x1Base64));
    emptyPhoto = File('${tempDir.path}/empty_scan.png')
      ..writeAsBytesSync(base64Decode(_png1x1Base64));
  });

  tearDownAll(() {
    api.close();
    if (tempDir.existsSync()) tempDir.deleteSync(recursive: true);
  });

  test('server phải đang chạy ở đúng API_BASE_URL', () {
    // Thiếu `--dart-define` thì `AppConfig` rơi về mặc định `10.0.2.2:5000`
    // (địa chỉ emulator) — trên máy tính sẽ là "connection refused" khó hiểu.
    // Chặn trước bằng câu thông báo nói rõ phải làm gì.
    expect(
      AppConfig.baseUrl,
      contains('127.0.0.1:5000'),
      reason: 'Chạy lại kèm --dart-define=API_BASE_URL=http://127.0.0.1:5000 '
          '(hiện tại: ${AppConfig.baseUrl})',
    );
  });

  group('Luồng 4 — tìm kiếm thuốc', () {
    test('"panadol" trả về Panadol 500mg kèm số liệu review', () async {
      final found = await meds.search('panadol');

      expect(found.results, isNotEmpty);
      expect(found.results.first.name, 'Panadol 500mg');
      // CSV để cột review là **số** (93) chứ không phải "93%" -> asPercent
      // phải đọc được cả hai kiểu, đây là chỗ kiểm tra điều đó.
      expect(found.results.first.excellentReviewPercent, 93);
      expect(found.results.first.composition, 'Paracetamol 500mg');
      // Server echo lại query/total -> giữ được để UI hiện "1/1".
      expect(found.query, 'panadol');
      expect(found.total, found.results.length);
    });

    test('tên có dấu tiếng Việt được encode đúng (bug cũ nối chuỗi tay)',
        () async {
      // `?name=Thuốc ho Bảo Thanh` có dấu cách + ký tự có dấu: nối chuỗi thủ
      // công như code cũ sẽ hỏng URL, còn `AppConfig.uri` encode đúng.
      final found = await meds.search('Thuốc ho Bảo Thanh');
      expect(found.results.single.name, 'Thuốc ho Bảo Thanh');
    });

    test('gõ không dấu vẫn khớp tên có dấu', () async {
      final found = await meds.search('thuoc ho');
      expect(found.results.single.name, 'Thuốc ho Bảo Thanh');
    });

    test('từ khoá không tồn tại -> rỗng, không phải lỗi', () async {
      final found = await meds.search('khong co thuoc nay');
      expect(found.isEmpty, isTrue);
      expect(found.total, 0);
    });
  });

  group('Luồng 1 — đăng nhập sai', () {
    test('sai mật khẩu -> AppException(unauthorized), không crash', () async {
      await expectLater(
        auth.login('demo', 'sai-mat-khau'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.unauthorized)
              // Message tiếng Việt của server được giữ lại trong `detail`
              // (UI hiện message chung, log giữ bản gốc) — không mất thông tin.
              .having((e) => e.detail, 'detail', contains('Sai tài khoản')),
        ),
      );
    });

    test('server bản cũ trả HTTP 200 + {"success": false} vẫn là thất bại',
        () async {
      // Tài khoản "legacy" của mock mô phỏng đúng kiểu phản hồi đó; nếu adapter
      // chỉ nhìn mã HTTP thì test này sẽ "đăng nhập thành công" sai.
      await expectLater(
        auth.login('legacy', 'bat-ky'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.unauthorized),
        ),
      );
    });
  });

  group('Luồng 2 — đăng nhập đúng', () {
    test('đúng tài khoản -> có token và token được lưu vào TokenStore',
        () async {
      final session = await auth.login('demo', 'demo123');

      expect(session.isAuthenticated, isTrue);
      expect(session.user.username, 'demo');
      expect(session.token, isNotNull);
      // Token phải nằm trong store, nếu không thì các request sau (Day 11) sẽ
      // không có header Authorization.
      expect(await auth.currentToken(), session.token);
    });

    test('đăng ký mới -> không nhận token, và token phiên CŨ bị xoá', () async {
      // Đăng nhập trước để chắc chắn đang thật sự CÓ token của phiên cũ —
      // nếu không thì phép kiểm "đã bị xoá" dưới đây vô nghĩa.
      await auth.login('demo', 'demo123');
      expect(await auth.currentToken(), isNotNull);

      // Tên mỗi lần một khác để chạy lại file này không bị 409 trùng tên.
      final username = 'live_${DateTime.now().millisecondsSinceEpoch}';
      final session =
          await auth.register(username, 'abc12345', '$username@x.vn');

      expect(session.token, isNull);
      expect(session.isAuthenticated, isFalse);
      // Đăng ký là **đổi danh tính**: server cũ trả 201 mà không kèm token, và
      // adapter ghi `null` vào store. Đó là hành vi đúng — giữ lại token của
      // user cũ trong khi đã là user mới mới là lỗi bảo mật. Hệ quả có chủ đích:
      // đăng ký xong phải đăng nhập lại.
      expect(await auth.currentToken(), isNull);
    });
  });

  group('Luồng 3 — upload ảnh (multipart field "image")', () {
    test('ảnh có thuốc -> medicine_info được map vào AnalysisResult', () async {
      final result = await analyzer.analyze(photo.path);

      expect(result.isNotEmpty, isTrue);
      expect(result.medicines.first.name, 'Panadol 500mg');
      // Hai trường mới mà `AnalysisResult` đã đọc sẵn (Day 12 dùng cho cache).
      expect(result.imagePath, isNotNull);
      expect(result.analyzedAt, isNotNull);
      // Chuỗi UI hiển thị dựng từ field đã tách, không `split(': ')` như code cũ.
      expect(result.toDisplayLines().first, 'Tên: Panadol 500mg');

      // Nhớ id để luồng 5 xoá — lấy từ history chứ không đoán số.
      final items = await meds.history();
      scannedHistoryId = items
          .firstWhere((item) => item.filepath.contains('scan_thuoc.png'))
          .id;
      expect(scannedHistoryId, isNotNull);
    });

    test('ảnh không đọc ra thuốc nào -> kết quả rỗng, không phải lỗi',
        () async {
      final result = await analyzer.analyze(emptyPhoto.path);
      expect(result.isEmpty, isTrue);
    });

    test('chặn sớm: đường dẫn rỗng không tốn request', () async {
      await expectLater(
        analyzer.analyze('   '),
        throwsA(isA<AppException>()
            .having((e) => e.kind, 'kind', AppExceptionKind.validation)),
      );
    });
  });

  group('Luồng 5 — lịch sử quét', () {
    test('history có bản ghi vừa upload, mới nhất trước', () async {
      final items = await meds.history();

      expect(items, isNotEmpty);
      // Ảnh "empty" được upload sau nên phải đứng đầu.
      expect(items.first.filepath, contains('empty_scan.png'));
      expect(items.any((item) => item.id == scannedHistoryId), isTrue);
      // Mọi bản ghi trả về đều xoá được (có id) — UI không có nút xoá chết.
      expect(items.every((item) => item.isValid), isTrue);
    });

    test('xoá 1 mục -> biến mất; xoá lại -> notFound', () async {
      final before = (await meds.history()).length;

      await meds.deleteHistory(scannedHistoryId!);

      final after = await meds.history();
      expect(after.length, before - 1);
      expect(after.any((item) => item.id == scannedHistoryId), isFalse);

      // Server đã bỏ id đó -> adapter phải dịch 404 thành notFound.
      await expectLater(
        meds.deleteHistory(scannedHistoryId!),
        throwsA(isA<AppException>()
            .having((e) => e.kind, 'kind', AppExceptionKind.notFound)),
      );
    });
  });

  test('đăng xuất -> xoá token, gọi lại nhiều lần không ném', () async {
    // Tự đăng nhập lấy token thay vì dựa vào test trước: test đăng ký ở trên
    // đã xoá token (hành vi đúng), nên chỗ này phải tự lo cho mình.
    await auth.login('demo', 'demo123');
    expect(await auth.currentToken(), isNotNull);

    await auth.logout();
    expect(await auth.currentToken(), isNull);

    // `logout()` còn được gọi tự động khi gặp 401, lúc server có thể đang lỗi,
    // nên lần gọi thứ hai phải tuyệt đối không ném.
    await auth.logout();
    expect(await auth.currentToken(), isNull);
  });
}
