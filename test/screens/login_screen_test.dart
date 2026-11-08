// Day 3 — test đầu tiên cho màn hình Đăng nhập.
// Day 9 — màn hình đã chuyển sang Riverpod, nên test **override**
// `authRepositoryProvider` bằng fake: không chạm socket, không phụ thuộc
// server thật, và vẫn kiểm tra đúng nhánh xử lý lỗi.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/data/local/app_lock_store.dart';
import 'package:image_processing_app/data/local/last_user_store.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/data/models/user.dart';
import 'package:image_processing_app/domain/ports/auth_repository.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/app_shell.dart';
import 'package:image_processing_app/screens/auth_gate.dart';
import 'package:image_processing_app/screens/login_screen.dart';
import 'package:image_processing_app/screens/register_screen.dart';

/// Repository giả: luôn ném lỗi 401 trừ khi tên đăng nhập là `ok_user`.
class _FakeAuthRepository implements AuthRepository {
  _FakeAuthRepository(this.token, {this.restorable = true});

  final String? token;

  /// `false` = chưa có phiên đã lưu lúc mở app (người dùng phải đăng nhập).
  final bool restorable;

  @override
  Future<AuthSession> login(String username, String password) async {
    if (username != 'ok_user') {
      throw const AppException(
        kind: AppExceptionKind.unauthorized,
        message: 'Sai tên đăng nhập hoặc mật khẩu.',
      );
    }
    return AuthSession(
      user: User(username: username),
      token: token,
    );
  }

  @override
  Future<AuthSession> register(
          String username, String password, String email) =>
      login(username, password);

  @override
  Future<void> logout() async {}

  @override
  Future<String?> currentToken() async => restorable ? token : null;

  @override
  Future<User?> currentUser() async => null;

  @override
  Future<bool> refreshSession() async => false;

  @override
  Future<void> deleteAccount(String password) async {}
}

Widget _wrap(Widget child, {String? token, bool restorable = true}) =>
    ProviderScope(
      overrides: [
        authRepositoryProvider.overrideWithValue(
          _FakeAuthRepository(token, restorable: restorable),
        ),
        // Kho bí mật thật không có trong test; bản RAM không có PIN -> không khoá.
        appLockStoreProvider.overrideWithValue(InMemoryAppLockStore()),
        lastUserStoreProvider.overrideWithValue(InMemoryLastUserStore()),
        healthVaultProvider.overrideWithValue(InMemoryHealthVault()),
        // Thẻ "Lời khuyên" ở Trang chủ gọi mạng thật -> vòng quay tải không bao giờ dừng.
        lifestyleProvider.overrideWith(
          (ref) async => const LifestyleSnapshot(tips: []),
        ),
      ],
      child: MaterialApp(
        // App mặc định là tiếng Việt; test khẳng định chuỗi tiếng Việt nên
        // **ghim** locale thay vì để máy quyết định.
        locale: const Locale('vi'),
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: child,
      ),
    );

void main() {
  group('LoginScreen', () {
    testWidgets('hiển thị đủ ô nhập, nút đăng nhập và link đăng ký', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(_wrap(const LoginScreen()));

      expect(find.text('Tên đăng nhập'), findsOneWidget);
      expect(find.text('Mật khẩu'), findsOneWidget);
      expect(find.widgetWithText(ElevatedButton, 'Đăng nhập'), findsOneWidget);
      expect(find.text('Đăng ký tài khoản mới'), findsOneWidget);
    });

    testWidgets('server lỗi -> hiện SnackBar báo lỗi và KHÔNG crash', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(_wrap(const LoginScreen()));

      await tester.enterText(find.byType(TextField).first, 'test_user');
      await tester.enterText(find.byType(TextField).last, 'test_pass');

      await tester.tap(find.widgetWithText(ElevatedButton, 'Đăng nhập'));
      await tester.pump(); // bắt đầu gọi repository
      await tester.pumpAndSettle();

      // Thông điệp lấy từ AppException -> không phải chuỗi chung chung.
      expect(find.text('Sai tên đăng nhập hoặc mật khẩu.'), findsOneWidget);
      expect(find.byType(LoginScreen), findsOneWidget); // không điều hướng sai

      // Cho SnackBar tự ẩn để không còn Timer/Animation treo khi kết thúc test.
      await tester.pump(const Duration(seconds: 5));
      await tester.pumpAndSettle();
    });

    testWidgets('đăng nhập thành công -> AuthGate chuyển sang AppShell', (
      WidgetTester tester,
    ) async {
      // Chưa có phiên lưu sẵn -> phải đăng nhập; điều hướng do AuthGate (theo trạng thái
      // phiên), không phải do LoginScreen tự push.
      await tester.pumpWidget(
        _wrap(const AuthGate(), token: 'tok', restorable: false),
      );
      await tester.pumpAndSettle();
      expect(find.byType(LoginScreen), findsOneWidget);
      expect(find.byType(AppShell), findsNothing);

      await tester.enterText(find.byType(TextField).first, 'ok_user');
      await tester.enterText(find.byType(TextField).last, 'test_pass');
      await tester.tap(find.widgetWithText(ElevatedButton, 'Đăng nhập'));
      await tester.pumpAndSettle();

      expect(find.byType(AppShell), findsOneWidget);
      expect(find.byType(LoginScreen), findsNothing);
      // 4 tab + nút quét ở giữa.
      expect(find.text('Trang chủ'), findsWidgets);
      expect(find.text('Nhắc thuốc'), findsWidgets);
      expect(find.text('Trợ lý'), findsWidgets);
      expect(find.text('Hồ sơ'), findsWidgets);
      expect(find.byTooltip('Quét thuốc'), findsOneWidget);
    });

    testWidgets('mở app khi đã có phiên lưu sẵn -> vào thẳng AppShell', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(_wrap(const AuthGate(), token: 'tok'));
      await tester.pumpAndSettle();

      expect(find.byType(AppShell), findsOneWidget);
      expect(find.byType(LoginScreen), findsNothing);
    });

    testWidgets('bấm "Đăng ký tài khoản mới" thì mở màn Đăng ký', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(_wrap(const LoginScreen()));

      await tester.tap(find.text('Đăng ký tài khoản mới'));
      await tester.pumpAndSettle();

      expect(find.byType(RegisterScreen), findsOneWidget);
      expect(find.text('Đăng ký tài khoản'), findsOneWidget);
      expect(find.widgetWithText(ElevatedButton, 'Đăng ký'), findsOneWidget);
    });
  });
}
