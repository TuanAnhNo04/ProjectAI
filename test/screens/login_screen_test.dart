// Day 3 — test đầu tiên cho màn hình Đăng nhập.
//
// Bối cảnh: flutter_test CHẶN HTTP thật và luôn trả về status 400 cho mọi request.
// Vì vậy các test dưới đây chạy hoàn toàn offline và kiểm tra đúng nhánh xử lý lỗi.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/screens/login_screen.dart';
import 'package:image_processing_app/screens/register_screen.dart';

Widget _wrap(Widget child) => MaterialApp(home: child);

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
      await tester.pump(); // bắt đầu gọi API
      await tester.pump(const Duration(milliseconds: 300)); // nhận response 400
      await tester.pump(); // dựng SnackBar

      expect(find.text('Đăng nhập thất bại!'), findsOneWidget);
      expect(find.byType(LoginScreen), findsOneWidget); // không điều hướng sai

      // Cho SnackBar tự ẩn để không còn Timer/Animation treo khi kết thúc test.
      await tester.pump(const Duration(seconds: 5));
      await tester.pumpAndSettle();
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
