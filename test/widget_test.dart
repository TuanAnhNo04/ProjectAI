// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:image_processing_app/main.dart';

// LƯU Ý LỊCH SỬ (Day 3): phần dưới đây trước là test "Counter increments" mặc định của
// template Flutter (tìm find.text('0') / Icons.add) nhưng app KHÔNG có counter => luôn FAIL.
// Đã thay bằng smoke test thật của app.

void main() {
  testWidgets('App khởi động và hiển thị màn hình Đăng nhập', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const MyApp());

    expect(find.byType(TextField), findsNWidgets(2));
    expect(find.widgetWithText(ElevatedButton, 'Đăng nhập'), findsOneWidget);
    expect(find.text('Đăng ký tài khoản mới'), findsOneWidget);
  });
}
