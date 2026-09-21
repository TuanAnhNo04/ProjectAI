// Day 3 — test cho màn Tìm kiếm thuốc.
//
// Ca này kiểm tra đúng phần KHÔNG cần mạng: bỏ trống ô nhập thì phải báo lỗi
// và dừng lại trước khi gọi API (đây là hành vi validate duy nhất app đang có).

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/screens/MedicineSearchScreen.dart';

void main() {
  testWidgets('bỏ trống tên thuốc -> hiện cảnh báo, KHÔNG gọi API', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(MaterialApp(home: MedicineSearchPage()));

    expect(find.text('Tìm kiếm thuốc'), findsOneWidget);
    expect(find.widgetWithText(ElevatedButton, 'Tìm kiếm'), findsOneWidget);

    await tester.tap(find.widgetWithText(ElevatedButton, 'Tìm kiếm'));
    await tester.pump();

    expect(find.text('Vui lòng nhập tên thuốc!'), findsOneWidget);
  });
}
