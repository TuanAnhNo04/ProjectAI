import 'package:flutter/material.dart';

/// Theme Material 3 của app (Day 13).
///
/// Màu chủ đạo lấy từ tím/blue đã dùng ở màn đăng nhập để giao diện đồng bộ,
/// nhưng **tín hiệu** (primary/secondary/tertiary) thì lấy từ `ColorScheme` để
/// tự sinh đủ các biến thể sáng/tối — không gắn cứng mã màu.
///
/// Dùng `ColorScheme.fromSeed` thay vì khai báo tay 30+ màu: bảng màu luôn đúng
/// với Material 3 và không bị "lệch" khi Flutter cập nhật.
class AppTheme {
  const AppTheme._();

  /// Màu hạt giống (seed) — tím y tế.
  static const Color seedColor = Color(0xFF6C4AB6);

  static ThemeData light() => _build(Brightness.light);

  static ThemeData dark() => _build(Brightness.dark);

  static ThemeData _build(Brightness brightness) {
    final scheme = ColorScheme.fromSeed(
      seedColor: seedColor,
      brightness: brightness,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      // Nút bo tròn đồng bộ với phong cách neumorphic sẵn có của app.
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          shape: const StadiumBorder(),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      // SnackBar dùng màu nền của theme để đọc tốt ở cả chế độ tối.
      snackBarTheme: const SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}
