import 'package:flutter/material.dart';

/// Bảng màu + đổ bóng kiểu **neumorphism** (thẻ "nổi" mềm trên nền mint nhạt) — phong cách
/// lấy cảm hứng từ ứng dụng nhắc thuốc *remedi*.
///
/// Đặt thành `ThemeExtension` để mọi widget lấy cùng một bộ màu/đổ bóng theo sáng/tối,
/// thay vì mã màu cứng từng màn: `NeuTheme.of(context)`.
@immutable
class NeuTheme extends ThemeExtension<NeuTheme> {
  const NeuTheme({
    required this.background,
    required this.surface,
    required this.highlight,
    required this.shadow,
    required this.accent,
    required this.accentSoft,
    required this.textStrong,
    required this.textMuted,
  });

  /// Nền màn hình **và** nền thẻ — neumorphism cần hai thứ cùng màu, chỉ khác bóng.
  final Color background;

  /// Nền thẻ (trắng ở sáng, hơi sáng hơn nền ở tối).
  final Color surface;

  /// Bóng sáng (góc trên-trái).
  final Color highlight;

  /// Bóng tối (góc dưới-phải).
  final Color shadow;

  /// Màu nhấn (teal) cho nút chính, công tắc, giờ uống thuốc.
  final Color accent;

  /// Nền nhạt của màu nhấn (chip, vòng icon).
  final Color accentSoft;

  /// Chữ chính / chữ phụ.
  final Color textStrong;
  final Color textMuted;

  static const NeuTheme light = NeuTheme(
    background: Color(0xFFF4F8F8),
    surface: Color(0xFFFFFFFF),
    highlight: Color(0xFFFFFFFF),
    shadow: Color(0x0F000000),
    // Teal đậm hơn bản mẫu (#1ECBC0) để chữ trắng trên nút đạt tương phản ≥ 3:1.
    accent: Color(0xFF0E9F95),
    accentSoft: Color(0xFFD5F3EF),
    textStrong: Color(0xFF0F3B44),
    textMuted: Color(0xFF55747C),
  );

  static const NeuTheme dark = NeuTheme(
    background: Color(0xFF0F1B1E),
    surface: Color(0xFF172629),
    highlight: Color(0x14FFFFFF),
    shadow: Color(0x99000000),
    accent: Color(0xFF2DD4C4),
    accentSoft: Color(0xFF14403F),
    textStrong: Color(0xFFE6F6F8),
    textMuted: Color(0xFFA2BEC4),
  );

  /// Lấy bộ màu của theme hiện tại (có dự phòng để widget dùng được trong test không
  /// bọc `AppTheme`).
  static NeuTheme of(BuildContext context) {
    final theme = Theme.of(context);
    return theme.extension<NeuTheme>() ??
        (theme.brightness == Brightness.dark ? dark : light);
  }

  /// Bóng phẳng, rất nhẹ cho thẻ (một lớp duy nhất). Tham số giữ để tương thích.
  List<BoxShadow> raised({double distance = 4, double blur = 12}) => [
        BoxShadow(
          color: shadow,
          offset: const Offset(0, 4),
          blurRadius: 12,
        ),
      ];

  /// Bóng nhẹ cho thành phần nhỏ (nút tròn, chip).
  List<BoxShadow> soft() => raised();
  @override
  NeuTheme copyWith({
    Color? background,
    Color? surface,
    Color? highlight,
    Color? shadow,
    Color? accent,
    Color? accentSoft,
    Color? textStrong,
    Color? textMuted,
  }) =>
      NeuTheme(
        background: background ?? this.background,
        surface: surface ?? this.surface,
        highlight: highlight ?? this.highlight,
        shadow: shadow ?? this.shadow,
        accent: accent ?? this.accent,
        accentSoft: accentSoft ?? this.accentSoft,
        textStrong: textStrong ?? this.textStrong,
        textMuted: textMuted ?? this.textMuted,
      );

  @override
  NeuTheme lerp(ThemeExtension<NeuTheme>? other, double t) {
    if (other is! NeuTheme) return this;
    return NeuTheme(
      background: Color.lerp(background, other.background, t)!,
      surface: Color.lerp(surface, other.surface, t)!,
      highlight: Color.lerp(highlight, other.highlight, t)!,
      shadow: Color.lerp(shadow, other.shadow, t)!,
      accent: Color.lerp(accent, other.accent, t)!,
      accentSoft: Color.lerp(accentSoft, other.accentSoft, t)!,
      textStrong: Color.lerp(textStrong, other.textStrong, t)!,
      textMuted: Color.lerp(textMuted, other.textMuted, t)!,
    );
  }
}

/// Theme Material 3 của app.
///
/// Một `ColorScheme.fromSeed` cho cả app, rồi ghi đè các vai trò chính bằng bảng màu
/// [NeuTheme] để các widget Material mặc định (nút, ô nhập, SnackBar…) cũng ăn khớp
/// với phong cách neumorphic.
class AppTheme {
  const AppTheme._();

  /// Màu hạt giống (seed) — teal y tế.
  static const Color seedColor = Color(0xFF14B8A6);

  /// Vùng chạm tối thiểu (Material: 48dp).
  static const double minTouchTarget = 48;

  static ThemeData light() => _build(Brightness.light, NeuTheme.light);

  static ThemeData dark() => _build(Brightness.dark, NeuTheme.dark);

  static ThemeData _build(Brightness brightness, NeuTheme neu) {
    final isDark = brightness == Brightness.dark;
    final scheme = ColorScheme.fromSeed(
      seedColor: seedColor,
      brightness: brightness,
    ).copyWith(
      primary: neu.accent,
      onPrimary: isDark ? const Color(0xFF00322E) : Colors.white,
      primaryContainer: neu.accentSoft,
      onPrimaryContainer: neu.textStrong,
      secondaryContainer: neu.accentSoft,
      onSecondaryContainer: neu.textStrong,
      surface: neu.background,
      onSurface: neu.textStrong,
      onSurfaceVariant: neu.textMuted,
    );

    const buttonShape = StadiumBorder();
    const buttonMinSize = Size(64, minTouchTarget);
    final baseTextTheme =
        (isDark ? Typography.whiteMountainView : Typography.blackMountainView)
            .apply(bodyColor: neu.textStrong, displayColor: neu.textStrong);
    final textTheme = baseTextTheme.copyWith(
      bodyMedium: baseTextTheme.bodyMedium?.copyWith(fontSize: 16),
      bodyLarge: baseTextTheme.bodyLarge?.copyWith(fontSize: 17),
      bodySmall: baseTextTheme.bodySmall?.copyWith(fontSize: 14),
      titleLarge: baseTextTheme.titleLarge
          ?.copyWith(fontSize: 22, fontWeight: FontWeight.w700),
      titleMedium: baseTextTheme.titleMedium
          ?.copyWith(fontSize: 18, fontWeight: FontWeight.w700),
      labelLarge: baseTextTheme.labelLarge
          ?.copyWith(fontSize: 16, fontWeight: FontWeight.w600),
    );

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      scaffoldBackgroundColor: neu.background,
      canvasColor: neu.background,
      textTheme: textTheme,
      extensions: <ThemeExtension<dynamic>>[neu],
      appBarTheme: AppBarTheme(
        centerTitle: false,
        backgroundColor: neu.background,
        foregroundColor: neu.textStrong,
        surfaceTintColor: Colors.transparent,
        scrolledUnderElevation: 0,
        elevation: 0,
        titleTextStyle: textTheme.titleLarge?.copyWith(
          fontWeight: FontWeight.w700,
          color: neu.textStrong,
        ),
      ),
      cardTheme: CardTheme(
        elevation: 0,
        color: neu.surface,
        surfaceTintColor: Colors.transparent,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      // Thanh dưới cùng màu nền, tách khỏi nội dung bằng bóng (không ám xám như mặc định M3).
      bottomAppBarTheme: BottomAppBarTheme(
        color: neu.surface,
        surfaceTintColor: Colors.transparent,
        elevation: 12,
        shadowColor: neu.shadow,
      ),
      // Nút quét ở giữa: teal đậm, nổi bật như mẫu.
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: neu.accent,
        foregroundColor: scheme.onPrimary,
        elevation: 6,
        shape: const CircleBorder(),
      ),
      dialogTheme: DialogTheme(
        backgroundColor: neu.surface,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      ),
      bottomSheetTheme: BottomSheetThemeData(
        backgroundColor: neu.surface,
        surfaceTintColor: Colors.transparent,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          shape: buttonShape,
          minimumSize: buttonMinSize,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          shape: buttonShape,
          minimumSize: buttonMinSize,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          shape: buttonShape,
          minimumSize: buttonMinSize,
          foregroundColor: neu.accent,
          side: BorderSide(color: neu.accent.withOpacity(0.5)),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          minimumSize: buttonMinSize,
          foregroundColor: neu.accent,
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: MaterialStateProperty.resolveWith(
          (states) => states.contains(MaterialState.selected)
              ? Colors.white
              : neu.textMuted,
        ),
        trackColor: MaterialStateProperty.resolveWith(
          (states) => states.contains(MaterialState.selected)
              ? neu.accent
              : neu.shadow.withOpacity(0.25),
        ),
        trackOutlineColor:
            const MaterialStatePropertyAll<Color>(Colors.transparent),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        // Ô nhập "lõm" nhẹ: tối hơn nền một chút.
        fillColor: isDark ? const Color(0xFF1E3338) : const Color(0xFFEEF4F4),
        floatingLabelBehavior: FloatingLabelBehavior.always,
        labelStyle: TextStyle(
            fontSize: 14, color: neu.textMuted, fontWeight: FontWeight.w600),
        floatingLabelStyle: TextStyle(
            fontSize: 14, color: neu.textMuted, fontWeight: FontWeight.w600),
        hintStyle: TextStyle(fontSize: 15, color: neu.textMuted),
        helperStyle: TextStyle(fontSize: 15, color: neu.textMuted),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide:
              BorderSide(color: neu.textMuted.withOpacity(0.25), width: 1.5),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide:
              BorderSide(color: neu.textMuted.withOpacity(0.25), width: 1.5),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: neu.accent, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: scheme.error, width: 1.5),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: scheme.error, width: 2),
        ),
      ),
      snackBarTheme: const SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
      ),
      dividerTheme: DividerThemeData(
        color: neu.textMuted.withOpacity(0.2),
        thickness: 1,
      ),
    );
  }
}
