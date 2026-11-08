import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/config/app_config.dart';
import 'package:image_processing_app/core/display_settings.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/local/health_database.dart';
import 'package:image_processing_app/data/local/notification_service.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'screens/auth_gate.dart';
import 'screens/consent_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Verify --dart-define: in ra server mà app đang trỏ tới (Day 5).
  if (kDebugMode) debugPrint('MedAssist → API: ${AppConfig.describe()}');
  // Day 27: khởi tạo notification channel (không chặn UI nếu lỗi).
  if (await NotificationService.instance.init()) {
    // Dựng lại lịch nhắc từ dữ liệu đã lưu (múi giờ đổi / máy khởi động lại).
    try {
      await NotificationService.instance
          .rescheduleAll(await HealthDatabase.instance.getReminders());
    } on Object {
      // Nhắc thuốc hỏng không được làm app không mở được; không log nội dung.
    }
  }
  // ProviderScope là nơi duy nhất chứa toàn bộ dependency (Day 9).
  runApp(const ProviderScope(child: MedAssistApp()));
}

class MyApp extends MedAssistApp {
  const MyApp({super.key});
}

/// Gốc ứng dụng.
///
/// Tách [MedAssistApp] khỏi `ProviderScope` để **test** có thể bọc
/// `MyApp` trong `ProviderScope` riêng và thay adapter bằng fake — không đụng
/// code chạy thật.
///
/// [MedAssistApp] là `ConsumerWidget` vì cần `ref` để cài callback 401
/// (Day 11): khi `ApiClient` nhận 401 ở endpoint có bảo vệ, nó gọi
/// `authControllerProvider.handleUnauthorized()` rồi điều hướng về Login.
class MedAssistApp extends ConsumerWidget {
  const MedAssistApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final display = ref.watch(displaySettingsProvider);
    return MaterialApp(
      onGenerateTitle: (context) => AppLocalizations.of(context).appTitle,
      theme: AppTheme.light(),
      darkTheme: AppTheme.dark(),
      // Mặc định `system` (bám theo máy); người dùng đổi được ở Hồ sơ > Hiển thị.
      themeMode: display.themeMode,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      // App là tiếng Việt: bắt đầu bằng `vi`, và khi máy đặt ngôn ngữ lạ
      // (ví dụ `fr`) thì rơi về `vi` thay vì trống.
      locale: const Locale('vi'),
      localeResolutionCallback: (locale, supported) {
        for (final candidate in supported) {
          if (candidate.languageCode == locale?.languageCode) return candidate;
        }
        return supported.first;
      },
      // Lần đầu (hoặc khi chính sách đổi) phải đồng ý; sau đó màn gốc theo phiên đăng
      // nhập: có phiên → AppShell, không → Login (xem AuthGate).
      home: const ConsentGate(child: AuthGate()),
      builder: (context, child) => ProviderScope(
        overrides: [
          unauthorizedNotifierProvider.overrideWithValue(
            () => _onUnauthorized(context, ref),
          ),
        ],
        // Cỡ chữ trong app nhân thêm vào cỡ chữ hệ thống, không thay thế nó.
        child: MediaQuery(
          data: MediaQuery.of(context).copyWith(
            textScaler: TextScaler.linear(
              MediaQuery.textScalerOf(context).scale(1) *
                  display.textSize.factor,
            ),
          ),
          child: child ?? const SizedBox.shrink(),
        ),
      ),
    );
  }

  /// Xoá token + đưa người dùng về màn Login.
  ///
  /// `pushAndRemoveUntil` xoá sạch stack: nếu chỉ `push` Login thì người dùng
  /// bấm "back" sẽ quay lại màn đang yêu cầu và lại gặp 401 ngay.
  static void _onUnauthorized(BuildContext context, WidgetRef ref) {
    ref.read(authControllerProvider.notifier).handleUnauthorized();

    final navigator = Navigator.of(context);
    if (navigator.canPop()) {
      // Đã ở trong app (có màn trên stack) -> quay về gốc Login.
      navigator.popUntil((route) => route.isFirst);
    }
  }
}
