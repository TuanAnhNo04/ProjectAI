import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/home_tab.dart';
import 'package:image_processing_app/screens/progress_screen.dart';

void main() {
  testWidgets('Trang chủ -> bấm ô Tiến độ -> mở màn báo cáo', (tester) async {
    tester.view.physicalSize = const Size(1080, 2400);
    tester.view.devicePixelRatio = 3;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          lifestyleProvider.overrideWith(
            (ref) async => const LifestyleSnapshot(tips: []),
          ),
        ],
        child: MaterialApp(
          locale: const Locale('vi'),
          theme: AppTheme.light(),
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: HomeTab(onOpenReminders: () {}, onOpenAssistant: () {}),
        ),
      ),
    );
    await tester.pump(const Duration(seconds: 1));

    final tile = find.text('Tiến độ');
    // Cuộn bằng thao tác kéo thật (như ngón tay), rồi bấm đúng vị trí đang hiển thị.
    await tester.drag(find.byType(ListView).first, const Offset(0, -900));
    await tester.pumpAndSettle();
    expect(tester.getCenter(tile).dy, lessThan(800));
    await tester.tap(tile);
    await tester.pump(const Duration(seconds: 1));
    await tester.pump(const Duration(seconds: 1));

    expect(find.byType(ProgressScreen), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
