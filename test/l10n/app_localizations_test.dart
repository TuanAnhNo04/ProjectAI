// Day 13 — kiểm chứng i18n: đổi ngôn ngữ thì chuỗi hiển thị đổi theo.
//
// DoD của ngày này là "đổi ngôn ngữ trong app không còn chuỗi hardcode", nên
// test ở đây **đổi `locale`** rồi assert chuỗi thật sự khác — nếu còn chuỗi
// hardcode thì test đỏ ngay.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';

Widget _app(Widget home, Locale locale) => MaterialApp(
      locale: locale,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      home: home,
    );

const _vi = Locale('vi');
const _en = Locale('en');

void main() {
  group('AppLocalizations', () {
    test('hỗ trợ cả vi và en', () {
      expect(AppLocalizations.supportedLocales, contains(_vi));
      expect(AppLocalizations.supportedLocales, contains(_en));
    });

    testWidgets('đổi vi -> en đổi cả chuỗi hiển thị', (tester) async {
      await tester.pumpWidget(_app(const _Probe(), _vi));
      final vi = tester.widget<Text>(find.byKey(const Key('probe'))).data;
      expect(vi, 'Vui lòng nhập tên đăng nhập.');

      await tester.pumpWidget(_app(const _Probe(), _en));
      final en = tester.widget<Text>(find.byKey(const Key('probe'))).data;
      expect(en, 'Please enter your username.');
      expect(en, isNot(vi));
    });

    testWidgets('placeholder validationPasswordTooShort lấy đúng số', (
      tester,
    ) async {
      await tester.pumpWidget(_app(const _Probe(), _vi));
      await tester.pumpAndSettle();
      expect(find.textContaining('6 ký tự'), findsOneWidget);
    });

    testWidgets('disclaimer đổi theo ngôn ngữ và luôn hiện', (tester) async {
      await tester.pumpWidget(_app(const MedicalDisclaimer(), _vi));
      await tester.pumpAndSettle();
      expect(
        find.textContaining('không thay thế chẩn đoán'),
        findsOneWidget,
      );

      await tester.pumpWidget(_app(const MedicalDisclaimer(), _en));
      await tester.pumpAndSettle();
      expect(find.textContaining('does not replace advice'), findsOneWidget);
    });
  });
}

/// Widget thử: hiện 1 thông điệp validate lấy từ `AppLocalizations`.
class _Probe extends StatelessWidget {
  const _Probe();

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return Scaffold(
      body: Column(
        children: [
          Text(
            CredentialIssue.usernameRequired.localize(l10n),
            key: const Key('probe'),
          ),
          Text(CredentialIssue.passwordTooShort.localize(l10n)),
        ],
      ),
    );
  }
}
