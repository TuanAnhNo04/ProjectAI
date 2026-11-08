// Nguồn gốc câu trả lời: parser, re-validate URL, hộp xác nhận mở link.
// Dữ liệu GIẢ, không chạm mạng (kênh native được mock).

import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/adapters/device_channel.dart';
import 'package:image_processing_app/data/adapters/sse_agent_client.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/external_link.dart';

import '../data/adapters/fake_http.dart';

ResponseBody _sse(String t) => ResponseBody(
      Stream<Uint8List>.value(Uint8List.fromList(utf8.encode(t))),
      200,
      headers: {
        Headers.contentTypeHeader: ['text/event-stream'],
      },
    );

void main() {
  test('SSE citation: có và không có 4 trường mới', () async {
    const s = 'event: citation\n'
        'data: {"documentId":"R1","quote":"q","title":"T","source":"Nguồn giả",'
        '"isAuthoritative":true,"sourceUrl":"https://example.invalid/x",'
        '"retrievedAt":"2026-10-01","license":"CC-FAKE","reviewedBy":"Dược sĩ giả"}\n\n'
        'event: citation\n'
        'data: {"documentId":"R2","quote":"q"}\n\n'
        'event: done\ndata: {}\n\n';
    final f = fakeClient((_) => _sse(s));
    final ev = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'hi')
        .toList();
    final a = ev[0] as Citation;
    expect(a.sourceUrl, 'https://example.invalid/x');
    expect(a.retrievedAt, '2026-10-01');
    expect(a.license, 'CC-FAKE');
    expect(a.reviewedBy, 'Dược sĩ giả');
    expect(a.isAuthoritative, isTrue);
    final b = ev[1] as Citation;
    expect([b.sourceUrl, b.retrievedAt, b.license, b.reviewedBy, b.source],
        everyElement(''));
  });

  test('LifestyleCitation: snake_case + round trip + thiếu -> rỗng', () {
    final c = LifestyleCitation.fromJson({
      'id': 'd',
      'source_url': 'https://example.invalid/d',
      'retrieved_at': '2026-10-02',
      'license': 'L',
      'reviewed_by': 'R',
    });
    final r = LifestyleCitation.fromJson(c.toJson());
    expect(r.sourceUrl, 'https://example.invalid/d');
    expect(r.retrievedAt, '2026-10-02');
    expect(r.license, 'L');
    expect(r.reviewedBy, 'R');
    final e = LifestyleCitation.fromJson({'id': 'x'});
    expect([e.sourceUrl, e.retrievedAt, e.license, e.reviewedBy],
        everyElement(''));
  });

  test('safeExternalUri từ chối http/userinfo/cổng lạ/javascript:', () {
    expect(safeExternalUri('https://example.invalid/a'), isNotNull);
    expect(safeExternalUri('https://example.invalid:443/a'), isNotNull);
    for (final bad in [
      'http://example.invalid/a',
      'https://user:pw@example.invalid/a',
      'https://example.invalid:8443/a',
      'javascript:alert(1)',
      'https:///nohost',
      '',
      null,
    ]) {
      expect(safeExternalUri(bad), isNull, reason: '$bad');
    }
  });

  test('formatSourceDate: dd/MM/yyyy, chuỗi lỗi bị bỏ qua', () {
    expect(formatSourceDate('2026-10-01'), '01/10/2026');
    expect(formatSourceDate('không phải ngày'), isNull);
    expect(formatSourceDate(''), isNull);
  });

  group('openExternalLink', () {
    Future<List<MethodCall>> pump(WidgetTester tester,
        {required bool nativeFails}) async {
      final calls = <MethodCall>[];
      const ch = MethodChannel('test/device');
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(ch, (call) async {
        calls.add(call);
        if (nativeFails) throw PlatformException(code: 'no_activity');
        return null;
      });
      addTearDown(() => TestDefaultBinaryMessengerBinding
          .instance.defaultBinaryMessenger
          .setMockMethodCallHandler(ch, null));
      await tester.pumpWidget(ProviderScope(
        overrides: [
          deviceChannelProvider
              .overrideWithValue(DeviceChannel(channel: ch, supported: true)),
        ],
        child: MaterialApp(
          locale: const Locale('vi'),
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          home: Scaffold(
            body: Consumer(
              builder: (context, ref, _) => TextButton(
                onPressed: () => openExternalLink(
                    context, ref, 'https://example.invalid/doc'),
                child: const Text('go'),
              ),
            ),
          ),
        ),
      ));
      return calls;
    }

    testWidgets('hiện hộp xác nhận rồi mở', (tester) async {
      final calls = await pump(tester, nativeFails: false);
      await tester.tap(find.text('go'));
      await tester.pumpAndSettle();
      expect(find.text('Rời ứng dụng để đọc bài?'), findsOneWidget);
      expect(calls, isEmpty);
      await tester.tap(find.byType(FilledButton));
      await tester.pumpAndSettle();
      expect(calls.single.method, 'openUrl');
    });

    testWidgets('DeviceChannelException -> chép link', (tester) async {
      String? copied;
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(SystemChannels.platform, (c) async {
        if (c.method == 'Clipboard.setData') {
          copied = (c.arguments as Map)['text'] as String?;
        }
        return null;
      });
      addTearDown(() => TestDefaultBinaryMessengerBinding
          .instance.defaultBinaryMessenger
          .setMockMethodCallHandler(SystemChannels.platform, null));
      await pump(tester, nativeFails: true);
      await tester.tap(find.text('go'));
      await tester.pumpAndSettle();
      await tester.tap(find.byType(FilledButton));
      await tester.pumpAndSettle();
      expect(copied, 'https://example.invalid/doc');
      expect(find.byType(SnackBar), findsOneWidget);
    });
  });
}
