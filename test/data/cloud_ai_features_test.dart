// Cloud AI consent, emergency SSE event, health_context gating, lifestyle
// request building. Dữ liệu GIẢ, không chạm mạng.

import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/adapters/sse_agent_client.dart';
import 'package:image_processing_app/data/local/cloud_ai_consent_store.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';
import 'package:image_processing_app/screens/chat_controller.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'adapters/fake_http.dart';

ResponseBody _sse(String text) => ResponseBody(
      Stream<Uint8List>.fromIterable([Uint8List.fromList(utf8.encode(text))]),
      200,
      headers: {
        Headers.contentTypeHeader: ['text/event-stream'],
      },
    );

void main() {
  group('CloudAiConsentStore', () {
    test('mặc định chưa đồng ý; grant -> có; revoke -> không', () async {
      final store = InMemoryCloudAiConsentStore();
      expect(await store.read(), isNull);
      await store.grant();
      final c = await store.read();
      expect(c, isNotNull);
      expect(c!.policyVersion, kCloudAiPolicyVersion);
      await store.revoke();
      expect(await store.read(), isNull);
    });

    test('sai phiên bản chính sách -> không coi là đã đồng ý', () async {
      final store = InMemoryCloudAiConsentStore(
        initial: CloudAiConsent(
          grantedAt: DateTime(2026, 1, 1),
          policyVersion: kCloudAiPolicyVersion + 1,
        ),
      );
      expect(await store.read(), isNull);
    });

    test('Preferences: policyVersion lệch -> null', () async {
      SharedPreferences.setMockInitialValues({
        PreferencesCloudAiConsentStore.keyVersion: 0,
        PreferencesCloudAiConsentStore.keyGrantedAt:
            DateTime(2026, 1, 1).toIso8601String(),
      });
      final store = PreferencesCloudAiConsentStore();
      expect(await store.read(), isNull);
      await store.grant();
      expect((await store.read())?.policyVersion, kCloudAiPolicyVersion);
    });
  });

  group('SseAgentClient', () {
    test('sự kiện emergency -> EmergencyAlert, tới trước token', () async {
      final f = fakeClient(
        (_) => _sse(
          'event: emergency\n'
          'data: {"matched": ["đau ngực","khó thở"], "message": "Gọi 115 ngay."}\n\n'
          'event: token\ndata: {"text": "Gọi 115 ngay."}\n\n'
          'event: done\ndata: {"sessionId": "s1"}\n\n',
        ),
      );
      final events = await SseAgentClient(f.client)
          .chat(sessionId: 's1', message: 'hi')
          .toList();

      final alert = events.first as EmergencyAlert;
      expect(alert.matched, ['đau ngực', 'khó thở']);
      expect(alert.message, 'Gọi 115 ngay.');
      expect(events[1], isA<TokenDelta>());
      expect(events.last, isA<Done>());
    });

    test('body mang cloud_ai_consent + health_context khi được truyền',
        () async {
      final f = fakeClient((_) => _sse('event: done\ndata: {}\n\n'));
      await SseAgentClient(f.client)
          .chat(
            sessionId: 's1',
            message: 'hi',
            cloudAiConsent: true,
            healthContext: const HealthContext(
              conditions: ['Bệnh giả A'],
              allergies: ['Dị ứng giả B'],
            ),
          )
          .toList();
      expect(f.adapter.last.data, {
        'message': 'hi',
        'session_id': 's1',
        'cloud_ai_consent': true,
        'health_context': {
          'conditions': ['Bệnh giả A'],
          'allergies': ['Dị ứng giả B'],
        },
      });
    });

    test('không có healthContext -> không có khoá health_context', () async {
      final f = fakeClient((_) => _sse('event: done\ndata: {}\n\n'));
      await SseAgentClient(f.client)
          .chat(sessionId: 's1', message: 'hi')
          .toList();
      final data = f.adapter.last.data as Map;
      expect(data.containsKey('health_context'), isFalse);
      expect(data['cloud_ai_consent'], false);
    });
  });

  group('buildHealthContext', () {
    final withConsent = HealthProfile(
      conditions: const ['Bệnh giả A'],
      allergies: const ['Dị ứng giả B'],
      consentGivenAt: DateTime(2026, 1, 1),
    );

    test('không đồng ý AI đám mây -> null', () {
      expect(buildHealthContext(profile: withConsent, cloudAiConsent: false),
          isNull);
    });

    test('hồ sơ chưa consent -> null', () {
      expect(
        buildHealthContext(
          profile: const HealthProfile(conditions: ['X']),
          cloudAiConsent: true,
        ),
        isNull,
      );
    });

    test('đủ hai consent -> có bệnh nền + dị ứng', () {
      final ctx =
          buildHealthContext(profile: withConsent, cloudAiConsent: true)!;
      expect(ctx.conditions, ['Bệnh giả A']);
      expect(ctx.allergies, ['Dị ứng giả B']);
    });
  });

  group('LifestyleRequest.build', () {
    test('hồ sơ chưa consent -> conditions rỗng; thuốc được gộp, bỏ trùng', () {
      final r = LifestyleRequest.build(
        reminderMedicineNames: const ['Thuốc Giả A', 'thuốc giả a'],
        profile: const HealthProfile(
          conditions: ['Bệnh giả'],
          currentMedications: ['Thuốc Giả B'],
        ),
        cloudAiConsent: false,
      );
      expect(r.conditions, isEmpty);
      expect(r.currentMedications, ['Thuốc Giả A', 'Thuốc Giả B']);
      expect(r.recentTopics, isEmpty);
      expect(r.toJson()['cloud_ai_consent'], false);
    });

    test('hồ sơ có consent -> gửi conditions; tối đa 20 thuốc', () {
      final r = LifestyleRequest.build(
        reminderMedicineNames: [for (var i = 0; i < 30; i++) 'Thuốc Giả $i'],
        profile: HealthProfile(
          conditions: const ['Bệnh giả'],
          consentGivenAt: DateTime(2026, 1, 1),
        ),
        cloudAiConsent: true,
      );
      expect(r.conditions, ['Bệnh giả']);
      expect(r.currentMedications, hasLength(20));
    });

    test('snapshot parse + round-trip', () {
      final snap = LifestyleSnapshot.fromJson({
        'tips': [
          {
            'title': 'T',
            'body': 'B',
            'kind': 'medication',
            'citations': [
              {
                'id': 'c1',
                'medicine_name': 'Thuốc Giả',
                'source': 's',
                'excerpt': 'e'
              },
            ],
          },
        ],
        'disclaimer': 'D',
        'generated_by': 'claude',
      });
      expect(snap.isAi, isTrue);
      expect(snap.tips.single.isMedication, isTrue);
      final again = LifestyleSnapshot.fromJson(snap.toJson());
      expect(again.tips.single.citations.single.medicineName, 'Thuốc Giả');
    });
  });
}
