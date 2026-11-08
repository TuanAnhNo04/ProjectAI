// Két mã hoá (HealthVault), di chuyển hồ sơ cũ, trí nhớ triệu chứng (opt-in).
// Toàn bộ dữ liệu là GIẢ; không chạm mạng/đĩa.

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/assistant_memory_store.dart';
import 'package:image_processing_app/data/local/cloud_ai_consent_store.dart';
import 'package:image_processing_app/data/local/health_database.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/data/local/notification_service.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';
import 'package:image_processing_app/screens/chat_controller.dart';

class _LegacyDb implements HealthDatabase {
  _LegacyDb(this.row);
  HealthProfile? row;
  @override
  Future<HealthProfile?> getProfile() async => row;
  @override
  Future<void> deleteProfile() async => row = null;
  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

class _Notif implements NotificationService {
  @override
  Future<void> cancelAll() async {}
  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

class _Agent implements AgentClient {
  _Agent(this.events);
  final List<AgentEvent> events;
  HealthContext? lastContext;
  @override
  Stream<AgentEvent> chat({
    required String sessionId,
    required String message,
    List<String> currentMedications = const [],
    bool cloudAiConsent = false,
    HealthContext? healthContext,
  }) {
    lastContext = healthContext;
    return Stream<AgentEvent>.fromIterable(events);
  }
}

class _FixedProfile extends HealthProfileController {
  static HealthProfile? value;
  @override
  Future<HealthProfile?> build() => SynchronousFuture(value);
}

final _consented = HealthProfile(
  conditions: const ['Bệnh giả A'],
  allergies: const ['Dị ứng giả B'],
  currentMedications: const ['Thuốc Giả C'],
  consentGivenAt: DateTime(2026, 1, 1),
);

ProviderContainer _container({
  InMemoryHealthVault? vault,
  HealthProfile? profile,
  bool memory = false,
  bool cloud = false,
  _Agent? agent,
  _LegacyDb? legacy,
}) {
  _FixedProfile.value = profile;
  final c = ProviderContainer(overrides: [
    healthVaultProvider.overrideWithValue(vault ?? InMemoryHealthVault()),
    assistantMemoryStoreProvider
        .overrideWithValue(InMemoryAssistantMemoryStore(value: memory)),
    cloudAiConsentStoreProvider.overrideWithValue(InMemoryCloudAiConsentStore(
      initial: cloud
          ? CloudAiConsent(
              grantedAt: DateTime(2026, 1, 1),
              policyVersion: kCloudAiPolicyVersion)
          : null,
    )),
    healthDatabaseProvider.overrideWithValue(legacy ?? _LegacyDb(null)),
    notificationServiceProvider.overrideWithValue(_Notif()),
    if (profile != null)
      healthProfileControllerProvider.overrideWith(_FixedProfile.new),
    if (agent != null) agentClientProvider.overrideWithValue(agent),
  ]);
  addTearDown(c.dispose);
  return c;
}

void main() {
  group('InMemoryHealthVault', () {
    test('profile CRUD + sanitise', () async {
      final v = InMemoryHealthVault();
      expect(await v.readProfile(), isNull);
      await v.writeProfile(HealthProfile(
        conditions: ['  Bệnh giả  ', '', 'x' * 300],
        consentGivenAt: DateTime(2026, 2, 3),
      ));
      final p = (await v.readProfile())!;
      expect(p.conditions.first, 'Bệnh giả');
      expect(p.conditions, hasLength(2));
      expect(p.conditions.last.length, kVaultProfileItemLen);
      expect(p.hasConsent, isTrue);
      await v.deleteProfile();
      expect(await v.readProfile(), isNull);
    });

    test('episodes: newest first, sanitised, capped at 100', () async {
      final v = InMemoryHealthVault();
      for (var i = 0; i < 105; i++) {
        await v.appendEpisode(SymptomEpisode(
          at: DateTime(2026, 1, 1).add(Duration(hours: i)),
          terms: ['t$i', 'a', 'b', 'c', 'd', 'e', 'f'],
          meds: [for (var m = 0; m < 15; m++) 'm$m'],
        ));
      }
      final eps = await v.readEpisodes();
      expect(eps, hasLength(kVaultMaxEpisodes));
      expect(eps.first.terms.first, 't104');
      expect(eps.last.terms.first, 't5');
      expect(eps.first.terms, hasLength(kVaultMaxTerms));
      expect(eps.first.meds, hasLength(kVaultMaxMeds));
    });

    test('term length clipped; empty terms ignored; delete + clear', () async {
      final v = InMemoryHealthVault();
      await v.appendEpisode(
          SymptomEpisode(at: DateTime(2026, 1, 1), terms: const ['  ', '']));
      expect(await v.readEpisodes(), isEmpty);
      final at = DateTime(2026, 1, 2);
      await v.appendEpisode(SymptomEpisode(at: at, terms: ['y' * 200]));
      await v.appendEpisode(
          SymptomEpisode(at: DateTime(2026, 1, 3), terms: const ['ho']));
      expect((await v.readEpisodes()).last.terms.first.length, kVaultTermLen);
      expect(await v.deleteEpisode(at), isTrue);
      expect(await v.deleteEpisode(at), isFalse);
      expect(await v.readEpisodes(), hasLength(1));
      await v.clearEpisodes();
      expect(await v.readEpisodes(), isEmpty);
    });

    test('unreadable -> VaultUnreadable, never overwrites', () async {
      final v = InMemoryHealthVault();
      await v.writeProfile(_consented);
      v.data['medassist.vault.profile'] = '{corrupt';
      await expectLater(v.readProfile(), throwsA(isA<VaultUnreadable>()));
      await expectLater(v.writeProfile(const HealthProfile()),
          throwsA(isA<VaultUnreadable>()));
      expect(v.data['medassist.vault.profile'], '{corrupt');

      v.data['medassist.vault.episodes'] = 'not json';
      await expectLater(
          v.appendEpisode(
              SymptomEpisode(at: DateTime(2026, 1, 1), terms: const ['ho'])),
          throwsA(isA<VaultUnreadable>()));
      expect(v.data['medassist.vault.episodes'], 'not json');

      v.unreadable = true;
      await expectLater(v.readEpisodes(), throwsA(isA<VaultUnreadable>()));
      v.unreadable = false;
      await v.wipe(); // đường phục hồi
      expect(await v.readProfile(), isNull);
      expect(await v.readEpisodes(), isEmpty);
    });

    test('recentSymptomTerms: 14 days, distinct, newest first, max 5', () {
      final now = DateTime(2026, 10, 30);
      final eps = [
        SymptomEpisode(at: DateTime(2026, 10, 29), terms: const ['Ho', 'sốt']),
        SymptomEpisode(at: DateTime(2026, 10, 28), terms: const ['ho', 'đau']),
        SymptomEpisode(at: DateTime(2026, 9, 1), terms: const ['quá cũ']),
        SymptomEpisode(
            at: DateTime(2026, 10, 27), terms: const ['a', 'b', 'c', 'd']),
      ];
      expect(recentSymptomTerms(eps, now), ['Ho', 'sốt', 'đau', 'a', 'b']);
    });
  });

  group('HealthProfileController migration', () {
    test('legacy -> vault, verified, legacy row deleted, idempotent', () async {
      final legacy = _LegacyDb(_consented);
      final vault = InMemoryHealthVault();
      final c = _container(vault: vault, legacy: legacy);
      final ctl = c.read(healthProfileControllerProvider.notifier);

      final first = await ctl.load();
      expect(first?.conditions, ['Bệnh giả A']);
      expect(first?.hasConsent, isTrue);
      expect(legacy.row, isNull);
      expect(await vault.readProfile(), isNotNull);

      final second = await ctl.load();
      expect(second?.currentMedications, ['Thuốc Giả C']);
      expect(legacy.row, isNull);
    });

    test('vault already has profile -> stray legacy row removed', () async {
      final legacy = _LegacyDb(const HealthProfile(conditions: ['cũ']));
      final vault = InMemoryHealthVault();
      await vault.writeProfile(_consented);
      final c = _container(vault: vault, legacy: legacy);
      final p = await c.read(healthProfileControllerProvider.notifier).load();
      expect(p?.conditions, ['Bệnh giả A']);
      expect(legacy.row, isNull);
    });

    test('unreadable vault: error state, legacy kept, nothing overwritten',
        () async {
      final legacy = _LegacyDb(_consented);
      final vault = InMemoryHealthVault()..unreadable = true;
      final c = _container(vault: vault, legacy: legacy);
      final ctl = c.read(healthProfileControllerProvider.notifier);
      await ctl.load();
      expect(c.read(healthProfileControllerProvider).error,
          isA<VaultUnreadable>());
      expect(legacy.row, isNotNull);
      await ctl.save(const HealthProfile(conditions: ['mới']));
      expect(c.read(healthProfileControllerProvider).hasError, isTrue);
      vault.unreadable = false;
      expect(await vault.readProfile(), isNull);
    });

    test('save/giveConsent persist into vault', () async {
      final vault = InMemoryHealthVault();
      final c = _container(vault: vault);
      final ctl = c.read(healthProfileControllerProvider.notifier);
      await ctl.save(const HealthProfile(allergies: ['Dị ứng giả']));
      await ctl.giveConsent();
      final p = (await vault.readProfile())!;
      expect(p.allergies, ['Dị ứng giả']);
      expect(p.hasConsent, isTrue);
    });
  });

  group('memory toggle + wipe', () {
    test('cannot enable without profile consent', () async {
      final c = _container(profile: const HealthProfile());
      final ok = await c
          .read(assistantMemoryControllerProvider.notifier)
          .setEnabled(true);
      expect(ok, isFalse);
      expect(c.read(assistantMemoryControllerProvider).valueOrNull, isFalse);
    });

    test('can enable with consent; default off', () async {
      final c = _container(profile: _consented);
      expect(await c.read(assistantMemoryControllerProvider.future), isFalse);
      expect(
          await c
              .read(assistantMemoryControllerProvider.notifier)
              .setEnabled(true),
          isTrue);
      expect(c.read(assistantMemoryControllerProvider).valueOrNull, isTrue);
    });

    test('deleteProfile wipes vault (profile + episodes) and turns memory off',
        () async {
      final vault = InMemoryHealthVault();
      await vault.writeProfile(_consented);
      await vault.appendEpisode(
          SymptomEpisode(at: DateTime(2026, 1, 1), terms: const ['ho']));
      final c = _container(vault: vault, memory: true);
      await c.read(healthProfileControllerProvider.notifier).deleteProfile();
      expect(await vault.readProfile(), isNull);
      expect(await vault.readEpisodes(), isEmpty);
      expect(await c.read(assistantMemoryControllerProvider.future), isFalse);
    });
  });

  group('buildHealthContext', () {
    final now = DateTime(2026, 10, 30);
    final eps = [
      SymptomEpisode(at: DateTime(2026, 10, 29), terms: const ['ho', 'sốt']),
    ];

    test('memory off -> no recent_symptoms', () {
      expect(
        buildHealthContext(
            profile: _consented,
            cloudAiConsent: false,
            memoryEnabled: false,
            episodes: eps,
            now: now),
        isNull,
      );
      final ctx = buildHealthContext(
          profile: _consented,
          cloudAiConsent: true,
          memoryEnabled: false,
          episodes: eps,
          now: now)!;
      expect(ctx.recentSymptoms, isEmpty);
      expect(ctx.conditions, ['Bệnh giả A']);
    });

    test('memory on + profile consent -> symptoms only (no cloud)', () {
      final ctx = buildHealthContext(
          profile: _consented,
          cloudAiConsent: false,
          memoryEnabled: true,
          episodes: eps,
          now: now)!;
      expect(ctx.recentSymptoms, ['ho', 'sốt']);
      expect(ctx.conditions, isEmpty);
      expect(ctx.allergies, isEmpty);
    });

    test('no profile consent -> null even with memory on', () {
      expect(
        buildHealthContext(
            profile: const HealthProfile(),
            cloudAiConsent: true,
            memoryEnabled: true,
            episodes: eps,
            now: now),
        isNull,
      );
      expect(
        buildHealthContext(
            profile: null,
            cloudAiConsent: true,
            memoryEnabled: true,
            episodes: eps,
            now: now),
        isNull,
      );
    });
  });

  group('ChatController memory gating', () {
    const noted = SymptomNoted(terms: ['buồn nôn'], recurring: true);

    Future<(ProviderContainer, InMemoryHealthVault, _Agent)> run({
      required bool memory,
      HealthProfile? profile,
    }) async {
      final vault = InMemoryHealthVault();
      await vault.appendEpisode(
          SymptomEpisode(at: DateTime.now(), terms: const ['ho']));
      final agent = _Agent(const [noted, Done()]);
      final c = _container(
          vault: vault, profile: profile, memory: memory, agent: agent);
      c.listen(chatControllerProvider, (_, __) {});
      await c.read(assistantMemoryControllerProvider.future);
      await c.read(symptomEpisodesProvider.future);
      c.read(chatControllerProvider.notifier).send('xin chào');
      await pumpEventQueue();
      return (c, vault, agent);
    }

    test('memory on + consent: sends recent_symptoms and saves episode',
        () async {
      final (c, vault, agent) = await run(memory: true, profile: _consented);
      expect(agent.lastContext?.recentSymptoms, ['ho']);
      final eps = await vault.readEpisodes();
      expect(eps, hasLength(2));
      expect(eps.first.terms, ['buồn nôn']);
      expect(eps.first.meds, ['Thuốc Giả C']);
      expect(eps.first.recurring, isTrue);
      final msgs = c.read(chatControllerProvider).valueOrNull!;
      expect(msgs.last.savedEpisodeAt, isNotNull);

      await c.read(chatControllerProvider.notifier).forgetSymptom(msgs.last);
      expect(await vault.readEpisodes(), hasLength(1));
      expect(c.read(chatControllerProvider).valueOrNull!.last.savedEpisodeAt,
          isNull);
    });

    test('memory off: nothing saved, nothing sent', () async {
      final (c, vault, agent) = await run(memory: false, profile: _consented);
      expect(agent.lastContext, isNull);
      expect(await vault.readEpisodes(), hasLength(1));
      expect(c.read(chatControllerProvider).valueOrNull!.last.symptom, noted);
      expect(c.read(chatControllerProvider).valueOrNull!.last.savedEpisodeAt,
          isNull);
    });

    test('memory on but no profile consent: nothing saved, nothing sent',
        () async {
      final (_, vault, agent) =
          await run(memory: true, profile: const HealthProfile());
      expect(agent.lastContext, isNull);
      expect(await vault.readEpisodes(), hasLength(1));
    });
  });
}
