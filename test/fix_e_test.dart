// Remediation A-01 / A-03 / A-04 — fake data only.
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/app_lock_store.dart';
import 'package:image_processing_app/data/local/assistant_memory_store.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/data/local/health_database.dart';
import 'package:image_processing_app/data/local/last_user_store.dart';
import 'package:image_processing_app/data/local/notification_service.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/user.dart';
import 'package:image_processing_app/domain/ports/auth_repository.dart';

class _Clock {
  DateTime now = DateTime(2026, 10, 29, 12, 0);
  DateTime call() => now;
}

class _ThrowingStorage extends FlutterSecureStorage {
  const _ThrowingStorage();
  @override
  Future<String?> read({
    required String key,
    IOSOptions? iOptions,
    AndroidOptions? aOptions,
    LinuxOptions? lOptions,
    WebOptions? webOptions,
    MacOsOptions? mOptions,
    WindowsOptions? wOptions,
  }) async =>
      throw Exception('storage broken');
}

class _FakeAuth implements AuthRepository {
  _FakeAuth(this.id);
  int id;
  @override
  Future<AuthSession> login(String u, String p) async =>
      AuthSession(user: User(id: id, username: u), token: 't');
  @override
  Future<AuthSession> register(String u, String p, String e) => login(u, p);
  @override
  Future<void> logout() async {}
  @override
  Future<bool> refreshSession() async => false;
  @override
  Future<void> deleteAccount(String password) async {}
  @override
  Future<User?> currentUser() async => null;
  @override
  Future<String?> currentToken() async => 't';
}

class _FakeDb implements HealthDatabase {
  int wiped = 0;
  @override
  Future<void> deleteAll() async => wiped++;
  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

class _FakeNotif implements NotificationService {
  int cancelled = 0;
  @override
  Future<void> cancelAll() async => cancelled++;
  @override
  dynamic noSuchMethod(Invocation i) => super.noSuchMethod(i);
}

typedef _Setup = ({
  ProviderContainer c,
  _FakeDb db,
  _FakeNotif n,
  InMemoryLastUserStore u,
  InMemoryHealthVault v
});

_Setup _setup(String? remembered, _FakeAuth auth) {
  final db = _FakeDb();
  final n = _FakeNotif();
  final u = InMemoryLastUserStore(remembered);
  final v = InMemoryHealthVault();
  final c = ProviderContainer(overrides: [
    authRepositoryProvider.overrideWithValue(auth),
    healthDatabaseProvider.overrideWithValue(db),
    notificationServiceProvider.overrideWithValue(n),
    lastUserStoreProvider.overrideWithValue(u),
    cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
    appLockStoreProvider.overrideWithValue(InMemoryAppLockStore()),
    healthVaultProvider.overrideWithValue(v),
    assistantMemoryStoreProvider
        .overrideWithValue(InMemoryAssistantMemoryStore(value: true)),
  ]);
  addTearDown(c.dispose);
  return (c: c, db: db, n: n, u: u, v: v);
}

void main() {
  group('PinAttemptLimiter', () {
    test('locks 30s after 5 failures, escalates, caps, resets', () {
      final c = _Clock();
      final l = PinAttemptLimiter(clock: c.call);
      for (var i = 0; i < 4; i++) {
        l.recordFailure();
      }
      expect(l.isLocked, isFalse);
      l.recordFailure();
      expect(l.remaining, const Duration(seconds: 30));
      c.now = c.now.add(const Duration(seconds: 31));
      expect(l.isLocked, isFalse);
      l.recordFailure();
      expect(l.remaining, const Duration(minutes: 1));
      c.now = c.now.add(const Duration(minutes: 2));
      l.recordFailure();
      expect(l.remaining, const Duration(minutes: 5));
      c.now = c.now.add(const Duration(minutes: 6));
      l.recordFailure();
      expect(l.remaining, const Duration(minutes: 15));
      c.now = c.now.add(const Duration(minutes: 16));
      l.recordFailure();
      expect(l.remaining, const Duration(minutes: 15));
      l.recordSuccess();
      expect(l.isLocked, isFalse);
      expect(l.failures, 0);
    });

    test('persists via JSON and corrupt data fails safe', () {
      final c = _Clock();
      final l = PinAttemptLimiter(clock: c.call);
      for (var i = 0; i < 5; i++) {
        l.recordFailure();
      }
      final back = PinAttemptLimiter.fromJson(l.toJson(), clock: c.call);
      expect(back.remaining, const Duration(seconds: 30));
      expect(PinAttemptLimiter.fromJson('garbage', clock: c.call).isLocked,
          isTrue);
    });

    test('store refuses correct PIN while locked', () async {
      final c = _Clock();
      final s = InMemoryAppLockStore(clock: c.call);
      await s.setPin('1234');
      for (var i = 0; i < 5; i++) {
        expect(await s.verifyPin('0000'), isFalse);
      }
      expect(await s.verifyPin('1234'), isFalse);
      c.now = c.now.add(const Duration(seconds: 31));
      expect(await s.verifyPin('1234'), isTrue);
    });
  });

  group('fail-closed', () {
    test('unreadable secure storage is not reported as noPin', () async {
      final s = SecureAppLockStore(storage: const _ThrowingStorage());
      expect(await s.pinState(), PinState.unreadable);
      expect(await s.hasPin(), isFalse);
      expect(await s.verifyPin('1234'), isFalse);
    });

    test('in-memory tri-state', () async {
      final s = InMemoryAppLockStore();
      expect(await s.pinState(), PinState.noPin);
      await s.setPin('1234');
      expect(await s.pinState(), PinState.hasPin);
      s.unreadable = true;
      expect(await s.pinState(), PinState.unreadable);
    });
  });

  group('shared device user switch', () {
    test('different user wipes local data and remembers new id', () async {
      final t = _setup('1', _FakeAuth(2));
      await t.c.read(authControllerProvider.notifier).login('fake_b', 'x');
      expect(t.db.wiped, 1);
      expect(t.n.cancelled, 1);
      expect(t.u.value, '2');
    });

    test('different user wipes the vault; deleteAccount wipes it too',
        () async {
      final t = _setup('1', _FakeAuth(2));
      await t.v.writeProfile(const HealthProfile(conditions: ['Bệnh giả']));
      await t.v.appendEpisode(
          SymptomEpisode(at: DateTime(2026, 1, 1), terms: const ['ho']));
      await t.c.read(authControllerProvider.notifier).login('fake_b', 'x');
      expect(await t.v.readProfile(), isNull);
      expect(await t.v.readEpisodes(), isEmpty);
      expect(await t.c.read(assistantMemoryStoreProvider).read(), isFalse);

      final u = _setup('2', _FakeAuth(2));
      await u.v.writeProfile(const HealthProfile(conditions: ['Bệnh giả']));
      await u.c.read(authControllerProvider.notifier).deleteAccount('pw');
      expect(await u.v.readProfile(), isNull);
    });

    test('same user keeps everything', () async {
      final t = _setup('2', _FakeAuth(2));
      await t.v.writeProfile(const HealthProfile(conditions: ['Bệnh giả']));
      await t.c.read(authControllerProvider.notifier).login('fake_b', 'x');
      expect(await t.v.readProfile(), isNotNull);
      expect(t.db.wiped, 0);
      expect(t.n.cancelled, 0);
    });

    test('first remembered id does not wipe', () async {
      final t = _setup(null, _FakeAuth(7));
      await t.c.read(authControllerProvider.notifier).login('fake_b', 'x');
      expect(t.db.wiped, 0);
      expect(t.u.value, '7');
    });

    test('deleteAccount clears PIN', () async {
      final t = _setup('2', _FakeAuth(2));
      final lock = t.c.read(appLockStoreProvider);
      await lock.setPin('1234');
      await t.c.read(authControllerProvider.notifier).deleteAccount('pw');
      expect(await lock.pinState(), PinState.noPin);
    });
  });

  group('dose logic', () {
    const late2330 =
        MedicationReminder(medicineName: 'Fake', hour: 23, minute: 30);
    const at13 = MedicationReminder(medicineName: 'Fake', hour: 13, minute: 0);

    test('23:30 dose logged 00:45 next day is late', () {
      expect(doseStatusForTaken(late2330, DateTime(2026, 10, 30, 0, 45)),
          kDoseLate);
      expect(doseStatusForTaken(late2330, DateTime(2026, 10, 30, 0, 20)),
          kDoseTaken);
    });

    test('dose due later today logged early is taken', () {
      expect(
          doseStatusForTaken(at13, DateTime(2026, 10, 29, 8, 0)), kDoseTaken);
      expect(
          doseStatusForTaken(at13, DateTime(2026, 10, 29, 15, 0)), kDoseLate);
    });

    test('pillsAfterDose clamps at 0', () {
      expect(pillsAfterDose(5, 2), 3);
      expect(pillsAfterDose(1, 2), 0);
      expect(pillsAfterDose(0, 1), 0);
    });

    test('duplicate taken/late is detected via takenIdsToday', () {
      final now = DateTime(2026, 10, 29, 15);
      final logs = [
        AdherenceRecord(reminderId: 1, takenAt: now, status: kDoseLate),
        AdherenceRecord(reminderId: 2, takenAt: now, status: kDoseSkipped),
      ];
      expect(takenIdsToday(logs, now).contains(1), isTrue);
      expect(takenIdsToday(logs, now).contains(2), isFalse);
    });
  });
}
