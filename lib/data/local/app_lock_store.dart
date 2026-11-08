import 'dart:convert';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Trạng thái khoá PIN: có PIN / chưa đặt / không đọc được kho bí mật.
///
/// [unreadable] phải **fail-closed**: không được coi là "chưa đặt PIN", nếu không
/// kẻ xấu chỉ cần làm hỏng kho là qua được màn khoá.
enum PinState { hasPin, noPin, unreadable }

/// Bộ đếm lần nhập sai PIN — logic thuần, đồng hồ có thể tiêm để test.
///
/// 5 lần sai đầu -> khoá 30 giây; mỗi lần sai tiếp theo (sau khi hết khoá) tăng
/// dần: 1 phút, 5 phút, rồi trần 15 phút. Đúng PIN thì đặt lại.
class PinAttemptLimiter {
  PinAttemptLimiter({
    DateTime Function()? clock,
    this.failures = 0,
    this.lockedUntil,
  }) : _clock = clock ?? DateTime.now;

  factory PinAttemptLimiter.fromJson(
    String? raw, {
    DateTime Function()? clock,
  }) {
    if (raw == null || raw.isEmpty) return PinAttemptLimiter(clock: clock);
    try {
      final m = jsonDecode(raw) as Map<String, dynamic>;
      final until = m['until'] as int?;
      return PinAttemptLimiter(
        clock: clock,
        failures: (m['n'] as int?) ?? 0,
        lockedUntil:
            until == null ? null : DateTime.fromMillisecondsSinceEpoch(until),
      );
    } on Object {
      // Dữ liệu hỏng: giữ trạng thái an toàn (khoá ngắn) thay vì bỏ qua giới hạn.
      return PinAttemptLimiter(
        clock: clock,
        failures: freeAttempts,
        lockedUntil: (clock ?? DateTime.now)().add(lockSteps.first),
      );
    }
  }

  static const int freeAttempts = 5;
  static const List<Duration> lockSteps = [
    Duration(seconds: 30),
    Duration(minutes: 1),
    Duration(minutes: 5),
    Duration(minutes: 15),
  ];

  final DateTime Function() _clock;
  int failures;
  DateTime? lockedUntil;

  Duration get remaining {
    final until = lockedUntil;
    if (until == null) return Duration.zero;
    final left = until.difference(_clock());
    return left.isNegative ? Duration.zero : left;
  }

  bool get isLocked => remaining > Duration.zero;

  void recordFailure() {
    failures++;
    if (failures >= freeAttempts) {
      final step = (failures - freeAttempts).clamp(0, lockSteps.length - 1);
      lockedUntil = _clock().add(lockSteps[step]);
    }
  }

  void recordSuccess() {
    failures = 0;
    lockedUntil = null;
  }

  String toJson() => jsonEncode({
        'n': failures,
        'until': lockedUntil?.millisecondsSinceEpoch,
      });
}

/// Khoá ứng dụng bằng mã PIN cục bộ (Day 33).
///
/// Mô hình đe doạ: đây là lớp bảo vệ **mức thiết bị** (người cầm máy đang mở
/// khoá), không chống kẻ có quyền root/đọc ổ đĩa. Mã PIN lưu trong kho bí mật
/// của hệ điều hành (Keystore/DPAPI) — **cùng vùng lưu trữ với token đăng
/// nhập** — được mã hoá at-rest, không dùng `shared_preferences` (AGENTS.md mục
/// 3). Không thêm gói băm; thay vào đó giới hạn số lần thử ([PinAttemptLimiter]).
abstract interface class AppLockStore {
  /// Có PIN / chưa đặt / không đọc được — quyết định có hiện màn khoá không.
  Future<PinState> pinState();

  /// Tiện lợi: `true` chỉ khi [pinState] là [PinState.hasPin].
  Future<bool> hasPin();

  Future<void> setPin(String pin);

  /// Kiểm tra PIN và ghi nhận lần thử. Đang bị khoá tạm -> luôn `false`.
  Future<bool> verifyPin(String pin);

  /// Thời gian còn lại của lần khoá tạm do nhập sai ([Duration.zero] = không khoá).
  Future<Duration> lockoutRemaining();

  /// Tắt khoá PIN (xoá mã đã lưu và bộ đếm sai).
  Future<void> clearPin();
}

class SecureAppLockStore implements AppLockStore {
  SecureAppLockStore(
      {FlutterSecureStorage? storage, DateTime Function()? clock})
      : _clock = clock,
        _storage = storage ??
            const FlutterSecureStorage(
              aOptions: AndroidOptions(encryptedSharedPreferences: true),
              iOptions: IOSOptions(
                accessibility: KeychainAccessibility.first_unlock,
              ),
            );

  final FlutterSecureStorage _storage;
  final DateTime Function()? _clock;

  static const String _pinKey = 'medassist.lock_pin';
  static const String _attemptsKey = 'medassist.lock_attempts';

  @override
  Future<PinState> pinState() async {
    try {
      final value = await _storage.read(key: _pinKey);
      return value != null && value.isNotEmpty
          ? PinState.hasPin
          : PinState.noPin;
    } catch (_) {
      return PinState.unreadable;
    }
  }

  @override
  Future<bool> hasPin() async => await pinState() == PinState.hasPin;

  @override
  Future<void> setPin(String pin) async {
    try {
      await _storage.write(key: _pinKey, value: pin);
      await _storage.delete(key: _attemptsKey);
    } catch (_) {
      // Không ghi được -> pinState() sẽ phản ánh đúng ở lần đọc sau.
    }
  }

  Future<PinAttemptLimiter> _loadLimiter() async {
    String? raw;
    try {
      raw = await _storage.read(key: _attemptsKey);
    } catch (_) {
      raw = null;
    }
    return PinAttemptLimiter.fromJson(raw, clock: _clock);
  }

  Future<void> _saveLimiter(PinAttemptLimiter l) async {
    try {
      await _storage.write(key: _attemptsKey, value: l.toJson());
    } catch (_) {}
  }

  @override
  Future<Duration> lockoutRemaining() async => (await _loadLimiter()).remaining;

  @override
  Future<bool> verifyPin(String pin) async {
    final limiter = await _loadLimiter();
    if (limiter.isLocked) return false;
    String? stored;
    try {
      stored = await _storage.read(key: _pinKey);
    } catch (_) {
      return false;
    }
    final ok = stored != null && stored == pin;
    if (ok) {
      limiter.recordSuccess();
    } else {
      limiter.recordFailure();
    }
    await _saveLimiter(limiter);
    return ok;
  }

  @override
  Future<void> clearPin() async {
    try {
      await _storage.delete(key: _pinKey);
      await _storage.delete(key: _attemptsKey);
    } catch (_) {
      // Xem ghi chú ở setPin().
    }
  }
}

/// Bản trong RAM — dùng cho test.
class InMemoryAppLockStore implements AppLockStore {
  InMemoryAppLockStore({DateTime Function()? clock})
      : _limiter = PinAttemptLimiter(clock: clock);

  String? _pin;
  final PinAttemptLimiter _limiter;

  /// Test: giả lập kho bí mật không đọc được.
  bool unreadable = false;

  @override
  Future<PinState> pinState() async {
    if (unreadable) return PinState.unreadable;
    return _pin != null && _pin!.isNotEmpty ? PinState.hasPin : PinState.noPin;
  }

  @override
  Future<bool> hasPin() async => await pinState() == PinState.hasPin;

  @override
  Future<void> setPin(String pin) async {
    _pin = pin;
    _limiter.recordSuccess();
  }

  @override
  Future<bool> verifyPin(String pin) async {
    if (unreadable || _limiter.isLocked) return false;
    final ok = _pin == pin;
    ok ? _limiter.recordSuccess() : _limiter.recordFailure();
    return ok;
  }

  @override
  Future<Duration> lockoutRemaining() async => _limiter.remaining;

  @override
  Future<void> clearPin() async {
    _pin = null;
    _limiter.recordSuccess();
    unreadable = false;
  }
}
