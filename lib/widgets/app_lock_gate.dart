import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/app_lock_store.dart';
import 'package:image_processing_app/screens/lock_screen.dart';

/// Bọc nội dung sau đăng nhập ([AppShell]) để khoá bằng PIN khi cần (Day 33).
///
/// - Chỉ thực sự khoá nếu người dùng đã đặt PIN ([AppLockStore.hasPin]) — mặc
///   định KHÔNG bật, không ảnh hưởng ai chưa dùng tính năng này.
/// - Khoá lại khi app quay về foreground sau khi đã rời đi (nền/khoá máy) quá
///   [_lockAfter] — rời đi rất ngắn (vd chuyển qua app khác xem thông báo rồi
///   quay lại ngay) thì không bắt nhập lại PIN để không gây khó chịu.
class AppLockGate extends ConsumerStatefulWidget {
  const AppLockGate({super.key, required this.child});

  final Widget child;

  @override
  ConsumerState<AppLockGate> createState() => _AppLockGateState();
}

class _AppLockGateState extends ConsumerState<AppLockGate>
    with WidgetsBindingObserver {
  static const _lockAfter = Duration(seconds: 30);

  bool _ready = false;
  bool _locked = false;
  bool _unreadable = false;
  DateTime? _pausedAt;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkInitialLock();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  Future<void> _checkInitialLock() async {
    final state = await ref.read(appLockStoreProvider).pinState();
    if (!mounted) return;
    setState(() {
      _locked = state != PinState.noPin;
      _unreadable = state == PinState.unreadable;
      _ready = true;
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive) {
      _pausedAt = DateTime.now();
      return;
    }
    if (state != AppLifecycleState.resumed) return;
    final pausedAt = _pausedAt;
    _pausedAt = null;
    if (pausedAt == null) return;
    if (DateTime.now().difference(pausedAt) < _lockAfter) return;
    _relock();
  }

  Future<void> _relock() async {
    final state = await ref.read(appLockStoreProvider).pinState();
    if (mounted && state != PinState.noPin) {
      setState(() {
        _locked = true;
        _unreadable = state == PinState.unreadable;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) return const SizedBox.shrink();
    if (_locked) {
      return LockScreen(
        key: ValueKey(_unreadable),
        unreadable: _unreadable,
        onUnlocked: () => setState(() => _locked = false),
      );
    }
    return widget.child;
  }
}
