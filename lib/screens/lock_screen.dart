import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Màn khoá — che toàn bộ nội dung cho tới khi nhập đúng mã PIN (Day 33).
///
/// Chỉ hiện khi PIN đã đặt hoặc kho bí mật không đọc được ([unreadable], fail-closed);
/// xem `widgets/app_lock_gate.dart`. Nhập sai nhiều lần bị khoá tạm
/// (`PinAttemptLimiter`). Nút "Quên PIN? Đăng xuất" xoá PIN + phiên cục bộ.
class LockScreen extends ConsumerStatefulWidget {
  const LockScreen({
    super.key,
    required this.onUnlocked,
    this.unreadable = false,
  });

  final VoidCallback onUnlocked;

  /// Kho bí mật không đọc được: không có PIN để nhập, chỉ cho đăng xuất.
  final bool unreadable;

  @override
  ConsumerState<LockScreen> createState() => _LockScreenState();
}

class _LockScreenState extends ConsumerState<LockScreen> {
  final _pinCtrl = TextEditingController();
  String? _error;
  bool _checking = false;
  Duration _lockLeft = Duration.zero;
  Timer? _ticker;

  @override
  void initState() {
    super.initState();
    if (!widget.unreadable) _refreshLockout();
  }

  @override
  void dispose() {
    _ticker?.cancel();
    _pinCtrl.dispose();
    super.dispose();
  }

  Future<void> _refreshLockout() async {
    final left = await ref.read(appLockStoreProvider).lockoutRemaining();
    if (!mounted) return;
    setState(() => _lockLeft = left);
    _ticker?.cancel();
    if (left > Duration.zero) {
      _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
        if (!mounted) return;
        final next = _lockLeft - const Duration(seconds: 1);
        if (next <= Duration.zero) {
          _ticker?.cancel();
          setState(() => _lockLeft = Duration.zero);
        } else {
          setState(() => _lockLeft = next);
        }
      });
    }
  }

  Future<void> _submit() async {
    final l10n = AppLocalizations.of(context);
    final pin = _pinCtrl.text.trim();
    if (pin.isEmpty || _lockLeft > Duration.zero) return;
    setState(() {
      _checking = true;
      _error = null;
    });
    final ok = await ref.read(appLockStoreProvider).verifyPin(pin);
    if (!mounted) return;
    if (ok) {
      widget.onUnlocked();
      return;
    }
    _pinCtrl.clear();
    setState(() {
      _checking = false;
      _error = l10n.lockPinWrong;
    });
    await _refreshLockout();
  }

  Future<void> _forgotPin() async {
    // Xoá PIN trước khi đăng xuất (đăng xuất làm cây widget này bị gỡ).
    await ref.read(appLockStoreProvider).clearPin();
    await ref.read(authControllerProvider.notifier).logout();
  }

  String _fmt(Duration d) {
    final m = d.inMinutes;
    final s = d.inSeconds % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final l10n = AppLocalizations.of(context);
    final locked = _lockLeft > Duration.zero;
    final disabled = widget.unreadable || locked;

    return Scaffold(
      backgroundColor: neu.background,
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: neu.accentSoft,
                  ),
                  child: Icon(Icons.lock_outline, size: 34, color: neu.accent),
                ),
                const SizedBox(height: 20),
                Text(
                  l10n.lockEnterPinTitle,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w800,
                    color: neu.textStrong,
                  ),
                ),
                const SizedBox(height: 20),
                if (widget.unreadable)
                  Text(
                    l10n.fixELockUnreadable,
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodyMedium,
                  )
                else ...[
                  TextField(
                    controller: _pinCtrl,
                    autofocus: true,
                    enabled: !disabled,
                    obscureText: true,
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    maxLength: 6,
                    onSubmitted: (_) => _submit(),
                    decoration: InputDecoration(
                      counterText: '',
                      errorText: locked
                          ? l10n.fixELockTooMany(_fmt(_lockLeft))
                          : _error,
                      hintText: '••••',
                    ),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: (_checking || disabled) ? null : _submit,
                      child: Text(l10n.lockUnlock),
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                TextButton(
                  onPressed: _forgotPin,
                  child: Text(l10n.fixELockForgot),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
