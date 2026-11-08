import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/widgets/app_lock_gate.dart';
import 'app_shell.dart';
import 'login_screen.dart';

/// Chọn màn gốc theo **trạng thái đăng nhập** thay vì điều hướng bằng tay.
///
/// - Đăng nhập thành công → state có phiên → hiện [AppShell].
/// - Đăng xuất / hết phiên (401) / xoá tài khoản → state về `null` → tự hiện [LoginScreen].
///
/// Nhờ vậy không còn `popUntil(isFirst)` rải rác (dễ sai khi stack thay đổi): màn gốc
/// luôn khớp với phiên hiện tại.
class AuthGate extends ConsumerStatefulWidget {
  const AuthGate({super.key});

  @override
  ConsumerState<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends ConsumerState<AuthGate> {
  /// Đang khôi phục token đã lưu — hiện spinner để không nháy màn Login khi người dùng
  /// thật ra đã có phiên.
  bool _restoring = true;

  @override
  void initState() {
    super.initState();
    Future.microtask(() async {
      await ref.read(authControllerProvider.notifier).restoreSession();
      if (mounted) setState(() => _restoring = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_restoring) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    final signedIn =
        ref.watch(authControllerProvider).valueOrNull?.isAuthenticated ?? false;
    return signedIn
        ? const AppLockGate(child: AppShell())
        : const LoginScreen();
  }
}
