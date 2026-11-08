import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/brand_header.dart';
import 'package:image_processing_app/widgets/credential_field.dart';
import 'register_screen.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  /// Hiện/ẩn mật khẩu — `obscureText` cần rebuild nên phải là biến state.
  bool _obscurePassword = true;

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> login() async {
    final l10n = AppLocalizations.of(context);
    final messenger = ScaffoldMessenger.of(context);

    // Validate **trước khi gọi mạng**: không gửi mật khẩu đi vô ích khi biết
    // chắc là thiếu/sai.
    final issue = checkCredentials(
      username: _usernameController.text,
      password: _passwordController.text,
    );
    if (issue != null) {
      messenger.showSnackBar(
        SnackBar(
          content: Text(issue.localize(l10n)),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    final ok = await ref.read(authControllerProvider.notifier).login(
          _usernameController.text.trim(),
          _passwordController.text,
        );

    // `if (!mounted) return;` sau mỗi await: tránh dùng context của màn hình
    // đã bị đóng (use_build_context_synchronously).
    if (!mounted) return;

    // Thành công: `AuthGate` thấy phiên mới và tự chuyển sang AppShell — không điều
    // hướng bằng tay ở đây.
    if (ok) return;

    final error = ref.read(authControllerProvider).error;
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          error is AppException ? error.message : l10n.signInFailed,
        ),
        backgroundColor: Colors.red,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return Scaffold(
      // SafeArea: nội dung (nút, ô nhập) không được bị che bởi notch/rìa.
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  BrandHeader(title: l10n.appTitle),
                  const SizedBox(height: 32),
                  CredentialField(
                    controller: _usernameController,
                    label: l10n.usernameLabel,
                    icon: Icons.person_outline,
                    autofillHint: AutofillHints.username,
                  ),
                  const SizedBox(height: 16),
                  CredentialField(
                    controller: _passwordController,
                    label: l10n.passwordLabel,
                    icon: Icons.lock_outline,
                    obscureText: _obscurePassword,
                    autofillHint: AutofillHints.password,
                    onToggleObscure: () =>
                        setState(() => _obscurePassword = !_obscurePassword),
                  ),
                  const SizedBox(height: 24),
                  _buildLoginButton(),
                  const SizedBox(height: 8),
                  TextButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const RegisterScreen(),
                        ),
                      );
                    },
                    child: Text(l10n.signUpLink),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLoginButton() {
    final l10n = AppLocalizations.of(context);
    final scheme = Theme.of(context).colorScheme;
    final neu = NeuTheme.of(context);
    final isLoading = ref.watch(authControllerProvider).isLoading;
    return ElevatedButton(
      onPressed: isLoading ? null : login,
      style: ElevatedButton.styleFrom(
        backgroundColor: neu.accent,
        foregroundColor: scheme.onPrimary,
        minimumSize: const Size.fromHeight(52),
        shape: const StadiumBorder(),
      ),
      child: isLoading
          ? SizedBox(
              height: 22,
              width: 22,
              child: CircularProgressIndicator(
                strokeWidth: 2.5,
                color: scheme.onPrimary,
              ),
            )
          : Text(l10n.signIn, style: const TextStyle(fontSize: 18)),
    );
  }
}
