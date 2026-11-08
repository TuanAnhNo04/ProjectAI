import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/brand_header.dart';
import 'package:image_processing_app/widgets/credential_field.dart';

class RegisterScreen extends ConsumerStatefulWidget {
  const RegisterScreen({super.key});

  @override
  ConsumerState<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends ConsumerState<RegisterScreen> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();

  /// Hiện/ẩn mật khẩu.
  bool _obscurePassword = true;

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  Future<void> register() async {
    final l10n = AppLocalizations.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);

    // Validate trước khi gọi mạng — không gửi mật khẩu/email đi vô ích.
    final issue = checkCredentials(
      username: _usernameController.text,
      password: _passwordController.text,
      email: _emailController.text,
      requireEmail: true,
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

    final ok = await ref.read(authControllerProvider.notifier).register(
          _usernameController.text.trim(),
          _passwordController.text,
          _emailController.text.trim(),
        );

    if (!mounted) return;

    if (ok) {
      messenger.showSnackBar(
        SnackBar(content: Text(l10n.processSuccess)),
      );
      // Quay về màn hình đăng nhập. `Navigator.pop` trả `void` ở Flutter
      // 3.19 nên không được `await` (sẽ báo use_of_void_result).
      navigator.pop();
      return;
    }

    final error = ref.read(authControllerProvider).error;
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          error is AppException ? error.message : l10n.signUpFailed,
        ),
        backgroundColor: Colors.red,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return Scaffold(
      appBar: AppBar(
        backgroundColor: NeuTheme.of(context).background,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  BrandHeader(title: l10n.signUpTitle),
                  const SizedBox(height: 32),
                  CredentialField(
                    controller: _usernameController,
                    label: l10n.usernameLabel,
                    icon: Icons.person_outline,
                    autofillHint: AutofillHints.newUsername,
                  ),
                  const SizedBox(height: 16),
                  CredentialField(
                    controller: _emailController,
                    label: l10n.emailLabel,
                    icon: Icons.email_outlined,
                    autofillHint: AutofillHints.email,
                    // Bàn phím có @ và . để giảm lỗi gõ.
                    keyboardType: TextInputType.emailAddress,
                  ),
                  const SizedBox(height: 16),
                  CredentialField(
                    controller: _passwordController,
                    label: l10n.passwordLabel,
                    icon: Icons.lock_outline,
                    obscureText: _obscurePassword,
                    autofillHint: AutofillHints.newPassword,
                    onToggleObscure: () =>
                        setState(() => _obscurePassword = !_obscurePassword),
                  ),
                  const SizedBox(height: 24),
                  _buildRegisterButton(),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildRegisterButton() {
    final l10n = AppLocalizations.of(context);
    final scheme = Theme.of(context).colorScheme;
    final neu = NeuTheme.of(context);
    final isLoading = ref.watch(authControllerProvider).isLoading;
    return ElevatedButton(
      onPressed: isLoading ? null : register,
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
          : Text(l10n.signUp, style: const TextStyle(fontSize: 18)),
    );
  }
}
