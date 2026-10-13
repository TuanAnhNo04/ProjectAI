import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

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
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF4A148C), Color(0xFF880E4F)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: SingleChildScrollView(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 50),
                    _buildLogo(),
                    const SizedBox(height: 30),
                    const Text(
                      'Đăng ký tài khoản',
                      style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 40),
                    _buildTextField(
                      controller: _usernameController,
                      label: l10n.usernameLabel,
                      icon: Icons.person,
                      l10n: l10n,
                      autofillHint: AutofillHints.newUsername,
                    ),
                    const SizedBox(height: 20),
                    _buildTextField(
                      controller: _emailController,
                      label: l10n.emailLabel,
                      icon: Icons.email,
                      l10n: l10n,
                      autofillHint: AutofillHints.email,
                      // Bàn phím có @ và . để giảm lỗi gõ.
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 20),
                    _buildTextField(
                      controller: _passwordController,
                      label: l10n.passwordLabel,
                      icon: Icons.lock,
                      l10n: l10n,
                      obscureText: _obscurePassword,
                      autofillHint: AutofillHints.newPassword,
                      onToggleObscure: () =>
                          setState(() => _obscurePassword = !_obscurePassword),
                    ),
                    const SizedBox(height: 40),
                    _buildRegisterButton(),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLogo() {
    return Container(
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade500,
            offset: const Offset(4, 4),
            blurRadius: 15,
            spreadRadius: 1,
          ),
          const BoxShadow(
            color: Colors.white,
            offset: Offset(-4, -4),
            blurRadius: 15,
            spreadRadius: 1,
          ),
        ],
      ),
      child: const Padding(
        padding: EdgeInsets.all(30.0),
        child: Icon(
          Icons.person_add,
          size: 60,
          color: Colors.purple,
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required AppLocalizations l10n,
    bool obscureText = false,
    String? autofillHint,
    TextInputType? keyboardType,
    VoidCallback? onToggleObscure,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      autofillHints: autofillHint == null ? null : [autofillHint],
      textInputAction:
          obscureText ? TextInputAction.done : TextInputAction.next,
      style: const TextStyle(color: Colors.white),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: Colors.white70),
        prefixIcon: Icon(icon, color: Colors.white),
        suffixIcon: onToggleObscure == null
            ? null
            : IconButton(
                icon: Icon(
                  obscureText ? Icons.visibility_off : Icons.visibility,
                  color: Colors.white,
                ),
                tooltip: obscureText ? l10n.showPassword : l10n.hidePassword,
                onPressed: onToggleObscure,
              ),
        filled: true,
        fillColor: Colors.white.withOpacity(0.1),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(vertical: 18.0),
        enabledBorder: _neumorphicBorder(),
        focusedBorder: _neumorphicBorder(isFocused: true),
      ),
    );
  }

  OutlineInputBorder _neumorphicBorder({bool isFocused = false}) {
    return OutlineInputBorder(
      borderRadius: BorderRadius.circular(30),
      borderSide: BorderSide(
        color: isFocused ? Colors.white : Colors.transparent,
        width: 2,
      ),
    );
  }

  Widget _buildRegisterButton() {
    final l10n = AppLocalizations.of(context);
    final isLoading = ref.watch(authControllerProvider).isLoading;
    return ElevatedButton(
      onPressed: isLoading ? null : register,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.pinkAccent,
        padding: const EdgeInsets.symmetric(horizontal: 80, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
        ),
        elevation: 10,
      ),
      child: isLoading
          ? const SizedBox(
              height: 22,
              width: 22,
              child: CircularProgressIndicator(
                strokeWidth: 2.5,
                color: Colors.white,
              ),
            )
          : Text(
              l10n.signUp,
              style: const TextStyle(fontSize: 18, color: Colors.white),
            ),
    );
  }
}
