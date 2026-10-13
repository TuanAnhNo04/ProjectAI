import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'register_screen.dart';
import 'process_image_screen.dart';

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
  void initState() {
    super.initState();
    // Khôi phục token đã lưu (Day 11) sau khi khung hình đầu tiên đã dựng,
    // để mở app không bị một vòng loading.
    Future.microtask(
      () => ref.read(authControllerProvider.notifier).restoreSession(),
    );
  }

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  Future<void> login() async {
    final l10n = AppLocalizations.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);

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

    if (ok) {
      await navigator.push(
        MaterialPageRoute(builder: (context) => const ImageProcessingScreen()),
      );
      return;
    }

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
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purpleAccent, Colors.blueAccent],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        // SafeArea: gradient tràn ra mép màn hình là đẹp, nhưng **nội dung**
        // (nút, ô nhập) thì không được bị che bởi notch/rìa.
        child: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(32.0),
              child: SingleChildScrollView(
                child: Card(
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  elevation: 10,
                  child: Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const SizedBox(height: 20),
                        _buildLogo(),
                        const SizedBox(height: 40),
                        _buildTextField(
                          controller: _usernameController,
                          label: l10n.usernameLabel,
                          icon: Icons.person,
                          l10n: l10n,
                          autofillHint: AutofillHints.username,
                        ),
                        const SizedBox(height: 20),
                        _buildTextField(
                          controller: _passwordController,
                          label: l10n.passwordLabel,
                          icon: Icons.lock,
                          l10n: l10n,
                          obscureText: _obscurePassword,
                          autofillHint: AutofillHints.password,
                          onToggleObscure: () => setState(
                              () => _obscurePassword = !_obscurePassword),
                        ),
                        const SizedBox(height: 40),
                        _buildLoginButton(),
                        const SizedBox(height: 20),
                        TextButton(
                          onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => const RegisterScreen(),
                              ),
                            );
                          },
                          child: Text(
                            l10n.signUpLink,
                            style: const TextStyle(
                              color: Colors.blue,
                              fontSize: 16,
                              decoration: TextDecoration.underline,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
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
            color: Colors.black.withOpacity(0.2),
            offset: const Offset(4, 4),
            blurRadius: 15,
          ),
          const BoxShadow(
            color: Colors.white,
            offset: Offset(-4, -4),
            blurRadius: 15,
          ),
        ],
      ),
      child: const Padding(
        padding: EdgeInsets.all(30.0),
        child: Icon(
          Icons.lock_outline,
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
    VoidCallback? onToggleObscure,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      // Cho trình quản lý mật khẩu của hệ điều hành đề xuất/ghi nhớ — không
      // phải tự lưu (và tự lưu là sai về mặt bảo mật).
      autofillHints: autofillHint == null ? null : [autofillHint],
      textInputAction:
          obscureText ? TextInputAction.done : TextInputAction.next,
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, color: Colors.purple),
        suffixIcon: onToggleObscure == null
            ? null
            : IconButton(
                icon: Icon(
                  obscureText ? Icons.visibility_off : Icons.visibility,
                  color: Colors.purple,
                ),
                tooltip: obscureText ? l10n.showPassword : l10n.hidePassword,
                onPressed: onToggleObscure,
              ),
        filled: true,
        fillColor: Colors.white,
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
        color: isFocused ? Colors.purple : Colors.transparent,
        width: 2,
      ),
    );
  }

  Widget _buildLoginButton() {
    final l10n = AppLocalizations.of(context);
    final isLoading = ref.watch(authControllerProvider).isLoading;
    return ElevatedButton(
      onPressed: isLoading ? null : login,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.purple,
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
              l10n.signIn,
              style: const TextStyle(fontSize: 18, color: Colors.white),
            ),
    );
  }
}
