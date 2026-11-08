import 'package:flutter/material.dart';

import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Ô nhập tên đăng nhập / email / mật khẩu dùng chung cho Đăng nhập và Đăng ký.
///
/// Kiểu dáng đến từ `inputDecorationTheme` của app (không tự đặt màu ở đây).
class CredentialField extends StatelessWidget {
  const CredentialField({
    super.key,
    required this.controller,
    required this.label,
    required this.icon,
    this.obscureText = false,
    this.autofillHint,
    this.keyboardType,
    this.onToggleObscure,
  });

  final TextEditingController controller;
  final String label;
  final IconData icon;
  final bool obscureText;
  final String? autofillHint;
  final TextInputType? keyboardType;

  /// Có thì hiện nút 👁 hiện/ẩn mật khẩu.
  final VoidCallback? onToggleObscure;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final accent = NeuTheme.of(context).accent;
    return TextField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      // Cho trình quản lý mật khẩu của hệ điều hành đề xuất/ghi nhớ — không phải
      // tự lưu (và tự lưu là sai về mặt bảo mật).
      autofillHints: autofillHint == null ? null : [autofillHint!],
      textInputAction:
          obscureText ? TextInputAction.done : TextInputAction.next,
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, color: accent),
        suffixIcon: onToggleObscure == null
            ? null
            : IconButton(
                icon: Icon(
                  obscureText ? Icons.visibility_off : Icons.visibility,
                ),
                tooltip: obscureText ? l10n.showPassword : l10n.hidePassword,
                onPressed: onToggleObscure,
              ),
      ),
    );
  }
}
