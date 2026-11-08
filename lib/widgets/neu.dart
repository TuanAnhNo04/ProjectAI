import 'package:flutter/material.dart';

import 'package:image_processing_app/core/theme/app_theme.dart';

/// Thẻ "nổi" kiểu neumorphism: cùng màu nền, tách khỏi nền bằng bóng sáng + bóng tối.
///
/// - Có [onTap] thì tự thành vùng bấm (có hiệu ứng gợn) và được đọc như một nút bởi
///   TalkBack/VoiceOver.
/// - [accent] tô thẻ bằng màu nhấn (dùng cho thẻ đang chọn / hành động chính).
class NeuCard extends StatelessWidget {
  const NeuCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.radius = 20,
    this.onTap,
    this.accent = false,
    this.color,
    this.semanticLabel,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  final VoidCallback? onTap;
  final bool accent;

  /// Màu nền riêng (vd. nền nhạt của màu nhấn); mặc định là surface.
  final Color? color;

  /// Nhãn đọc cho trình đọc màn hình khi thẻ bấm được.
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final borderRadius = BorderRadius.circular(radius);
    Widget content = Padding(padding: padding, child: child);
    if (onTap != null) {
      content = Material(
        type: MaterialType.transparency,
        child: InkWell(
          borderRadius: borderRadius,
          onTap: onTap,
          child: content,
        ),
      );
    }
    final card = DecoratedBox(
      decoration: BoxDecoration(
        color: accent ? neu.accent : (color ?? neu.surface),
        borderRadius: borderRadius,
        boxShadow: neu.raised(),
      ),
      child: content,
    );
    if (onTap == null) return card;
    return Semantics(button: true, label: semanticLabel, child: card);
  }
}

/// Nút tròn nổi (icon) — nút "quay lại", "tìm kiếm", "gửi"… Vùng chạm ≥ 48dp.
class NeuCircleButton extends StatelessWidget {
  const NeuCircleButton({
    super.key,
    required this.icon,
    required this.onPressed,
    required this.tooltip,
    this.size = 48,
    this.filled = false,
  });

  final IconData icon;
  final VoidCallback? onPressed;

  /// Bắt buộc: nút chỉ có icon cần nhãn cho trình đọc màn hình.
  final String tooltip;
  final double size;

  /// `true` = nền màu nhấn, icon trắng (hành động chính).
  final bool filled;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final scheme = Theme.of(context).colorScheme;
    final enabled = onPressed != null;
    return Tooltip(
      message: tooltip,
      child: Semantics(
        button: true,
        enabled: enabled,
        label: tooltip,
        excludeSemantics: true,
        child: GestureDetector(
          onTap: onPressed,
          child: Container(
            width:
                size < AppTheme.minTouchTarget ? AppTheme.minTouchTarget : size,
            height:
                size < AppTheme.minTouchTarget ? AppTheme.minTouchTarget : size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: filled ? neu.accent : neu.surface,
              boxShadow: enabled ? neu.soft() : const [],
            ),
            child: Icon(
              icon,
              color: !enabled
                  ? neu.textMuted.withOpacity(0.5)
                  : (filled ? scheme.onPrimary : neu.accent),
            ),
          ),
        ),
      ),
    );
  }
}

/// Tiêu đề nhóm nội dung + (tuỳ chọn) liên kết "Xem tất cả" bên phải.
class NeuSectionTitle extends StatelessWidget {
  const NeuSectionTitle({
    super.key,
    required this.title,
    this.actionLabel,
    this.onAction,
  });

  final String title;
  final String? actionLabel;
  final VoidCallback? onAction;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      children: [
        Expanded(
          child: Text(
            title,
            style: theme.textTheme.titleMedium
                ?.copyWith(fontSize: 20, fontWeight: FontWeight.w700),
          ),
        ),
        if (actionLabel != null && onAction != null)
          TextButton(onPressed: onAction, child: Text(actionLabel!)),
      ],
    );
  }
}
