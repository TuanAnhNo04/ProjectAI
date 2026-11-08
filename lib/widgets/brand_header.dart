import 'package:flutter/material.dart';

import 'package:image_processing_app/core/theme/app_theme.dart';

/// Logo + tên app ở đầu màn Đăng nhập/Đăng ký (kiểu neumorphism: logo nằm trong
/// vòng tròn nổi). Màu lấy từ [NeuTheme].
class BrandHeader extends StatelessWidget {
  const BrandHeader({super.key, required this.title, this.subtitle});

  final String title;
  final String? subtitle;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        ExcludeSemantics(
          child: Container(
            width: 96,
            height: 96,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: neu.background,
              boxShadow: neu.raised(),
            ),
            child: Icon(Icons.health_and_safety, size: 48, color: neu.accent),
          ),
        ),
        const SizedBox(height: 20),
        Text(
          title,
          textAlign: TextAlign.center,
          style: theme.textTheme.headlineMedium
              ?.copyWith(fontWeight: FontWeight.bold, color: neu.textStrong),
        ),
        if (subtitle != null) ...[
          const SizedBox(height: 4),
          Text(
            subtitle!,
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyLarge?.copyWith(color: neu.textMuted),
          ),
        ],
      ],
    );
  }
}
