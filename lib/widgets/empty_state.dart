import 'package:flutter/material.dart';

import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Trạng thái rỗng dùng chung: icon lớn trong vòng accentSoft + tiêu đề + nút.
class AppEmptyState extends StatelessWidget {
  const AppEmptyState({
    super.key,
    required this.icon,
    required this.title,
    this.message,
    this.actionLabel,
    this.onAction,
  });

  final IconData icon;
  final String title;
  final String? message;
  final String? actionLabel;
  final VoidCallback? onAction;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: NeuCard(
          padding: const EdgeInsets.fromLTRB(24, 28, 24, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 112,
                height: 112,
                decoration: BoxDecoration(
                    shape: BoxShape.circle, color: neu.accentSoft),
                child: Icon(icon, size: 52, color: neu.accent),
              ),
              const SizedBox(height: 18),
              Text(
                title,
                textAlign: TextAlign.center,
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w800,
                  color: neu.textStrong,
                ),
              ),
              if (message != null && message!.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text(
                  message!,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(color: neu.textMuted),
                ),
              ],
              if (actionLabel != null && onAction != null) ...[
                const SizedBox(height: 20),
                FilledButton.icon(
                  onPressed: onAction,
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(AppTheme.minTouchTarget),
                    shape: const StadiumBorder(),
                  ),
                  icon: const Icon(Icons.document_scanner_outlined),
                  label: Text(actionLabel!),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
