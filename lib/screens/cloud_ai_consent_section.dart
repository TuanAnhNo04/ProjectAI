import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Mục "AI đám mây" trong màn Quyền riêng tư — đồng ý RIÊNG (có mốc thời gian +
/// phiên bản chính sách), tách khỏi đồng ý hồ sơ sức khoẻ. Mặc định TẮT.
class CloudAiConsentSection extends ConsumerWidget {
  const CloudAiConsentSection({super.key});

  static String _fmt(DateTime t) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(t.day)}/${two(t.month)}/${t.year} ${two(t.hour)}:${two(t.minute)}';
  }

  Future<void> _onToggle(BuildContext context, WidgetRef ref, bool on) async {
    final ctrl = ref.read(cloudAiConsentControllerProvider.notifier);
    if (!on) {
      await ctrl.revoke();
      return;
    }
    final l10n = AppLocalizations.of(context);
    final agreed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l10n.cloudAiDialogTitle),
        content: SingleChildScrollView(child: Text(l10n.cloudAiDialogBody)),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l10n.cloudAiDecline),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: Text(l10n.cloudAiAgree),
          ),
        ],
      ),
    );
    if (agreed == true) await ctrl.grant();
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final l10n = AppLocalizations.of(context);
    final async = ref.watch(cloudAiConsentControllerProvider);
    final consent = async.valueOrNull;

    return NeuCard(
      child: Row(
        children: [
          Icon(Icons.cloud_outlined, color: neu.accent),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  l10n.cloudAiTitle,
                  style: theme.textTheme.titleMedium
                      ?.copyWith(fontWeight: FontWeight.w700),
                ),
                Text(
                  l10n.cloudAiSubtitle,
                  style:
                      theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
                ),
                const SizedBox(height: 4),
                Text(
                  l10n.modeGConsentModes,
                  style:
                      theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
                ),
                if (consent != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      l10n.cloudAiGrantedAt(_fmt(consent.grantedAt)),
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: neu.accent,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          Switch(
            value: consent != null,
            onChanged:
                async.isLoading ? null : (v) => _onToggle(context, ref, v),
          ),
        ],
      ),
    );
  }
}
