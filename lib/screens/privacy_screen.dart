import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/neu.dart';

import 'app_lock_section.dart';
import 'assistant_memory_screen.dart';
import 'cloud_ai_consent_section.dart';

/// Quyền riêng tư: tóm tắt chính sách, thu hồi đồng ý, xoá tài khoản + dữ liệu.
class PrivacyScreen extends ConsumerWidget {
  const PrivacyScreen({super.key});

  Future<void> _revoke(BuildContext context, WidgetRef ref) async {
    final l10n = AppLocalizations.of(context);
    final messenger = ScaffoldMessenger.of(context);
    await ref.read(consentControllerProvider.notifier).revoke();
    messenger
        .showSnackBar(SnackBar(content: Text(l10n.privacyRevokeConsentBody)));
  }

  Future<void> _deleteAccount(BuildContext context, WidgetRef ref) async {
    final l10n = AppLocalizations.of(context);
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);

    final password = await showDialog<String>(
      context: context,
      builder: (dialogContext) => const _DeleteAccountDialog(),
    );
    if (password == null) return;

    try {
      await ref.read(authControllerProvider.notifier).deleteAccount(password);
    } on AppException catch (e) {
      // Sai mật khẩu: giữ nguyên mọi thứ và nói rõ lý do.
      messenger.showSnackBar(SnackBar(content: Text(e.message)));
      return;
    }
    messenger.showSnackBar(SnackBar(content: Text(l10n.deleteAccountDone)));
    // Về Login (route gốc) và xoá sạch stack — tài khoản không còn tồn tại.
    navigator.popUntil((route) => route.isFirst);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final signedIn =
        ref.watch(authControllerProvider).valueOrNull?.isAuthenticated ?? false;

    return Scaffold(
      appBar: AppBar(title: Text(l10n.privacyTitle)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            NeuCard(
              padding: const EdgeInsets.all(20),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.shield_outlined, color: neu.accent),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      l10n.privacySummary,
                      style: Theme.of(context)
                          .textTheme
                          .bodyMedium
                          ?.copyWith(color: neu.textStrong, height: 1.4),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            const AppLockSection(),
            const SizedBox(height: 12),
            const CloudAiConsentSection(),
            const SizedBox(height: 12),
            NeuCard(
              child: InkWell(
                borderRadius: BorderRadius.circular(16),
                onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(
                  builder: (_) => const AssistantMemoryScreen(),
                )),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      Icon(Icons.psychology_outlined, color: neu.accent),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              l10n.memHTitle,
                              style: Theme.of(context)
                                  .textTheme
                                  .titleMedium
                                  ?.copyWith(fontWeight: FontWeight.w700),
                            ),
                            Text(
                              l10n.memHSubtitle,
                              style: Theme.of(context)
                                  .textTheme
                                  .bodySmall
                                  ?.copyWith(color: neu.textMuted),
                            ),
                          ],
                        ),
                      ),
                      const Icon(Icons.chevron_right),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 20),
            NeuCard(
              child: OutlinedButton.icon(
                icon: const Icon(Icons.undo),
                label: Text(l10n.privacyRevokeConsent),
                onPressed: () => _revoke(context, ref),
              ),
            ),
            if (signedIn) ...[
              const SizedBox(height: 20),
              NeuCard(
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    backgroundColor: Theme.of(context).colorScheme.error,
                    foregroundColor: Theme.of(context).colorScheme.onError,
                  ),
                  icon: const Icon(Icons.delete_forever),
                  label: Text(l10n.privacyDeleteAccount),
                  onPressed: () => _deleteAccount(context, ref),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Hộp thoại xác nhận bằng mật khẩu. Trả mật khẩu, hoặc `null` nếu huỷ.
class _DeleteAccountDialog extends StatefulWidget {
  const _DeleteAccountDialog();

  @override
  State<_DeleteAccountDialog> createState() => _DeleteAccountDialogState();
}

class _DeleteAccountDialogState extends State<_DeleteAccountDialog> {
  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return AlertDialog(
      title: Text(l10n.deleteAccountTitle),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(l10n.deleteAccountBody),
          const SizedBox(height: 12),
          TextField(
            controller: _controller,
            obscureText: true,
            autocorrect: false,
            enableSuggestions: false,
            decoration:
                InputDecoration(labelText: l10n.deleteAccountPasswordLabel),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(l10n.cancel),
        ),
        FilledButton(
          onPressed: () => Navigator.of(context).pop(_controller.text),
          child: Text(l10n.deleteAccountConfirm),
        ),
      ],
    );
  }
}
