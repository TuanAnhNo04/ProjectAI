import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Mục "Khoá ứng dụng bằng mã PIN" trong màn Quyền riêng tư (Day 33).
///
/// Tách riêng khỏi `ConsentStore`/hồ sơ sức khoẻ: đây là một loại dữ liệu khác
/// (bảo vệ THIẾT BỊ, không phải dữ liệu sức khoẻ) nên có công tắc + luồng bật/tắt
/// riêng, đúng tinh thần "tách riêng cho từng loại dữ liệu" của AGENTS.md.
class AppLockSection extends ConsumerStatefulWidget {
  const AppLockSection({super.key});

  @override
  ConsumerState<AppLockSection> createState() => _AppLockSectionState();
}

class _AppLockSectionState extends ConsumerState<AppLockSection> {
  bool? _hasPin;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    final hasPin = await ref.read(appLockStoreProvider).hasPin();
    if (mounted) setState(() => _hasPin = hasPin);
  }

  Future<void> _onToggle(bool enable) async {
    final messenger = ScaffoldMessenger.of(context);
    final l10n = AppLocalizations.of(context);
    if (enable) {
      final pin = await showDialog<String>(
        context: context,
        builder: (_) => const _SetPinDialog(),
      );
      if (pin == null) return;
      await ref.read(appLockStoreProvider).setPin(pin);
      messenger.showSnackBar(SnackBar(content: Text(l10n.lockEnabled)));
    } else {
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (_) => const _VerifyCurrentPinDialog(),
      );
      if (confirmed != true) return;
      await ref.read(appLockStoreProvider).clearPin();
      messenger.showSnackBar(SnackBar(content: Text(l10n.lockDisabled)));
    }
    await _refresh();
  }

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final l10n = AppLocalizations.of(context);
    final hasPin = _hasPin ?? false;

    return NeuCard(
      child: Row(
        children: [
          Icon(Icons.lock_outline, color: neu.accent),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  l10n.lockSectionTitle,
                  style: theme.textTheme.titleMedium
                      ?.copyWith(fontWeight: FontWeight.w700),
                ),
                Text(
                  l10n.lockSectionSubtitle,
                  style:
                      theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
                ),
              ],
            ),
          ),
          Switch(
            value: hasPin,
            onChanged: _hasPin == null ? null : _onToggle,
          ),
        ],
      ),
    );
  }
}

class _SetPinDialog extends StatefulWidget {
  const _SetPinDialog();

  @override
  State<_SetPinDialog> createState() => _SetPinDialogState();
}

class _SetPinDialogState extends State<_SetPinDialog> {
  final _pinCtrl = TextEditingController();
  final _confirmCtrl = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _pinCtrl.dispose();
    _confirmCtrl.dispose();
    super.dispose();
  }

  void _submit(AppLocalizations l10n) {
    final pin = _pinCtrl.text.trim();
    final confirm = _confirmCtrl.text.trim();
    if (pin.length < 4) {
      setState(() => _error = l10n.lockPinTooShort);
      return;
    }
    if (pin != confirm) {
      setState(() => _error = l10n.lockPinMismatch);
      return;
    }
    Navigator.of(context).pop(pin);
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return AlertDialog(
      title: Text(l10n.lockSetNewPinTitle),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: _pinCtrl,
            obscureText: true,
            keyboardType: TextInputType.number,
            maxLength: 6,
            decoration: InputDecoration(
              labelText: l10n.lockPinLabel,
              errorText: _error,
            ),
          ),
          TextField(
            controller: _confirmCtrl,
            obscureText: true,
            keyboardType: TextInputType.number,
            maxLength: 6,
            onSubmitted: (_) => _submit(l10n),
            decoration: InputDecoration(labelText: l10n.lockPinConfirmLabel),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(l10n.cancel),
        ),
        FilledButton(
          onPressed: () => _submit(l10n),
          child: Text(l10n.lockSave),
        ),
      ],
    );
  }
}

class _VerifyCurrentPinDialog extends ConsumerStatefulWidget {
  const _VerifyCurrentPinDialog();

  @override
  ConsumerState<_VerifyCurrentPinDialog> createState() =>
      _VerifyCurrentPinDialogState();
}

class _VerifyCurrentPinDialogState
    extends ConsumerState<_VerifyCurrentPinDialog> {
  final _pinCtrl = TextEditingController();
  String? _error;

  @override
  void dispose() {
    _pinCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit(AppLocalizations l10n) async {
    final ok =
        await ref.read(appLockStoreProvider).verifyPin(_pinCtrl.text.trim());
    if (!mounted) return;
    if (ok) {
      Navigator.of(context).pop(true);
    } else {
      setState(() => _error = l10n.lockPinWrong);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return AlertDialog(
      title: Text(l10n.lockDisableTitle),
      content: TextField(
        controller: _pinCtrl,
        obscureText: true,
        keyboardType: TextInputType.number,
        maxLength: 6,
        autofocus: true,
        onSubmitted: (_) => _submit(l10n),
        decoration: InputDecoration(
          labelText: l10n.lockPinLabel,
          errorText: _error,
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(false),
          child: Text(l10n.cancel),
        ),
        FilledButton(
          onPressed: () => _submit(l10n),
          child: Text(l10n.lockUnlock),
        ),
      ],
    );
  }
}
