import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/chat_controller.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// "Dữ liệu trợ lý nhớ về tôi": minh bạch + quyền kiểm soát. Chỉ hiển thị dữ
/// liệu trong két mã hoá trên máy; không gửi gì đi từ màn này.
class AssistantMemoryScreen extends ConsumerStatefulWidget {
  const AssistantMemoryScreen({super.key});

  @override
  ConsumerState<AssistantMemoryScreen> createState() =>
      _AssistantMemoryScreenState();
}

class _AssistantMemoryScreenState extends ConsumerState<AssistantMemoryScreen> {
  @override
  void initState() {
    super.initState();
    Future.microtask(() async {
      await ref.read(healthProfileControllerProvider.notifier).load();
      await ref.read(symptomEpisodesProvider.notifier).reload();
    });
  }

  static String _fmt(DateTime t) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(t.day)}/${two(t.month)}/${t.year} ${two(t.hour)}:${two(t.minute)}';
  }

  Future<bool> _confirm(String body) async {
    final l10n = AppLocalizations.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        content: Text(body),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text(l10n.cancel),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: Text(l10n.delete),
          ),
        ],
      ),
    );
    return ok == true;
  }

  void _toast(String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  Future<void> _clearLog() async {
    final l10n = AppLocalizations.of(context);
    if (!await _confirm(l10n.memHClearLogConfirm)) return;
    try {
      await ref.read(symptomEpisodesProvider.notifier).clear();
      if (mounted) _toast(l10n.memHCleared);
    } on Object {
      if (mounted) _toast(l10n.memHUnreadable);
    }
  }

  Future<void> _clearAll() async {
    final l10n = AppLocalizations.of(context);
    if (!await _confirm(l10n.memHClearAllConfirm)) return;
    try {
      await ref.read(healthProfileControllerProvider.notifier).deleteProfile();
      await ref
          .read(assistantMemoryControllerProvider.notifier)
          .setEnabled(false);
      await ref.read(symptomEpisodesProvider.notifier).reload();
      if (mounted) _toast(l10n.memHCleared);
    } on Object {
      if (mounted) _toast(l10n.memHUnreadable);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    final profileAsync = ref.watch(healthProfileControllerProvider);
    final episodesAsync = ref.watch(symptomEpisodesProvider);
    final enabled =
        ref.watch(assistantMemoryControllerProvider).valueOrNull ?? false;
    final cloud =
        ref.watch(cloudAiConsentControllerProvider).valueOrNull.granted;

    final unreadable = profileAsync.error is VaultUnreadable ||
        episodesAsync.error is VaultUnreadable;
    final profile = profileAsync.valueOrNull;
    final episodes = episodesAsync.valueOrNull ?? const <SymptomEpisode>[];
    final hasConsent = profile?.hasConsent ?? false;
    final memoryActive = enabled && hasConsent;

    final muted = theme.textTheme.bodySmall?.copyWith(color: neu.textMuted);
    final body = theme.textTheme.bodyMedium?.copyWith(color: neu.textStrong);
    final head = theme.textTheme.titleMedium
        ?.copyWith(fontWeight: FontWeight.w700, color: neu.textStrong);

    final children = <Widget>[];

    if (unreadable) {
      children.add(NeuCard(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.lock_outline, color: theme.colorScheme.error),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(l10n.memHUnreadable, style: head),
                  const SizedBox(height: 4),
                  Text(l10n.memHUnreadableBody, style: body),
                ],
              ),
            ),
          ],
        ),
      ));
    } else {
      // --- Công tắc ---------------------------------------------------------
      children.add(NeuCard(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(child: Text(l10n.memHToggle, style: head)),
                Switch(
                  value: memoryActive,
                  onChanged: hasConsent
                      ? (v) => ref
                          .read(assistantMemoryControllerProvider.notifier)
                          .setEnabled(v)
                      : null,
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(l10n.memHToggleBody, style: muted),
            if (!hasConsent) ...[
              const SizedBox(height: 8),
              Text(l10n.memHNeedProfile,
                  style: muted?.copyWith(color: theme.colorScheme.error)),
            ],
          ],
        ),
      ));
      children.add(const SizedBox(height: 12));

      // --- Đang lưu gì ------------------------------------------------------
      final shown = episodes.take(20).toList();
      children.add(NeuCard(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(l10n.memHStoredTitle, style: head),
            const SizedBox(height: 8),
            Text(l10n.memHConditions(profile?.conditions.length ?? 0),
                style: body),
            Text(l10n.memHAllergies(profile?.allergies.length ?? 0),
                style: body),
            Text(l10n.memHMeds(profile?.currentMedications.length ?? 0),
                style: body),
            Text(l10n.memHEpisodes(episodes.length), style: body),
            const SizedBox(height: 8),
            Text(l10n.memHEpisodesTitle, style: muted),
            const SizedBox(height: 4),
            if (shown.isEmpty)
              Text(l10n.memHNone, style: body)
            else
              for (final e in shown)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text(
                    '${_fmt(e.at)} · ${e.terms.join(', ')}'
                    '${e.recurring ? ' · ${l10n.memHRecurringTag}' : ''}',
                    style: body,
                  ),
                ),
          ],
        ),
      ));
      children.add(const SizedBox(height: 12));

      // --- Xem trước ngữ cảnh lượt hỏi sau -----------------------------------
      final ctx = buildHealthContext(
        profile: profile,
        cloudAiConsent: cloud,
        memoryEnabled: memoryActive,
        episodes: episodes,
      );
      Widget line(String label, List<String> items) => Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Text('$label: ${items.join(', ')}', style: body),
          );
      children.add(NeuCard(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(l10n.memHPreviewTitle, style: head),
            const SizedBox(height: 8),
            if (ctx == null || ctx.isEmpty)
              Text(l10n.memHPreviewNothing, style: body)
            else ...[
              if (ctx.conditions.isNotEmpty)
                line(l10n.memHPreviewConditions, ctx.conditions),
              if (ctx.allergies.isNotEmpty)
                line(l10n.memHPreviewAllergies, ctx.allergies),
              if (ctx.recentSymptoms.isNotEmpty)
                line(l10n.memHPreviewSymptoms, ctx.recentSymptoms),
            ],
            const SizedBox(height: 8),
            Text(cloud ? l10n.memHCloudYes : l10n.memHCloudNo, style: muted),
          ],
        ),
      ));
      children.add(const SizedBox(height: 12));

      children.add(OutlinedButton.icon(
        icon: const Icon(Icons.delete_sweep_outlined),
        label: Text(l10n.memHClearLog),
        onPressed: _clearLog,
      ));
      children.add(const SizedBox(height: 8));
    }

    children.add(FilledButton.icon(
      style: FilledButton.styleFrom(
        backgroundColor: theme.colorScheme.error,
        foregroundColor: theme.colorScheme.onError,
      ),
      icon: const Icon(Icons.delete_forever),
      label: Text(l10n.memHClearAll),
      onPressed: _clearAll,
    ));

    return Scaffold(
      appBar: AppBar(title: Text(l10n.memHTitle)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            if (profileAsync.isLoading) const LinearProgressIndicator(),
            ...children,
          ],
        ),
      ),
    );
  }
}
