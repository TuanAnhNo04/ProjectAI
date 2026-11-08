import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/neu.dart';
import 'package:image_processing_app/widgets/offline_banner.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> {
  @override
  void initState() {
    super.initState();
    // Tải ngay khi mở màn; lỗi hiển thị ngay trong body.
    Future.microtask(() => ref.read(historyControllerProvider.notifier).load());
  }

  @override
  Widget build(BuildContext context) {
    final history = ref.watch(historyControllerProvider);
    final snapshot = history.valueOrNull;
    final items = snapshot?.items ?? const <HistoryItem>[];
    final isStale = snapshot?.isStale ?? false;
    final cachedAt = snapshot?.cachedAt;
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(title: Text(l10n.historyTitle)),
      body: history.isLoading
          ? const Center(child: CircularProgressIndicator())
          : history.hasError
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.error_outline,
                            size: 40, color: scheme.error),
                        const SizedBox(height: 8),
                        Text(
                          history.error is AppException
                              ? (history.error! as AppException).message
                              : l10n.historyLoadFailed,
                          textAlign: TextAlign.center,
                          style: TextStyle(color: scheme.error),
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: () => ref
                              .read(historyControllerProvider.notifier)
                              .load(),
                          child: Text(l10n.retry),
                        ),
                      ],
                    ),
                  ),
                )
              : items.isEmpty
                  ? Center(
                      child: Padding(
                        padding: const EdgeInsets.all(24),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.history,
                                size: 48,
                                color: neu.textMuted.withOpacity(0.6)),
                            const SizedBox(height: 8),
                            Text(
                              l10n.historyEmpty,
                              textAlign: TextAlign.center,
                              style: Theme.of(context)
                                  .textTheme
                                  .titleMedium
                                  ?.copyWith(color: neu.textMuted),
                            ),
                          ],
                        ),
                      ),
                    )
                  : RefreshIndicator(
                      onRefresh: () =>
                          ref.read(historyControllerProvider.notifier).load(),
                      child: ListView(
                        physics: const AlwaysScrollableScrollPhysics(),
                        padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
                        children: [
                          // Banner ngoại tuyến: dữ liệu là bản cache (Day 12).
                          if (isStale)
                            OfflineBanner(
                              cachedAt: cachedAt,
                              onRetry: () => ref
                                  .read(historyControllerProvider.notifier)
                                  .load(),
                            ),
                          for (final item in items)
                            Padding(
                              padding: const EdgeInsets.symmetric(vertical: 8),
                              child: _HistoryCard(item: item),
                            ),
                          // Bắt buộc: màn lịch sử cũng là màn ra kết quả.
                          const SizedBox(height: 12),
                          const MedicalDisclaimer(),
                        ],
                      ),
                    ),
    );
  }
}

/// Một bản ghi lịch sử — dựng từ model [HistoryItem] đã kiểu hoá, không phải
/// `Map` đọc tay như trước.
class _HistoryCard extends ConsumerWidget {
  const _HistoryCard({required this.item});

  final HistoryItem item;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    return NeuCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  '${l10n.historyFile} ${item.filepath}',
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w800,
                    color: neu.textStrong,
                  ),
                ),
              ),
              IconButton(
                tooltip: l10n.delete,
                icon: Icon(Icons.delete, color: theme.colorScheme.error),
                onPressed: () => _confirmDelete(context, ref),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            l10n.searchResultsLabel,
            style: theme.textTheme.labelLarge?.copyWith(color: neu.accent),
          ),
          if (item.searchResults.isEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4.0),
              child: Text(
                l10n.noResults,
                style: TextStyle(color: neu.textMuted),
              ),
            )
          else
            ...item.searchResults
                .map((medicine) => _medicineBlock(context, medicine, l10n)),
        ],
      ),
    );
  }

  /// Xoá là thao tác không hoàn tác được (mất luôn lịch sử quét ảnh) nên hỏi
  /// lại trước, thay vì xoá ngay khi bấm nút xoá.
  Future<void> _confirmDelete(BuildContext context, WidgetRef ref) async {
    final l10n = AppLocalizations.of(context);
    final id = item.id;
    if (id == null) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(l10n.deleteRecordTitle),
        content: Text(l10n.deleteRecordBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text(l10n.cancel),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text(l10n.delete),
          ),
        ],
      ),
    );
    if (confirmed != true || !context.mounted) return;

    final messenger = ScaffoldMessenger.of(context);
    final scheme = Theme.of(context).colorScheme;
    final neu = NeuTheme.of(context);
    final ok = await ref.read(historyControllerProvider.notifier).delete(id);
    if (!context.mounted) return;

    messenger.showSnackBar(
      SnackBar(
        content: Text(ok ? l10n.deleted : l10n.deleteFailed),
        backgroundColor: ok ? neu.accent : scheme.error,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  Widget _line(BuildContext context, String label, Object value) {
    final neu = NeuTheme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 2),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: '$label: ',
              style: TextStyle(color: neu.textMuted),
            ),
            TextSpan(
              text: '$value',
              style: TextStyle(color: neu.textStrong),
            ),
          ],
        ),
      ),
    );
  }

  Widget _medicineBlock(
    BuildContext context,
    Medicine medicine,
    AppLocalizations l10n,
  ) {
    final neu = NeuTheme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 6),
        _line(context, l10n.medicineName, medicine.name),
        if (medicine.composition != null)
          _line(context, l10n.composition, medicine.composition!),
        if (medicine.uses != null) _line(context, l10n.uses, medicine.uses!),
        if (medicine.sideEffects != null)
          _line(context, l10n.sideEffects, medicine.sideEffects!),
        if (medicine.manufacturer != null)
          _line(context, l10n.manufacturer, medicine.manufacturer!),
        if (medicine.excellentReviewPercent != null)
          _line(
              context, l10n.reviewExcellent, medicine.excellentReviewPercent!),
        if (medicine.averageReviewPercent != null)
          _line(context, l10n.reviewAverage, medicine.averageReviewPercent!),
        if (medicine.poorReviewPercent != null)
          _line(context, l10n.reviewPoor, medicine.poorReviewPercent!),
        Divider(color: neu.textMuted.withOpacity(0.2)),
      ],
    );
  }
}
