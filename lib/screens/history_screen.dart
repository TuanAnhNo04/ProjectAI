import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
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

    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60),
        child: AppBar(
          flexibleSpace: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.deepPurple, Colors.purpleAccent],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
          ),
          title: Text(
            l10n.historyTitle,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              shadows: [
                Shadow(
                  blurRadius: 5,
                  color: Colors.black26,
                  offset: Offset(2, 2),
                ),
              ],
            ),
          ),
          centerTitle: true,
          elevation: 5,
        ),
      ),
      body: history.isLoading
          ? const Center(child: CircularProgressIndicator())
          : history.hasError
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24.0),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          history.error is AppException
                              ? (history.error! as AppException).message
                              : l10n.historyLoadFailed,
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: Colors.red),
                        ),
                        const SizedBox(height: 12),
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
                      child: Text(
                        l10n.historyEmpty,
                        style: const TextStyle(
                          fontSize: 18,
                          color: Colors.grey,
                        ),
                      ),
                    )
                  : RefreshIndicator(
                      onRefresh: () =>
                          ref.read(historyControllerProvider.notifier).load(),
                      child: ListView(
                        padding: const EdgeInsets.all(16.0),
                        children: [
                          // Banner ngoại tuyến: dữ liệu là bản cache (Day 12).
                          if (isStale)
                            OfflineBanner(
                              cachedAt: cachedAt,
                              onRetry: () => ref
                                  .read(historyControllerProvider.notifier)
                                  .load(),
                            ),
                          for (final item in items) _HistoryCard(item: item),
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
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    '${l10n.historyFile} ${item.filepath}',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.delete, color: Colors.red),
                  onPressed: () => _confirmDelete(context, ref),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              l10n.searchResultsLabel,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
            if (item.searchResults.isEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 4.0),
                child: Text(
                  l10n.noResults,
                  style: const TextStyle(color: Colors.black54),
                ),
              )
            else
              ...item.searchResults
                  .map((medicine) => _medicineBlock(medicine, l10n)),
          ],
        ),
      ),
    );
  }

  /// Xoá là thao tác không hoàn tác được (mất luôn lịch sử quét ảnh) nên hỏi
  /// lại trước, thay vì xoá ngay khi bấm nút đỏ.
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
    final ok = await ref.read(historyControllerProvider.notifier).delete(id);
    if (!context.mounted) return;

    messenger.showSnackBar(
      SnackBar(
        content: Text(ok ? l10n.deleted : l10n.deleteFailed),
        backgroundColor: ok ? Colors.green : Colors.red,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  Widget _medicineBlock(Medicine medicine, AppLocalizations l10n) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 4),
          Text(
            '${l10n.medicineName}: ${medicine.name}',
            style: const TextStyle(color: Colors.black54),
          ),
          if (medicine.composition != null)
            Text(
              '${l10n.composition}: ${medicine.composition}',
              style: const TextStyle(color: Colors.black54),
            ),
          if (medicine.uses != null)
            Text(
              '${l10n.uses}: ${medicine.uses}',
              style: const TextStyle(color: Colors.black54),
            ),
          if (medicine.sideEffects != null)
            Text(
              '${l10n.sideEffects}: ${medicine.sideEffects}',
              style: const TextStyle(color: Colors.black54),
            ),
          if (medicine.manufacturer != null)
            Text(
              '${l10n.manufacturer}: ${medicine.manufacturer}',
              style: const TextStyle(color: Colors.black54),
            ),
          if (medicine.excellentReviewPercent != null)
            Text(
              '${l10n.reviewExcellent}: ${medicine.excellentReviewPercent}',
              style: const TextStyle(color: Colors.black54),
            ),
          if (medicine.averageReviewPercent != null)
            Text(
              '${l10n.reviewAverage}: ${medicine.averageReviewPercent}',
              style: const TextStyle(color: Colors.black54),
            ),
          if (medicine.poorReviewPercent != null)
            Text(
              '${l10n.reviewPoor}: ${medicine.poorReviewPercent}',
              style: const TextStyle(color: Colors.black54),
            ),
          const Divider(),
        ],
      );
}
