import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Cảnh báo sắp hết thuốc (còn <= 3 liều). Ẩn khi không có thuốc nào sắp hết.
class LowStockBanner extends ConsumerWidget {
  const LowStockBanner({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final low = ref.watch(lowStockRemindersProvider);
    if (low.isEmpty) return const SizedBox.shrink();
    final l10n = AppLocalizations.of(context);
    final color = Theme.of(context).colorScheme.error;
    return Column(
      children: [
        for (final r in low)
          Container(
            width: double.infinity,
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: color.withOpacity(0.45)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.inventory_2_outlined, color: color, size: 22),
                const SizedBox(width: 10),
                Expanded(
                  child:
                      Text(l10n.stockBLowStock(r.medicineName, r.pillsLeft!)),
                ),
              ],
            ),
          ),
      ],
    );
  }
}
