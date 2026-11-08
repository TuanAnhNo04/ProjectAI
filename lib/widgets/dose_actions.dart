import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Nút "Đã uống rồi" (cao >=56) + hàng nhỏ "Bỏ qua" / "Hoãn 15 phút".
class DoseActions extends ConsumerWidget {
  const DoseActions({super.key, required this.reminder});

  final MedicationReminder reminder;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final id = reminder.id;
    final taken = id != null && ref.watch(takenTodayProvider).contains(id);
    final skipped = id != null && ref.watch(skippedTodayProvider).contains(id);
    final ctrl = ref.read(adherenceControllerProvider.notifier);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        FilledButton.icon(
          style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(56),
            shape: const StadiumBorder(),
          ),
          onPressed:
              (taken || id == null) ? null : () => ctrl.logDose(reminder),
          icon: Icon(taken ? Icons.check_circle : Icons.check_circle_outline),
          label: Text(taken ? l10n.homeMarkedTaken : l10n.homeMarkTaken),
        ),
        if (skipped)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(l10n.doseASkippedToday,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodySmall),
          ),
        if (!taken && id != null)
          Row(
            children: [
              Expanded(
                child: TextButton.icon(
                  onPressed: skipped
                      ? null
                      : () => ctrl.logDose(reminder, status: kDoseSkipped),
                  icon: const Icon(Icons.skip_next_rounded, size: 18),
                  label: FittedBox(
                    fit: BoxFit.scaleDown,
                    child: Text(l10n.doseASkip, maxLines: 1, softWrap: false),
                  ),
                ),
              ),
              Expanded(
                child: TextButton.icon(
                  onPressed: () async {
                    final messenger = ScaffoldMessenger.of(context);
                    final minutes = await _pickSnooze(context, l10n);
                    if (minutes == null) return;
                    await ctrl.snooze(reminder, minutes: minutes);
                    messenger.showSnackBar(SnackBar(
                        content: Text(l10n.doseASnoozedFor(minutes))));
                  },
                  icon: const Icon(Icons.snooze_rounded, size: 18),
                  label: FittedBox(
                    fit: BoxFit.scaleDown,
                    child: Text(l10n.doseASnoozeMenu,
                        maxLines: 1, softWrap: false),
                  ),
                ),
              ),
            ],
          ),
      ],
    );
  }
}

/// Các mức hoãn trong app. Nút trên thông báo luôn hoãn 15 phút.
const List<int> snoozeOptions = [10, 15, 30, 60];

Future<int?> _pickSnooze(BuildContext context, AppLocalizations l10n) =>
    showModalBottomSheet<int>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            for (final m in snoozeOptions)
              ListTile(
                key: ValueKey('snooze$m'),
                leading: const Icon(Icons.snooze_rounded),
                title: Text(l10n.doseASnoozeFor(m)),
                onTap: () => Navigator.pop(ctx, m),
              ),
          ],
        ),
      ),
    );

/// Nhóm trong ngày: 0 Sáng(5-10), 1 Trưa(11-14), 2 Chiều(15-18), 3 Tối(19-4).
int dayPartOf(int hour) {
  if (hour >= 5 && hour < 11) return 0;
  if (hour >= 11 && hour < 15) return 1;
  if (hour >= 15 && hour < 19) return 2;
  return 3;
}
