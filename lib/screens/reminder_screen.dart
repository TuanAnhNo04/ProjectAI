import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/process_image_screen.dart';
import 'package:image_processing_app/widgets/empty_state.dart';
import 'package:image_processing_app/widgets/dose_actions.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Màn hình quản lý lịch nhắc uống thuốc (tab, kiểu neumorphism).
class ReminderScreen extends ConsumerStatefulWidget {
  const ReminderScreen({super.key});

  @override
  ConsumerState<ReminderScreen> createState() => _ReminderScreenState();
}

class _ReminderScreenState extends ConsumerState<ReminderScreen> {
  @override
  void initState() {
    super.initState();
    // Hoãn tới sau khung hình đầu: đổi state của provider ngay trong lúc dựng widget bị
    // Riverpod từ chối khi tab Trang chủ đang lắng nghe cùng provider này.
    Future.microtask(() {
      if (!mounted) return;
      ref.read(reminderControllerProvider.notifier).load();
      ref.read(adherenceControllerProvider.notifier).load();
    });
  }

  @override
  Widget build(BuildContext context) {
    final remindersAsync = ref.watch(reminderControllerProvider);
    final reminders = remindersAsync.valueOrNull ?? const [];
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(
        title: Text(l10n.reminderTitle),
        foregroundColor: Colors.white,
        titleTextStyle: Theme.of(context).textTheme.titleLarge?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w700,
            ),
        flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [neu.accent, neu.accent.withOpacity(0.72)],
            ),
          ),
        ),
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: reminders.isEmpty
                ? AppEmptyState(
                    icon: Icons.alarm_add_rounded,
                    title: l10n.stockBEmptyReminderTitle,
                    message: l10n.reminderEmpty,
                    actionLabel: l10n.stockBScanPrescription,
                    onAction: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => const ImageProcessingScreen(),
                      ),
                    ),
                  )
                : ListView(
                    // Chừa chỗ dưới để nút thêm không che thẻ cuối.
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 150),
                    children: [
                      for (final part in _partsOf(reminders, l10n)) ...[
                        _PartHeader(part: part),
                        for (final r in part.items)
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 6),
                            child: _ReminderCard(reminder: r),
                          ),
                      ],
                    ],
                  ),
          ),
          Positioned(
            right: 20,
            bottom: 20,
            child: NeuCircleButton(
              icon: Icons.add,
              filled: true,
              size: 56,
              tooltip: l10n.reminderAdd,
              onPressed: () => _showAddSheet(context),
            ),
          ),
        ],
      ),
    );
  }

  List<_Part> _partsOf(List<MedicationReminder> list, AppLocalizations l10n) {
    final meta = [
      (Icons.wb_sunny_outlined, l10n.doseAMorning, const Color(0xFFE08A00)),
      (Icons.light_mode, l10n.doseANoon, const Color(0xFFD35400)),
      (Icons.wb_twilight, l10n.doseAAfternoon, const Color(0xFF8E44AD)),
      (Icons.nights_stay_outlined, l10n.doseAEvening, const Color(0xFF2C5AA0)),
    ];
    final sorted = [...list]..sort((a, b) {
        final h = a.hour.compareTo(b.hour);
        return h != 0 ? h : a.minute.compareTo(b.minute);
      });
    return [
      for (var i = 0; i < 4; i++)
        if (sorted.any((r) => dayPartOf(r.hour) == i))
          _Part(meta[i].$1, meta[i].$2, meta[i].$3, [
            for (final r in sorted)
              if (dayPartOf(r.hour) == i) r
          ]),
    ];
  }

  Future<void> _showAddSheet(BuildContext context) async {
    // Danh sách tên thuốc đã có (nhắc thuốc + hồ sơ sức khoẻ) để cảnh báo trùng
    // tên khi thêm mới — so khớp văn bản thuần, KHÔNG suy luận tương tác thuốc
    // (AGENTS.md mục 3: mọi khẳng định y khoa phải có trích dẫn).
    final reminders =
        ref.read(reminderControllerProvider).valueOrNull ?? const [];
    final profile = ref.read(healthProfileControllerProvider).valueOrNull;
    final existingNames = <String>{
      for (final r in reminders) r.medicineName,
      ...?profile?.currentMedications,
    };
    final result = await showModalBottomSheet<MedicationReminder>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      backgroundColor: NeuTheme.of(context).surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (ctx) => _AddReminderSheet(existingMedicineNames: existingNames),
    );
    if (result != null) {
      await ref.read(reminderControllerProvider.notifier).add(result);
    }
  }
}

class _Part {
  const _Part(this.icon, this.label, this.color, this.items);
  final IconData icon;
  final String label;
  final Color color;
  final List<MedicationReminder> items;
}

class _PartHeader extends StatelessWidget {
  const _PartHeader({required this.part});
  final _Part part;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 12, bottom: 6, left: 4),
      child: Row(
        children: [
          Icon(part.icon, color: part.color),
          const SizedBox(width: 8),
          Text(
            part.label,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: NeuTheme.of(context).textStrong),
          ),
        ],
      ),
    );
  }
}

/// Nhãn ngày lặp: "Mỗi ngày" hoặc tên thứ viết tắt theo ngôn ngữ hiện tại.
String _daysLabel(BuildContext context, List<int> days) {
  final l10n = AppLocalizations.of(context);
  final valid = days.where((d) => d >= 1 && d <= 7).toSet();
  if (valid.isEmpty || valid.length == 7) return l10n.reminderEveryDay;
  final locale = Localizations.localeOf(context).toString();
  final fmt = DateFormat.E(locale);
  final sorted = valid.toList()..sort();
  // 2024-01-01 là thứ Hai => ngày d (1..7) tương ứng DateTime(2024, 1, d).
  return sorted.map((d) => fmt.format(DateTime(2024, 1, d))).join(', ');
}

class _ReminderCard extends ConsumerWidget {
  const _ReminderCard({
    required this.reminder,
  });

  final MedicationReminder reminder;

  Future<void> _confirmDelete(
      BuildContext context, WidgetRef ref, AppLocalizations l10n) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l10n.delete),
        content: Text(reminder.medicineName),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l10n.cancel),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(l10n.delete),
          ),
        ],
      ),
    );
    if (ok == true && reminder.id != null) {
      await ref.read(reminderControllerProvider.notifier).remove(reminder.id!);
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    final l10n = AppLocalizations.of(context);
    final active = reminder.isActive;
    final timeColor = active ? neu.accent : neu.textMuted;

    return NeuCard(
      padding: const EdgeInsets.fromLTRB(14, 14, 8, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: neu.accentSoft,
                ),
                child: Icon(Icons.medication_outlined,
                    color: active ? neu.accent : neu.textMuted),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      reminder.medicineName,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: neu.textStrong,
                      ),
                    ),
                    if (reminder.dosage != null && reminder.dosage!.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 2),
                        child: Text(
                          reminder.dosage!,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: theme.textTheme.bodyMedium
                              ?.copyWith(color: neu.textMuted),
                        ),
                      ),
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        _daysLabel(context, reminder.daysOfWeek),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodySmall
                            ?.copyWith(color: neu.textMuted),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: neu.accentSoft,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      reminder.timeDisplay,
                      style: theme.textTheme.labelLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: timeColor,
                      ),
                    ),
                  ),
                  Switch(
                    value: active,
                    onChanged: (v) => ref
                        .read(reminderControllerProvider.notifier)
                        .toggleActive(reminder, v),
                  ),
                ],
              ),
              IconButton(
                icon:
                    Icon(Icons.delete_outline, color: theme.colorScheme.error),
                tooltip: l10n.delete,
                onPressed: () => _confirmDelete(context, ref, l10n),
              ),
            ],
          ),
          const SizedBox(height: 8),
          DoseActions(reminder: reminder),
        ],
      ),
    );
  }
}

class _AddReminderSheet extends StatefulWidget {
  const _AddReminderSheet({required this.existingMedicineNames});

  /// Tên thuốc đã có (nhắc thuốc + hồ sơ sức khoẻ) — chỉ để cảnh báo trùng tên,
  /// không chặn lưu.
  final Set<String> existingMedicineNames;

  @override
  State<_AddReminderSheet> createState() => _AddReminderSheetState();
}

class _AddReminderSheetState extends State<_AddReminderSheet> {
  final _nameCtrl = TextEditingController();
  final _dosageCtrl = TextEditingController();
  final _pillsLeftCtrl = TextEditingController();
  final _pillsPerDoseCtrl = TextEditingController(text: '1');
  int _hour = 8;
  int _minute = 0;

  /// So khớp văn bản thuần (bỏ khoảng trắng/hoa-thường) — KHÔNG so hoạt chất hay
  /// suy luận tương tác thuốc.
  bool get _isDuplicateName {
    final name = _nameCtrl.text.trim().toLowerCase();
    if (name.isEmpty) return false;
    return widget.existingMedicineNames
        .any((existing) => existing.trim().toLowerCase() == name);
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _dosageCtrl.dispose();
    _pillsLeftCtrl.dispose();
    _pillsPerDoseCtrl.dispose();
    super.dispose();
  }

  String get _timeText =>
      '${_hour.toString().padLeft(2, '0')}:${_minute.toString().padLeft(2, '0')}';

  Future<void> _pickTime() async {
    final picked = await showTimePicker(
      context: context,
      initialTime: TimeOfDay(hour: _hour, minute: _minute),
    );
    if (picked != null) {
      setState(() {
        _hour = picked.hour;
        _minute = picked.minute;
      });
    }
  }

  void _save() {
    final name = _nameCtrl.text.trim();
    if (name.isEmpty) return;
    final dosage = _dosageCtrl.text.trim();
    Navigator.pop(
      context,
      MedicationReminder(
        medicineName: name,
        hour: _hour,
        minute: _minute,
        dosage: dosage.isEmpty ? null : dosage,
        pillsLeft: int.tryParse(_pillsLeftCtrl.text.trim()),
        pillsPerDose:
            (int.tryParse(_pillsPerDoseCtrl.text.trim()) ?? 1).clamp(1, 99),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    final l10n = AppLocalizations.of(context);

    return Padding(
      padding:
          EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: neu.textMuted.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              l10n.reminderSetAlarm,
              textAlign: TextAlign.center,
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w800,
                color: neu.textStrong,
              ),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _nameCtrl,
              autofocus: true,
              textInputAction: TextInputAction.next,
              onChanged: (_) => setState(() {}),
              decoration: InputDecoration(
                labelText: l10n.reminderMedicineName,
                prefixIcon: const Icon(Icons.medication),
              ),
            ),
            if (_isDuplicateName) ...[
              const SizedBox(height: 16),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: theme.colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded,
                        color: theme.colorScheme.onErrorContainer, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        l10n.reminderDuplicateWarning,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onErrorContainer,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            const SizedBox(height: 16),
            TextField(
              controller: _dosageCtrl,
              textInputAction: TextInputAction.done,
              decoration: InputDecoration(
                labelText: l10n.reminderDosage,
                hintText: l10n.reminderDosageHint,
                prefixIcon: const Icon(Icons.science),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _pillsLeftCtrl,
                    keyboardType: TextInputType.number,
                    inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                    decoration: InputDecoration(
                      labelText: l10n.stockBPillsLeft,
                      prefixIcon: const Icon(Icons.inventory_2_outlined),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: TextField(
                    controller: _pillsPerDoseCtrl,
                    keyboardType: TextInputType.number,
                    inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                    decoration: InputDecoration(
                      labelText: l10n.stockBPillsPerDose,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            NeuCard(
              onTap: _pickTime,
              semanticLabel: '${l10n.reminderTimeLabel} $_timeText',
              child: Row(
                children: [
                  Icon(Icons.access_time, color: neu.accent),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      l10n.reminderTimeLabel,
                      style: theme.textTheme.titleMedium
                          ?.copyWith(color: neu.textMuted),
                    ),
                  ),
                  Text(
                    _timeText,
                    style: theme.textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                      color: neu.accent,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    style: OutlinedButton.styleFrom(
                      minimumSize:
                          const Size.fromHeight(AppTheme.minTouchTarget),
                      shape: const StadiumBorder(),
                    ),
                    onPressed: () => Navigator.pop(context),
                    child: Text(l10n.cancel),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton(
                    style: FilledButton.styleFrom(
                      minimumSize:
                          const Size.fromHeight(AppTheme.minTouchTarget),
                      shape: const StadiumBorder(),
                    ),
                    onPressed: _nameCtrl.text.trim().isEmpty ? null : _save,
                    child: Text(l10n.reminderSave),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
