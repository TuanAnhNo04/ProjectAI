import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/data/models/prescription.dart';
import 'package:image_processing_app/domain/ocr_confidence.dart';
import 'package:image_processing_app/domain/prescription_parser.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Người dùng xem lại & sửa kết quả AI đọc đơn thuốc trước khi tạo lịch nhắc.
/// Không có gì được lưu cho tới khi bấm "Lưu".
class PrescriptionReviewScreen extends ConsumerStatefulWidget {
  const PrescriptionReviewScreen({
    super.key,
    required this.items,
    this.diagnosis,
  });

  final List<ParsedPrescriptionItem> items;
  final String? diagnosis;

  @override
  ConsumerState<PrescriptionReviewScreen> createState() =>
      _PrescriptionReviewScreenState();
}

class _EditableItem {
  _EditableItem(ParsedPrescriptionItem p)
      : name = TextEditingController(text: p.medicineName),
        dose = TextEditingController(text: p.doseText),
        times = [for (final t in p.times) _parseTime(t)],
        meal = p.mealRelation,
        timesPerDay = p.timesPerDay,
        needsReview = p.needsReview,
        pillsTotal = TextEditingController(),
        pillsPerDose =
            TextEditingController(text: '${_defaultPerDose(p.doseText)}');

  bool include = true;
  bool confirmed = false;
  final TextEditingController pillsTotal;
  final TextEditingController pillsPerDose;

  /// "2 viên" -> 2; không rõ -> 1.
  static int _defaultPerDose(String dose) {
    final m =
        RegExp(r'(\d+)\s*(viên|vien)', caseSensitive: false).firstMatch(dose);
    final n = m == null ? null : int.tryParse(m.group(1)!);
    return (n == null || n < 1 || n > 20) ? 1 : n;
  }

  /// Kết quả "Đối chiếu danh mục": `null` = chưa đối chiếu (hoặc tên đã sửa sau đó).
  CatalogMatch? match;
  List<Medicine> suggestions = const [];
  bool checking = false;
  bool checkFailed = false;

  void resetCheck() {
    match = null;
    suggestions = const [];
    checkFailed = false;
  }

  List<OcrIssue> get issues => assessOcrItem(
        name: name.text,
        dose: dose.text,
        needsReview: needsReview,
        match: match,
      );

  /// Cần khoá xác nhận: OCR không chắc, tên lạ/quá ngắn, hoặc không có trong danh mục.
  bool get needsLock => issues.any(lockingIssues.contains);

  /// Cho phép lưu khi mục này (nếu được chọn) đã có tên và đã tick xác nhận.
  bool get saveReady =>
      !include || (name.text.trim().isNotEmpty && (!needsLock || confirmed));
  final TextEditingController name;
  final TextEditingController dose;
  final List<TimeOfDay> times;
  MealRelation meal;
  final int? timesPerDay;
  final bool needsReview;

  static TimeOfDay _parseTime(String s) {
    final parts = s.split(':');
    return TimeOfDay(
      hour: int.tryParse(parts.first) ?? 8,
      minute: parts.length > 1 ? int.tryParse(parts[1]) ?? 0 : 0,
    );
  }

  void dispose() {
    name.dispose();
    dose.dispose();
    pillsTotal.dispose();
    pillsPerDose.dispose();
  }
}

String _hhmm(TimeOfDay t) =>
    '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

class _PrescriptionReviewScreenState
    extends ConsumerState<PrescriptionReviewScreen> {
  late final List<_EditableItem> _items = [
    for (final p in widget.items) _EditableItem(p),
  ];
  bool _addDiagnosis = false;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      if (ref.read(healthProfileControllerProvider).valueOrNull == null) {
        ref.read(healthProfileControllerProvider.notifier).load();
      }
      ref.read(reminderControllerProvider.notifier).load();
    });
  }

  @override
  void dispose() {
    for (final i in _items) {
      i.dispose();
    }
    super.dispose();
  }

  String _mealLabel(AppLocalizations l10n, MealRelation m) => switch (m) {
        MealRelation.none => l10n.rxMealNone,
        MealRelation.before => l10n.rxMealBefore,
        MealRelation.after => l10n.rxMealAfter,
        MealRelation.during => l10n.rxMealDuring,
      };

  Future<void> _pickTime(_EditableItem item, int? index) async {
    final initial =
        index == null ? const TimeOfDay(hour: 8, minute: 0) : item.times[index];
    final picked = await showTimePicker(context: context, initialTime: initial);
    if (picked == null || !mounted) return;
    setState(() {
      if (index == null) {
        item.times.add(picked);
      } else {
        item.times[index] = picked;
      }
      item.times.sort(
          (a, b) => (a.hour * 60 + a.minute).compareTo(b.hour * 60 + b.minute));
    });
  }

  /// Tra tên thuốc trong danh mục của server (`/v1/search`: công khai, không lưu,
  /// không ghi log truy vấn). Chỉ chạy khi người dùng bấm — không tự gửi tên thuốc.
  Future<void> _checkCatalog(_EditableItem item) async {
    final query = item.name.text.trim();
    if (query.isEmpty) return;
    setState(() {
      item.resetCheck();
      item.checking = true;
    });
    try {
      final res = await ref.read(medicineRepositoryProvider).search(query);
      if (!mounted) return;
      final top = res.results.where((m) => m.name.isNotEmpty).take(3).toList();
      setState(() {
        item.suggestions = top;
        item.match = matchCatalog(query, top.map((m) => m.name));
      });
    } catch (_) {
      // Không log: lỗi có thể chứa tên thuốc trong URL.
      if (!mounted) return;
      setState(() => item.checkFailed = true);
    } finally {
      if (mounted) setState(() => item.checking = false);
    }
  }

  String _issueLabel(AppLocalizations l10n, OcrIssue issue) => switch (issue) {
        OcrIssue.unclearFrequency => l10n.rxIssueFrequency,
        OcrIssue.missingDose => l10n.rxIssueDose,
        OcrIssue.shortName => l10n.rxIssueShortName,
        OcrIssue.suspiciousChars => l10n.rxIssueOddChars,
        OcrIssue.notInCatalog => l10n.rxIssueNotInCatalog,
      };

  Widget _buildCatalogCheck(_EditableItem item, AppLocalizations l10n) {
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    final match = item.match;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 8),
        OutlinedButton.icon(
          key: const ValueKey('checkCatalog'),
          onPressed: item.include && !item.checking
              ? () => _checkCatalog(item)
              : null,
          icon: item.checking
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.fact_check_outlined),
          label: Text(l10n.rxCheckCatalog),
        ),
        Text(
          l10n.rxCheckCatalogNote,
          style: theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
        ),
        if (item.checkFailed)
          _Banner(
            icon: Icons.cloud_off,
            text: l10n.rxCatalogError,
            color: Colors.orange.shade800,
          )
        else if (match != null)
          _Banner(
            icon: switch (match) {
              CatalogMatch.exact => Icons.verified_outlined,
              CatalogMatch.similar => Icons.help_outline,
              CatalogMatch.none => Icons.error_outline,
            },
            text: switch (match) {
              CatalogMatch.exact => l10n.rxCatalogExact,
              CatalogMatch.similar => l10n.rxCatalogSimilar,
              CatalogMatch.none => l10n.rxCatalogNone,
            },
            color: switch (match) {
              CatalogMatch.exact => neu.accent,
              CatalogMatch.similar => Colors.orange.shade800,
              CatalogMatch.none => theme.colorScheme.error,
            },
          ),
        for (final m in item.suggestions)
          ListTile(
            contentPadding: EdgeInsets.zero,
            title: Text(m.name),
            subtitle: (m.composition ?? '').trim().isEmpty
                ? null
                : Text(l10n.rxCatalogStrength(m.composition!.trim())),
            trailing: TextButton(
              onPressed: item.include
                  ? () => setState(() {
                        item.name.text = m.name;
                        item.match = CatalogMatch.exact;
                        item.suggestions = const [];
                      })
                  : null,
              child: Text(l10n.rxUseThisName),
            ),
          ),
      ],
    );
  }

  Future<void> _save() async {
    final l10n = AppLocalizations.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);
    final records = [
      for (final i in _items)
        if (i.include && i.name.text.trim().isNotEmpty && i.times.isNotEmpty)
          PrescriptionItemRecord(
            medicineName: i.name.text.trim(),
            doseText: i.dose.text.trim(),
            timesPerDay: i.times.length,
            slots: {for (final t in i.times) _hhmm(t)}.toList(),
            mealRelation: i.meal.name,
            pillsTotal: int.tryParse(i.pillsTotal.text.trim()),
            pillsPerDose: int.tryParse(i.pillsPerDose.text.trim()) ?? 1,
          ),
    ];
    if (records.isEmpty) {
      messenger.showSnackBar(SnackBar(content: Text(l10n.rxNoItemSelected)));
      return;
    }
    setState(() => _saving = true);
    try {
      final count = await ref
          .read(prescriptionControllerProvider.notifier)
          .saveWithReminders(
            items: records,
            mealLabel: (m) => _mealLabel(
              l10n,
              MealRelation.values.firstWhere(
                (e) => e.name == m,
                orElse: () => MealRelation.none,
              ),
            ),
            diagnosis: widget.diagnosis,
            addDiagnosisToProfile: _addDiagnosis,
          );
      if (!mounted) return;
      navigator.pop(count);
      messenger.showSnackBar(SnackBar(
        content: Text(l10n.rxSaved(count)),
        behavior: SnackBarBehavior.floating,
      ));
    } catch (_) {
      // Không log chi tiết: có thể chứa tên thuốc.
      if (!mounted) return;
      setState(() => _saving = false);
      messenger.showSnackBar(SnackBar(content: Text(l10n.rxSaveFailed)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final profile = ref.watch(healthProfileControllerProvider).valueOrNull;
    final hasConsent = profile?.hasConsent ?? false;
    final reminders = ref.watch(reminderControllerProvider).valueOrNull ??
        const <MedicationReminder>[];
    final existing = <String>{
      for (final r in reminders) r.medicineName.trim().toLowerCase(),
    };
    final diagnosis = widget.diagnosis?.trim();

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(title: Text(l10n.rxReviewTitle)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            _Banner(
              icon: Icons.auto_awesome,
              text: l10n.rxAiBanner,
              color: theme.colorScheme.tertiary,
            ),
            const SizedBox(height: 12),
            if (diagnosis != null && diagnosis.isNotEmpty) ...[
              NeuCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      l10n.rxDiagnosisTitle,
                      style: theme.textTheme.titleSmall
                          ?.copyWith(color: neu.textMuted),
                    ),
                    const SizedBox(height: 4),
                    Text(diagnosis,
                        style: theme.textTheme.bodyLarge
                            ?.copyWith(color: neu.textStrong)),
                    if (hasConsent)
                      CheckboxListTile(
                        contentPadding: EdgeInsets.zero,
                        value: _addDiagnosis,
                        onChanged: (v) =>
                            setState(() => _addDiagnosis = v ?? false),
                        title: Text(l10n.rxAddDiagnosisToProfile),
                        controlAffinity: ListTileControlAffinity.leading,
                      )
                    else
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(
                          l10n.rxProfileConsentRequired,
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: neu.textMuted),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
            ] else if (!hasConsent) ...[
              Text(
                l10n.rxProfileConsentRequired,
                style:
                    theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
              ),
              const SizedBox(height: 12),
            ],
            for (final item in _items) ...[
              _buildItemCard(item, l10n, existing),
              const SizedBox(height: 12),
            ],
            FilledButton.icon(
              onPressed:
                  _saving || !_items.every((i) => i.saveReady) ? null : _save,
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
                shape: const StadiumBorder(),
              ),
              icon: _saving
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2.5),
                    )
                  : const Icon(Icons.alarm_add),
              label: Text(l10n.rxSave),
            ),
            const SizedBox(height: 12),
            const MedicalDisclaimer(),
          ],
        ),
      ),
    );
  }

  Widget _buildItemCard(
    _EditableItem item,
    AppLocalizations l10n,
    Set<String> existing,
  ) {
    final theme = Theme.of(context);
    final isDuplicate = existing.contains(item.name.text.trim().toLowerCase());
    final locked = item.include && item.needsLock;
    // "Không rõ số lần dùng" đã có banner riêng ở trên.
    final otherIssues =
        item.issues.where((i) => i != OcrIssue.unclearFrequency).toList();
    final card = NeuCard(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CheckboxListTile(
            contentPadding: EdgeInsets.zero,
            value: item.include,
            onChanged: (v) => setState(() => item.include = v ?? false),
            title: Text(l10n.rxInclude),
            controlAffinity: ListTileControlAffinity.leading,
          ),
          if (item.needsReview)
            _Banner(
              icon: Icons.warning_amber_rounded,
              text: l10n.rxNeedsReview,
              color: theme.colorScheme.error,
            ),
          if (otherIssues.isNotEmpty)
            _Banner(
              key: const ValueKey('lowConfidence'),
              icon: Icons.report_gmailerrorred_outlined,
              text: l10n.rxLowConfidence(
                  otherIssues.map((i) => _issueLabel(l10n, i)).join(', ')),
              color: Colors.orange.shade800,
            ),
          if (isDuplicate)
            _Banner(
              icon: Icons.content_copy,
              text: l10n.rxDuplicate,
              color: Colors.orange.shade800,
            ),
          if (locked)
            CheckboxListTile(
              key: const ValueKey('confirmItem'),
              contentPadding: EdgeInsets.zero,
              value: item.confirmed,
              onChanged: (v) => setState(() => item.confirmed = v ?? false),
              title: Text(l10n.stockBConfirmed),
              subtitle: Text(l10n.stockBNeedsCheck),
              controlAffinity: ListTileControlAffinity.leading,
            ),
          const SizedBox(height: 8),
          TextField(
            controller: item.name,
            enabled: item.include,
            // Sửa tên -> kết quả đối chiếu cũ không còn đúng.
            onChanged: (_) => setState(item.resetCheck),
            decoration: InputDecoration(
              labelText: l10n.rxMedicineName,
              border: const OutlineInputBorder(),
            ),
          ),
          _buildCatalogCheck(item, l10n),
          const SizedBox(height: 16),
          TextField(
            controller: item.dose,
            enabled: item.include,
            onChanged: (_) => setState(() {}),
            decoration: InputDecoration(
              labelText: l10n.rxDose,
              border: const OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: item.pillsTotal,
            enabled: item.include,
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            decoration: InputDecoration(
              labelText: l10n.stockBPillsTotal,
              border: const OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: item.pillsPerDose,
            enabled: item.include,
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            decoration: InputDecoration(
              labelText: l10n.stockBPillsPerDose,
              border: const OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          Text(l10n.rxTimes, style: theme.textTheme.labelLarge),
          const SizedBox(height: 4),
          Wrap(
            spacing: 8,
            runSpacing: 4,
            children: [
              for (var i = 0; i < item.times.length; i++)
                InputChip(
                  label: Text(_hhmm(item.times[i])),
                  avatar: const Icon(Icons.schedule, size: 18),
                  isEnabled: item.include,
                  onPressed: () => _pickTime(item, i),
                  onDeleted: item.times.length > 1
                      ? () => setState(() => item.times.removeAt(i))
                      : null,
                ),
              ActionChip(
                avatar: const Icon(Icons.add, size: 18),
                label: Text(l10n.rxAddTime),
                onPressed: item.include ? () => _pickTime(item, null) : null,
              ),
            ],
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<MealRelation>(
            value: item.meal,
            decoration: InputDecoration(
              labelText: l10n.rxMealRelation,
              border: const OutlineInputBorder(),
            ),
            items: [
              for (final m in MealRelation.values)
                DropdownMenuItem(value: m, child: Text(_mealLabel(l10n, m))),
            ],
            onChanged: item.include
                ? (m) => setState(() => item.meal = m ?? MealRelation.none)
                : null,
          ),
        ],
      ),
    );
    if (!locked || item.confirmed) return card;
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: theme.colorScheme.error, width: 2),
      ),
      child: card,
    );
  }
}

class _Banner extends StatelessWidget {
  const _Banner({
    super.key,
    required this.icon,
    required this.text,
    required this.color,
  });

  final IconData icon;
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 4),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(width: 8),
          Expanded(child: Text(text)),
        ],
      ),
    );
  }
}
