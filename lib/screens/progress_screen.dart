import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/domain/adherence_report.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Màn hình "Báo cáo tuân thủ" — streak, tóm tắt 7/30 ngày, biểu đồ cột và
/// chế độ "Cho bác sĩ xem" (chỉ đọc, không xuất/chia sẻ, dữ liệu ở lại máy).
class ProgressScreen extends ConsumerStatefulWidget {
  const ProgressScreen({super.key});

  @override
  ConsumerState<ProgressScreen> createState() => _ProgressScreenState();
}

class _ProgressScreenState extends ConsumerState<ProgressScreen> {
  int _days = 7;
  bool _doctorMode = false;

  @override
  void initState() {
    super.initState();
    Future.microtask(() {
      if (mounted) ref.read(adherenceControllerProvider.notifier).load();
    });
  }

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final l10n = AppLocalizations.of(context);
    final logsAsync = ref.watch(adherenceControllerProvider);
    final streak = ref.watch(adherenceStreakProvider);

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(
        title: Text(_doctorMode ? l10n.reportCDoctorTitle : l10n.reportCTitle),
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
      body: logsAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (_, __) => Center(
          child: Text(
            Localizations.localeOf(context).languageCode == 'vi'
                ? 'Không tải được dữ liệu tiến độ.'
                : 'Could not load progress.',
            style: TextStyle(color: neu.textMuted),
          ),
        ),
        data: (logs) {
          final report = buildReport(
            logs,
            now: DateTime.now(),
            days: _doctorMode ? 30 : _days,
          );
          if (_doctorMode) {
            return _DoctorView(
              report: report,
              l10n: l10n,
              onBack: () => setState(() => _doctorMode = false),
            );
          }
          return ListView(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
            children: [
              SegmentedButton<int>(
                segments: [
                  ButtonSegment(value: 7, label: Text(l10n.reportCRange7)),
                  ButtonSegment(value: 30, label: Text(l10n.reportCRange30)),
                ],
                selected: {_days},
                onSelectionChanged: (v) => setState(() => _days = v.first),
              ),
              const SizedBox(height: 16),
              _StreakCard(streak: streak, l10n: l10n),
              const SizedBox(height: 16),
              if (report.isEmpty)
                _EmptyReport(text: l10n.reportCEmpty)
              else ...[
                _SummaryGrid(report: report, l10n: l10n),
                const SizedBox(height: 24),
                NeuSectionTitle(title: l10n.reportCChartTitle),
                const SizedBox(height: 12),
                _BarChart(report: report, l10n: l10n),
              ],
              const SizedBox(height: 24),
              NeuSectionTitle(title: l10n.progressWeekTitle),
              const SizedBox(height: 12),
              _WeekRow(
                takenDates: {
                  for (final log in logs)
                    if (!log.skipped &&
                        log.status != kDoseSkipped &&
                        log.status != kDoseSnoozed)
                      DateTime(
                        log.takenAt.year,
                        log.takenAt.month,
                        log.takenAt.day,
                      ),
                },
              ),
              const SizedBox(height: 24),
              _Note(text: l10n.reportCNote),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: () => setState(() => _doctorMode = true),
                icon: const Icon(Icons.medical_services_outlined),
                label: Text(l10n.reportCDoctorMode),
              ),
            ],
          );
        },
      ),
    );
  }
}

String _pct(double? r, AppLocalizations l10n) =>
    r == null ? l10n.reportCNoRate : '${(r * 100).round()}%';

class _EmptyReport extends StatelessWidget {
  const _EmptyReport({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        children: [
          Icon(Icons.insights_outlined, size: 48, color: neu.textMuted),
          const SizedBox(height: 12),
          Text(
            text,
            textAlign: TextAlign.center,
            style: TextStyle(color: neu.textMuted, fontSize: 16),
          ),
        ],
      ),
    );
  }
}

class _Note extends StatelessWidget {
  const _Note({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.info_outline, size: 20, color: neu.textMuted),
        const SizedBox(width: 8),
        Expanded(
          child:
              Text(text, style: TextStyle(color: neu.textMuted, fontSize: 14)),
        ),
      ],
    );
  }
}

class _SummaryGrid extends StatelessWidget {
  const _SummaryGrid({required this.report, required this.l10n});
  final AdherenceReport report;
  final AppLocalizations l10n;

  @override
  Widget build(BuildContext context) {
    final items = [
      (l10n.reportCRate, _pct(report.adherenceRate, l10n), Icons.percent),
      (l10n.reportCTaken, '${report.takenTotal}', Icons.check_circle_outline),
      (l10n.reportCSkipped, '${report.skipped}', Icons.cancel_outlined),
      (
        '${l10n.reportCStreakNow} / ${l10n.reportCStreakBest}',
        '${report.currentStreak} / ${report.bestStreak}',
        Icons.local_fire_department_outlined,
      ),
    ];
    return Column(
      children: [
        for (var i = 0; i < items.length; i += 2)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            // IntrinsicHeight: Row(stretch) trong ListView không có chiều cao hữu hạn
            // -> "BoxConstraints forces an infinite height" khi đã có log (QA phát hiện).
            child: IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (var j = i; j < i + 2; j++) ...[
                    if (j > i) const SizedBox(width: 12),
                    Expanded(
                      child: _StatCard(
                        label: items[j].$1,
                        value: items[j].$2,
                        icon: items[j].$3,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({
    required this.label,
    required this.value,
    required this.icon,
  });
  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Semantics(
      label: '$label: $value',
      excludeSemantics: true,
      child: NeuCard(
        radius: 20,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: neu.accent),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.w800,
                color: neu.textStrong,
              ),
            ),
            Text(label, style: TextStyle(fontSize: 14, color: neu.textMuted)),
          ],
        ),
      ),
    );
  }
}

class _BarChart extends StatelessWidget {
  const _BarChart({required this.report, required this.l10n});
  final AdherenceReport report;
  final AppLocalizations l10n;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final locale = Localizations.localeOf(context).toString();
    final maxV = report.perDay
        .map((d) => d.takenCount)
        .fold<int>(1, (a, b) => b > a ? b : a);
    final dense = report.perDay.length > 10;
    const chartH = 100.0;
    // Phần chữ (số lượng + nhãn ngày) cao theo cỡ chữ hệ thống -> chừa chỗ tương ứng.
    final textScale = MediaQuery.textScalerOf(context).scale(1).clamp(1.0, 2.0);
    return SizedBox(
      height: chartH + 48 * textScale,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          for (final d in report.perDay)
            Expanded(
              child: Semantics(
                label:
                    '${d.date.day}/${d.date.month}: ${l10n.reportCTaken} ${d.takenCount}',
                excludeSemantics: true,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    if (!dense || d.takenCount > 0)
                      Text(
                        '${d.takenCount}',
                        style: TextStyle(
                          fontSize: dense ? 9 : 14,
                          fontWeight: FontWeight.w700,
                          color: neu.textStrong,
                        ),
                      ),
                    const SizedBox(height: 2),
                    Container(
                      margin: EdgeInsets.symmetric(horizontal: dense ? 1 : 4),
                      height:
                          d.takenCount == 0 ? 3 : chartH * d.takenCount / maxV,
                      decoration: BoxDecoration(
                        color: d.takenCount == 0 ? neu.textMuted : neu.accent,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      dense
                          ? (d.date.day % 5 == 0 ? '${d.date.day}' : '')
                          : _WeekRow._weekdayLabel(d.date, locale),
                      maxLines: 1,
                      overflow: TextOverflow.clip,
                      style: TextStyle(
                        fontSize: dense ? 10 : 12,
                        color: neu.textMuted,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _DoctorView extends StatelessWidget {
  const _DoctorView({
    required this.report,
    required this.l10n,
    required this.onBack,
  });
  final AdherenceReport report;
  final AppLocalizations l10n;
  final VoidCallback onBack;

  @override
  Widget build(BuildContext context) {
    final dark = Theme.of(context).brightness == Brightness.dark;
    final fg = dark ? Colors.white : Colors.black;
    final bg = dark ? Colors.black : Colors.white;
    Widget row(String label, String value) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: TextStyle(fontSize: 24, color: fg)),
              Text(
                value,
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.w900,
                  color: fg,
                ),
              ),
            ],
          ),
        );
    return Container(
      color: bg,
      child: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Text(
            '${l10n.reportCPeriod}: ${report.days} ${l10n.reportCDays}',
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.w800,
              color: fg,
            ),
          ),
          Divider(thickness: 2, color: fg),
          row(l10n.reportCRate, _pct(report.adherenceRate, l10n)),
          row(l10n.reportCTaken, '${report.taken}'),
          row(l10n.reportCLate, '${report.late}'),
          row(l10n.reportCSkipped, '${report.skipped}'),
          row(l10n.reportCSnoozed, '${report.snoozed}'),
          row(
            l10n.reportCStreakNow,
            '${report.currentStreak} ${l10n.reportCDays}',
          ),
          row(
            l10n.reportCStreakBest,
            '${report.bestStreak} ${l10n.reportCDays}',
          ),
          Divider(thickness: 2, color: fg),
          Text(l10n.reportCNote, style: TextStyle(fontSize: 20, color: fg)),
          const SizedBox(height: 8),
          Text(
            l10n.reportCDoctorHint,
            style: TextStyle(fontSize: 18, color: fg),
          ),
          const SizedBox(height: 24),
          OutlinedButton(onPressed: onBack, child: Text(l10n.reportCBack)),
        ],
      ),
    );
  }
}

class _StreakCard extends StatelessWidget {
  const _StreakCard({required this.streak, required this.l10n});

  final int streak;
  final AppLocalizations l10n;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    return NeuCard(
      radius: 24,
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: neu.accentSoft,
            ),
            child: Icon(
              streak > 0 ? Icons.local_fire_department : Icons.bedtime_outlined,
              color: neu.accent,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              streak > 0
                  ? l10n.progressStreakDays(streak)
                  : l10n.progressStreakZero,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w800,
                color: neu.textStrong,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _WeekRow extends StatelessWidget {
  const _WeekRow({required this.takenDates});

  final Set<DateTime> takenDates;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final locale = Localizations.localeOf(context).toString();
    final today = DateTime.now();
    final days = List.generate(7, (i) {
      return DateTime(today.year, today.month, today.day)
          .subtract(Duration(days: 6 - i));
    });

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        for (final day in days)
          Column(
            children: [
              Text(
                _weekdayLabel(day, locale),
                style:
                    theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
              ),
              const SizedBox(height: 6),
              Container(
                width: 32,
                height: 32,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: takenDates.contains(day) ? neu.accent : neu.accentSoft,
                ),
                child: takenDates.contains(day)
                    ? const Icon(Icons.check, color: Colors.white, size: 18)
                    : Text(
                        '${day.day}',
                        style: theme.textTheme.bodySmall
                            ?.copyWith(color: neu.textMuted),
                      ),
              ),
            ],
          ),
      ],
    );
  }

  static String _weekdayLabel(DateTime day, String locale) {
    const viLabels = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'];
    if (locale.startsWith('vi')) return viLabels[day.weekday - 1];
    const enLabels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];
    return enLabels[day.weekday - 1];
  }
}
