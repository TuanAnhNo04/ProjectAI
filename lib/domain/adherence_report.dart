import 'package:image_processing_app/data/models/health_profile.dart';

/// Số liệu một ngày trong báo cáo.
class DayStat {
  const DayStat({
    required this.date,
    required this.takenCount,
    required this.skippedCount,
  });

  final DateTime date;
  final int takenCount; // taken + late
  final int skippedCount;
}

/// Báo cáo tuân thủ uống thuốc (thuần Dart, không phụ thuộc UI).
class AdherenceReport {
  const AdherenceReport({
    required this.days,
    required this.taken,
    required this.late,
    required this.skipped,
    required this.snoozed,
    required this.adherenceRate,
    required this.perDay,
    required this.currentStreak,
    required this.bestStreak,
  });

  final int days;
  final int taken;
  final int late;
  final int skipped;
  final int snoozed;

  /// (taken+late)/(taken+late+skipped); null khi không có dữ liệu.
  final double? adherenceRate;
  final List<DayStat> perDay;
  final int currentStreak;
  final int bestStreak;

  int get takenTotal => taken + late;
  bool get isEmpty => taken + late + skipped + snoozed == 0;
}

DateTime _day(DateTime d) => DateTime(d.year, d.month, d.day);

AdherenceReport buildReport(
  List<AdherenceRecord> logs, {
  required DateTime now,
  required int days,
}) {
  final today = _day(now);
  final start = DateTime(today.year, today.month, today.day - (days - 1));
  final takenBy = <DateTime, int>{};
  final skippedBy = <DateTime, int>{};
  var taken = 0, late = 0, skipped = 0, snoozed = 0;

  for (final r in logs) {
    final d = _day(r.takenAt);
    if (d.isBefore(start) || d.isAfter(today)) continue;
    final status =
        r.skipped && r.status == kDoseTaken ? kDoseSkipped : r.status;
    switch (status) {
      case kDoseSkipped:
        skipped++;
        skippedBy[d] = (skippedBy[d] ?? 0) + 1;
      case kDoseSnoozed:
        snoozed++;
      case kDoseLate:
        late++;
        takenBy[d] = (takenBy[d] ?? 0) + 1;
      default:
        taken++;
        takenBy[d] = (takenBy[d] ?? 0) + 1;
    }
  }

  final perDay = <DayStat>[
    for (var i = 0; i < days; i++)
      () {
        final d = DateTime(start.year, start.month, start.day + i);
        return DayStat(
          date: d,
          takenCount: takenBy[d] ?? 0,
          skippedCount: skippedBy[d] ?? 0,
        );
      }(),
  ];

  var best = 0, run = 0;
  for (final s in perDay) {
    if (s.takenCount > 0) {
      run++;
      if (run > best) best = run;
    } else {
      run = 0;
    }
  }
  // Streak hiện tại: tính lùi từ hôm nay; hôm nay chưa uống thì không phá chuỗi.
  var current = 0;
  var i = perDay.length - 1;
  if (i >= 0 && perDay[i].takenCount == 0) i--;
  while (i >= 0 && perDay[i].takenCount > 0) {
    current++;
    i--;
  }

  final denom = taken + late + skipped;
  return AdherenceReport(
    days: days,
    taken: taken,
    late: late,
    skipped: skipped,
    snoozed: snoozed,
    adherenceRate: denom == 0 ? null : (taken + late) / denom,
    perDay: perDay,
    currentStreak: current,
    bestStreak: best,
  );
}
