import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/domain/adherence_report.dart';

AdherenceRecord rec(DateTime t, [String status = kDoseTaken]) =>
    AdherenceRecord(reminderId: 1, takenAt: t, status: status);

void main() {
  final now = DateTime(2026, 10, 29, 12);

  test('empty list gives null rate and zero streaks', () {
    final r = buildReport([], now: now, days: 7);
    expect(r.adherenceRate, isNull);
    expect(r.isEmpty, isTrue);
    expect(r.perDay.length, 7);
    expect(r.currentStreak, 0);
    expect(r.bestStreak, 0);
  });

  test('counts, rate ignores snoozed, late counts as taken', () {
    final r = buildReport([
      rec(DateTime(2026, 10, 29, 8)),
      rec(DateTime(2026, 10, 29, 9), kDoseLate),
      rec(DateTime(2026, 10, 28, 8), kDoseSkipped),
      rec(DateTime(2026, 10, 28, 9), kDoseSnoozed),
    ], now: now, days: 7);
    expect(r.taken, 1);
    expect(r.late, 1);
    expect(r.skipped, 1);
    expect(r.snoozed, 1);
    expect(r.adherenceRate, closeTo(2 / 3, 1e-9));
    expect(r.perDay.last.takenCount, 2);
  });

  test('old-schema skipped flag with default status counts as skipped', () {
    final r = buildReport([
      AdherenceRecord(
          reminderId: 1, takenAt: DateTime(2026, 10, 29, 8), skipped: true),
    ], now: now, days: 7);
    expect(r.skipped, 1);
    expect(r.taken, 0);
  });

  test('window boundaries', () {
    final r = buildReport([
      rec(DateTime(2026, 10, 23, 0, 1)), // first day of 7-day window
      rec(DateTime(2026, 10, 22, 23, 59)), // outside
      rec(DateTime(2026, 10, 30, 8)), // future, outside
    ], now: now, days: 7);
    expect(r.taken, 1);
    expect(r.perDay.first.takenCount, 1);
  });

  test('streaks: today pending keeps streak, gap resets', () {
    final r = buildReport([
      rec(DateTime(2026, 10, 28)),
      rec(DateTime(2026, 10, 27)),
      rec(DateTime(2026, 10, 24)),
      rec(DateTime(2026, 10, 25)),
      rec(DateTime(2026, 10, 26)),
    ], now: now, days: 7);
    expect(r.currentStreak, 5);
    expect(r.bestStreak, 5);
    final g = buildReport([
      rec(DateTime(2026, 10, 29)),
      rec(DateTime(2026, 10, 26)),
      rec(DateTime(2026, 10, 25)),
    ], now: now, days: 30);
    expect(g.currentStreak, 1);
    expect(g.bestStreak, 2);
  });
}
