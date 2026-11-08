import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/health_profile.dart';

AdherenceRecord rec(int id, DateTime at, String status) =>
    AdherenceRecord(reminderId: id, takenAt: at, status: status);

void main() {
  final now = DateTime(2026, 10, 29, 15, 0);

  test('taken and late count as taken; skipped/snoozed do not', () {
    final logs = [
      rec(1, now, kDoseTaken),
      rec(2, now, kDoseLate),
      rec(3, now, kDoseSkipped),
      rec(4, now, kDoseSnoozed),
    ];
    expect(takenIdsToday(logs, now), {1, 2});
    expect(skippedIdsToday(logs, now), {3});
  });

  test('streak counts only taken/late days', () {
    final logs = [
      rec(1, now, kDoseSkipped),
      rec(1, now.subtract(const Duration(days: 1)), kDoseTaken),
      rec(1, now.subtract(const Duration(days: 2)), kDoseLate),
      rec(1, now.subtract(const Duration(days: 3)), kDoseSnoozed),
    ];
    expect(adherenceStreakOf(logs, now), 2);
  });

  test('taken more than 60 min after due becomes late', () {
    const r = MedicationReminder(medicineName: 'Fake', hour: 13, minute: 0);
    expect(doseStatusForTaken(r, now), kDoseLate);
    expect(doseStatusForTaken(r, DateTime(2026, 10, 29, 13, 30)), kDoseTaken);
  });
}
