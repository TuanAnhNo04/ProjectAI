import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/health_profile.dart';

MedicationReminder _r(String n, int? left, [int per = 1]) => MedicationReminder(
    medicineName: n, hour: 8, minute: 0, pillsLeft: left, pillsPerDose: per);

void main() {
  test('lowStockOf flags <= 3 doses left, ignores untracked', () {
    final low = lowStockOf([
      _r('A', 3),
      _r('B', 4),
      _r('C', null),
      _r('D', 6, 2),
      _r('E', 7, 2),
    ]);
    expect(low.map((r) => r.medicineName), ['A', 'D']);
  });
}
