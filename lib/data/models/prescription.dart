/// Đơn thuốc đã lưu local — CHỈ gồm các dòng thuốc đã tách và người dùng xác
/// nhận. Không bao giờ lưu text OCR thô hay ảnh đơn thuốc.
class SavedPrescription {
  const SavedPrescription({
    this.id,
    required this.createdAt,
    this.diagnosis,
    this.items = const [],
  });

  final int? id;
  final DateTime createdAt;

  /// Chỉ lưu khi người dùng chọn thêm chẩn đoán vào hồ sơ (đã consent).
  final String? diagnosis;
  final List<PrescriptionItemRecord> items;

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'created_at': createdAt.toIso8601String(),
        'diagnosis': diagnosis,
      };

  factory SavedPrescription.fromMap(
    Map<String, dynamic> map, {
    List<PrescriptionItemRecord> items = const [],
  }) =>
      SavedPrescription(
        id: map['id'] as int?,
        createdAt: DateTime.tryParse(map['created_at'] as String? ?? '') ??
            DateTime.fromMillisecondsSinceEpoch(0),
        diagnosis: map['diagnosis'] as String?,
        items: items,
      );
}

class PrescriptionItemRecord {
  const PrescriptionItemRecord({
    this.id,
    this.prescriptionId,
    required this.medicineName,
    this.doseText = '',
    this.timesPerDay,
    this.slots = const [],
    this.mealRelation = 'none',
    this.pillsTotal,
    this.pillsPerDose = 1,
  });

  /// Tổng số viên được cấp (chỉ dùng khi tạo lịch nhắc, KHÔNG lưu vào DB đơn).
  final int? pillsTotal;
  final int pillsPerDose;

  final int? id;
  final int? prescriptionId;
  final String medicineName;
  final String doseText;
  final int? timesPerDay;

  /// Giờ nhắc dạng 'HH:mm'.
  final List<String> slots;

  /// Tên enum `MealRelation` (none/before/after/during).
  final String mealRelation;

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        if (prescriptionId != null) 'prescription_id': prescriptionId,
        'medicine_name': medicineName,
        'dose_text': doseText,
        'times_per_day': timesPerDay,
        'slots': slots.join(','),
        'meal_relation': mealRelation,
      };

  factory PrescriptionItemRecord.fromMap(Map<String, dynamic> map) =>
      PrescriptionItemRecord(
        id: map['id'] as int?,
        prescriptionId: map['prescription_id'] as int?,
        medicineName: map['medicine_name'] as String? ?? '',
        doseText: map['dose_text'] as String? ?? '',
        timesPerDay: map['times_per_day'] as int?,
        slots: (map['slots'] as String? ?? '')
            .split(',')
            .where((s) => s.isNotEmpty)
            .toList(),
        mealRelation: map['meal_relation'] as String? ?? 'none',
      );
}
