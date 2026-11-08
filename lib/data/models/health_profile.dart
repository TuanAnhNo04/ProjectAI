/// Hồ sơ sức khoẻ cá nhân — Day 27.
///
/// Lưu local (SQLite), **không gửi server** trừ khi người dùng đồng ý rõ ràng.
/// Agent dùng danh sách [currentMedications] để cảnh báo tương tác.
class HealthProfile {
  const HealthProfile({
    this.id,
    this.allergies = const [],
    this.conditions = const [],
    this.currentMedications = const [],
    this.consentGivenAt,
  });

  final int? id;

  /// Dị ứng đã biết (tên hoạt chất / nhóm thuốc).
  final List<String> allergies;

  /// Bệnh nền (tiểu đường, cao huyết áp, ...).
  final List<String> conditions;

  /// Thuốc đang dùng (tên thương hiệu hoặc hoạt chất).
  final List<String> currentMedications;

  /// Thời điểm người dùng đồng ý lưu hồ sơ; `null` = chưa consent.
  final DateTime? consentGivenAt;

  bool get hasConsent => consentGivenAt != null;

  HealthProfile copyWith({
    int? id,
    List<String>? allergies,
    List<String>? conditions,
    List<String>? currentMedications,
    DateTime? consentGivenAt,
  }) =>
      HealthProfile(
        id: id ?? this.id,
        allergies: allergies ?? this.allergies,
        conditions: conditions ?? this.conditions,
        currentMedications: currentMedications ?? this.currentMedications,
        consentGivenAt: consentGivenAt ?? this.consentGivenAt,
      );

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'allergies': allergies.join('|'),
        'conditions': conditions.join('|'),
        'current_medications': currentMedications.join('|'),
        'consent_given_at': consentGivenAt?.toIso8601String(),
      };

  factory HealthProfile.fromMap(Map<String, dynamic> map) => HealthProfile(
        id: map['id'] as int?,
        allergies: _splitPipe(map['allergies']),
        conditions: _splitPipe(map['conditions']),
        currentMedications: _splitPipe(map['current_medications']),
        consentGivenAt: map['consent_given_at'] != null
            ? DateTime.tryParse(map['consent_given_at'] as String)
            : null,
      );

  static List<String> _splitPipe(Object? value) {
    if (value is! String || value.isEmpty) return const [];
    return value.split('|').where((s) => s.isNotEmpty).toList();
  }
}

/// Lịch nhắc uống thuốc — Day 27.
class MedicationReminder {
  const MedicationReminder({
    this.id,
    required this.medicineName,
    required this.hour,
    required this.minute,
    this.dosage,
    this.isActive = true,
    this.daysOfWeek = const [1, 2, 3, 4, 5, 6, 7],
    this.pillsLeft,
    this.pillsPerDose = 1,
  });

  final int? id;
  final String medicineName;

  /// Số viên còn lại (`null` = không theo dõi). Trừ [pillsPerDose] mỗi lần uống.
  final int? pillsLeft;

  /// Số viên mỗi liều (≥1).
  final int pillsPerDose;

  /// Giờ nhắc (0-23).
  final int hour;

  /// Phút nhắc (0-59).
  final int minute;

  /// Liều dùng (vd: "1 viên", "5ml").
  final String? dosage;

  /// Bật/tắt nhắc.
  final bool isActive;

  /// Ngày trong tuần (1=Thứ 2, 7=CN). Mặc định: mỗi ngày.
  final List<int> daysOfWeek;

  String get timeDisplay =>
      '${hour.toString().padLeft(2, '0')}:${minute.toString().padLeft(2, '0')}';

  MedicationReminder copyWith({
    int? id,
    String? medicineName,
    int? hour,
    int? minute,
    String? Function()? dosage,
    bool? isActive,
    List<int>? daysOfWeek,
    int? Function()? pillsLeft,
    int? pillsPerDose,
  }) =>
      MedicationReminder(
        id: id ?? this.id,
        medicineName: medicineName ?? this.medicineName,
        hour: hour ?? this.hour,
        minute: minute ?? this.minute,
        dosage: dosage != null ? dosage() : this.dosage,
        isActive: isActive ?? this.isActive,
        daysOfWeek: daysOfWeek ?? this.daysOfWeek,
        pillsLeft: pillsLeft != null ? pillsLeft() : this.pillsLeft,
        pillsPerDose: pillsPerDose ?? this.pillsPerDose,
      );

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'medicine_name': medicineName,
        'hour': hour,
        'minute': minute,
        'dosage': dosage,
        'is_active': isActive ? 1 : 0,
        'days_of_week': daysOfWeek.join(','),
        'pills_left': pillsLeft,
        'pills_per_dose': pillsPerDose,
      };

  factory MedicationReminder.fromMap(Map<String, dynamic> map) =>
      MedicationReminder(
        id: map['id'] as int?,
        medicineName: map['medicine_name'] as String? ?? '',
        hour: map['hour'] as int? ?? 8,
        minute: map['minute'] as int? ?? 0,
        dosage: map['dosage'] as String?,
        isActive: (map['is_active'] as int?) == 1,
        daysOfWeek: _parseDays(map['days_of_week']),
        pillsLeft: map['pills_left'] as int?,
        pillsPerDose: (map['pills_per_dose'] as int?) ?? 1,
      );

  static List<int> _parseDays(Object? value) {
    if (value is! String || value.isEmpty) return const [1, 2, 3, 4, 5, 6, 7];
    return value.split(',').map((s) => int.tryParse(s) ?? 1).toList();
  }
}

/// Trạng thái một liều trong adherence log. `taken` và `late` đều tính là đã uống
/// (late = uống trễ); `skipped` = bỏ qua; `snoozed` = hoãn (chưa uống).
const String kDoseTaken = 'taken';
const String kDoseSkipped = 'skipped';
const String kDoseSnoozed = 'snoozed';
const String kDoseLate = 'late';

/// Ghi nhận một lần uống thuốc (adherence log).
class AdherenceRecord {
  const AdherenceRecord({
    this.id,
    required this.reminderId,
    required this.takenAt,
    this.skipped = false,
    this.status = kDoseTaken,
  });

  final int? id;
  final int reminderId;
  final DateTime takenAt;

  /// Giữ để tương thích: `true` khi [status] là `skipped`.
  final bool skipped;

  /// `taken` | `skipped` | `snoozed` | `late` — xem hằng `kDose*`.
  final String status;

  Map<String, dynamic> toMap() => {
        if (id != null) 'id': id,
        'reminder_id': reminderId,
        'taken_at': takenAt.toIso8601String(),
        'skipped': (skipped || status == kDoseSkipped) ? 1 : 0,
        'status': status,
      };

  factory AdherenceRecord.fromMap(Map<String, dynamic> map) {
    final skipped = (map['skipped'] as int?) == 1;
    return AdherenceRecord(
      id: map['id'] as int?,
      reminderId: map['reminder_id'] as int? ?? 0,
      takenAt:
          DateTime.tryParse(map['taken_at'] as String? ?? '') ?? DateTime.now(),
      skipped: skipped,
      status:
          (map['status'] as String?) ?? (skipped ? kDoseSkipped : kDoseTaken),
    );
  }
}
