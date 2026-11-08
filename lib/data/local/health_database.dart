import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/prescription.dart';

/// SQLite local cho hồ sơ sức khoẻ + nhắc thuốc (Day 27).
///
/// Dữ liệu **chỉ** nằm trên thiết bị. Không gửi server, không backup cloud.
/// Người dùng xoá được toàn bộ qua [deleteAll].
class HealthDatabase {
  HealthDatabase._();
  static final HealthDatabase instance = HealthDatabase._();

  Database? _db;

  Future<Database> get database async {
    if (_db != null) return _db!;
    _db = await _open();
    return _db!;
  }

  Future<Database> _open() async {
    final dbPath = await getDatabasesPath();
    return openDatabase(
      p.join(dbPath, 'health_profile.db'),
      version: 3,
      // Bật khoá ngoại để ON DELETE CASCADE thực sự có hiệu lực.
      onConfigure: (db) => db.execute('PRAGMA foreign_keys = ON'),
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) await _createPrescriptionTables(db);
        if (oldVersion < 3) {
          // v3: trạng thái liều (taken|skipped|snoozed|late) + đếm số viên còn lại.
          await db.execute(
              "ALTER TABLE adherence_log ADD COLUMN status TEXT NOT NULL DEFAULT 'taken'");
          await db.execute(
              'ALTER TABLE medication_reminders ADD COLUMN pills_left INTEGER');
          await db.execute(
              'ALTER TABLE medication_reminders ADD COLUMN pills_per_dose INTEGER NOT NULL DEFAULT 1');
        }
      },
      onCreate: (db, version) async {
        await _createPrescriptionTables(db);
        await db.execute('''
          CREATE TABLE health_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            allergies TEXT NOT NULL DEFAULT '',
            conditions TEXT NOT NULL DEFAULT '',
            current_medications TEXT NOT NULL DEFAULT '',
            consent_given_at TEXT
          )
        ''');
        await db.execute('''
          CREATE TABLE medication_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name TEXT NOT NULL,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            dosage TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            days_of_week TEXT NOT NULL DEFAULT '1,2,3,4,5,6,7',
            pills_left INTEGER,
            pills_per_dose INTEGER NOT NULL DEFAULT 1
          )
        ''');
        await db.execute('''
          CREATE TABLE adherence_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL,
            taken_at TEXT NOT NULL,
            skipped INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'taken',
            FOREIGN KEY (reminder_id) REFERENCES medication_reminders(id)
              ON DELETE CASCADE
          )
        ''');
      },
    );
  }

  /// v2: đơn thuốc đã xác nhận. CHỈ lưu dòng thuốc đã tách — không text OCR, không ảnh.
  static Future<void> _createPrescriptionTables(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS prescriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        diagnosis TEXT
      )
    ''');
    await db.execute('''
      CREATE TABLE IF NOT EXISTS prescription_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prescription_id INTEGER NOT NULL,
        medicine_name TEXT NOT NULL,
        dose_text TEXT NOT NULL DEFAULT '',
        times_per_day INTEGER,
        slots TEXT NOT NULL DEFAULT '',
        meal_relation TEXT NOT NULL DEFAULT 'none',
        FOREIGN KEY (prescription_id) REFERENCES prescriptions(id)
          ON DELETE CASCADE
      )
    ''');
  }

  // --- Prescriptions (v2) ---------------------------------------------------

  Future<SavedPrescription> insertPrescription(SavedPrescription rx) async {
    final db = await database;
    return db.transaction((txn) async {
      final id = await txn.insert('prescriptions', rx.toMap()..remove('id'));
      final items = <PrescriptionItemRecord>[];
      for (final item in rx.items) {
        final map = item.toMap()
          ..remove('id')
          ..['prescription_id'] = id;
        final itemId = await txn.insert('prescription_items', map);
        items.add(PrescriptionItemRecord.fromMap({...map, 'id': itemId}));
      }
      return SavedPrescription(
        id: id,
        createdAt: rx.createdAt,
        diagnosis: rx.diagnosis,
        items: items,
      );
    });
  }

  Future<List<SavedPrescription>> getPrescriptions() async {
    final db = await database;
    final rows = await db.query('prescriptions', orderBy: 'created_at DESC');
    final itemRows = await db.query('prescription_items', orderBy: 'id');
    final byRx = <int, List<PrescriptionItemRecord>>{};
    for (final r in itemRows) {
      final item = PrescriptionItemRecord.fromMap(r);
      byRx.putIfAbsent(item.prescriptionId ?? -1, () => []).add(item);
    }
    return [
      for (final r in rows)
        SavedPrescription.fromMap(r, items: byRx[r['id']] ?? const []),
    ];
  }

  Future<void> deletePrescription(int id) async {
    final db = await database;
    await db.delete('prescription_items',
        where: 'prescription_id = ?', whereArgs: [id]);
    await db.delete('prescriptions', where: 'id = ?', whereArgs: [id]);
  }

  // --- Health Profile -------------------------------------------------------

  Future<HealthProfile?> getProfile() async {
    final db = await database;
    final rows = await db.query('health_profile', limit: 1);
    if (rows.isEmpty) return null;
    return HealthProfile.fromMap(rows.first);
  }

  Future<HealthProfile> saveProfile(HealthProfile profile) async {
    final db = await database;
    final existing = await getProfile();
    if (existing != null) {
      await db.update(
        'health_profile',
        profile.toMap()..remove('id'),
        where: 'id = ?',
        whereArgs: [existing.id],
      );
      return profile.copyWith(id: existing.id);
    } else {
      final id = await db.insert('health_profile', profile.toMap());
      return profile.copyWith(id: id);
    }
  }

  Future<void> deleteProfile() async {
    final db = await database;
    await db.delete('health_profile');
  }

  // --- Medication Reminders -------------------------------------------------

  Future<List<MedicationReminder>> getReminders() async {
    final db = await database;
    final rows =
        await db.query('medication_reminders', orderBy: 'hour, minute');
    return rows.map(MedicationReminder.fromMap).toList();
  }

  Future<MedicationReminder> insertReminder(MedicationReminder r) async {
    final db = await database;
    final id = await db.insert('medication_reminders', r.toMap());
    return r.copyWith(id: id);
  }

  Future<void> updateReminder(MedicationReminder r) async {
    final db = await database;
    await db.update(
      'medication_reminders',
      r.toMap()..remove('id'),
      where: 'id = ?',
      whereArgs: [r.id],
    );
  }

  Future<void> deleteReminder(int id) async {
    final db = await database;
    await db.delete('medication_reminders', where: 'id = ?', whereArgs: [id]);
  }

  // --- Adherence Log --------------------------------------------------------

  Future<void> logAdherence(AdherenceRecord record) async {
    final db = await database;
    await db.insert('adherence_log', record.toMap());
  }

  Future<List<AdherenceRecord>> getAdherenceLogs(int reminderId,
      {int limit = 30}) async {
    final db = await database;
    final rows = await db.query(
      'adherence_log',
      where: 'reminder_id = ?',
      whereArgs: [reminderId],
      orderBy: 'taken_at DESC',
      limit: limit,
    );
    return rows.map(AdherenceRecord.fromMap).toList();
  }

  /// Tất cả lượt ghi nhận (mọi nhắc thuốc) từ [since] trở lại — dùng để tính
  /// streak/lịch tuân thủ tổng quát. Nhóm theo ngày/nhắc thuốc do bên gọi tự xử lý
  /// (tập dữ liệu của một app cá nhân rất nhỏ, không cần gộp bằng SQL).
  Future<List<AdherenceRecord>> getAdherenceSince(DateTime since) async {
    final db = await database;
    final rows = await db.query(
      'adherence_log',
      where: 'taken_at >= ?',
      whereArgs: [since.toIso8601String()],
      orderBy: 'taken_at DESC',
    );
    return rows.map(AdherenceRecord.fromMap).toList();
  }

  // --- Xoá toàn bộ (quyền được quên) ----------------------------------------

  Future<void> deleteAll() async {
    final db = await database;
    await db.delete('prescription_items');
    await db.delete('prescriptions');
    await db.delete('adherence_log');
    await db.delete('medication_reminders');
    await db.delete('health_profile');
  }

  Future<void> close() async {
    await _db?.close();
    _db = null;
  }
}
