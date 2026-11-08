import 'dart:io';

import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/data/latest.dart' as tzdata;
import 'package:timezone/timezone.dart' as tz;

import 'package:image_processing_app/data/models/health_profile.dart';

/// Thời điểm nhắc kế tiếp (giờ địa phương) cho `hour:minute` vào một trong
/// [weekdays] (1=Thứ Hai … 7=Chủ Nhật, như `DateTime.weekday`), tính từ [now].
///
/// Hàm thuần để test được; trả về một lần nhắc **sau** [now] (không bao giờ bằng).
DateTime nextReminderTime(
  DateTime now,
  int hour,
  int minute,
  List<int> weekdays,
) {
  final days = weekdays.isEmpty ? const [1, 2, 3, 4, 5, 6, 7] : weekdays;
  for (var offset = 0; offset <= 7; offset++) {
    final day = DateTime(now.year, now.month, now.day + offset, hour, minute);
    if (day.isAfter(now) && days.contains(day.weekday)) return day;
  }
  // Không thể tới đây vì days không rỗng và ta duyệt đủ 8 ngày.
  return DateTime(now.year, now.month, now.day + 1, hour, minute);
}

/// Tên múi giờ mà Java `ZoneId.of` hiểu được, dựng từ độ lệch so với UTC
/// (`UTC`, `UTC+07:00`, `UTC-03:30`). Hàm thuần để test được.
String zoneIdForOffset(Duration offset) {
  if (offset == Duration.zero) return 'UTC';
  final sign = offset.isNegative ? '-' : '+';
  final abs = offset.abs();
  final hh = abs.inHours.toString().padLeft(2, '0');
  final mm = (abs.inMinutes % 60).toString().padLeft(2, '0');
  return 'UTC$sign$hh:$mm';
}

/// Dịch vụ nhắc uống thuốc qua local notification (Day 27, nhắc đúng giờ ở Day 30).
///
/// **Không gửi dữ liệu ra ngoài.** Notification chạy hoàn toàn local —
/// không dùng FCM hay push server.
class NotificationService {
  NotificationService._();
  static final NotificationService instance = NotificationService._();

  final _plugin = FlutterLocalNotificationsPlugin();
  bool _initialized = false;

  static const actionTaken = 'dose_taken';
  static const actionSnooze = 'dose_snooze';
  static const _snoozeBase = 900000000;

  /// Callback đăng ký từ providers: (reminderId, actionId). Giới hạn: nút hành động
  /// chỉ ghi liều khi tiến trình app còn sống (không có isolate nền); app đã tắt thì chỉ mở app.
  Future<void> Function(int reminderId, String action)? onDoseAction;

  void _onResponse(NotificationResponse r) {
    final id = int.tryParse(r.payload ?? '');
    final action = r.actionId;
    if (id == null || action == null || action.isEmpty) return;
    onDoseAction?.call(id, action);
  }

  /// Gọi 1 lần từ `main.dart`. Trả `false` nếu hệ thống không hỗ trợ.
  Future<bool> init() async {
    if (_initialized) return true;

    tzdata.initializeTimeZones();
    _useDeviceOffset();

    const android = AndroidInitializationSettings('@mipmap/ic_launcher');
    const ios = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );
    const settings = InitializationSettings(android: android, iOS: ios);

    final ok = await _plugin.initialize(settings,
            onDidReceiveNotificationResponse: _onResponse) ??
        false;
    _initialized = ok;
    return ok;
  }

  /// Đặt múi giờ "local" của gói `timezone` theo độ lệch hiện tại của thiết bị.
  ///
  /// Không thêm gói đọc tên múi giờ: độ lệch cố định đủ cho giờ nhắc theo đồng
  /// hồ máy. Lịch được dựng lại mỗi lần mở app ([rescheduleAll]) nên lệch do
  /// giờ mùa hè chỉ kéo dài tới lần mở kế tiếp.
  void _useDeviceOffset() {
    final offset = DateTime.now().timeZoneOffset;
    tz.setLocalLocation(
      tz.Location(
        // ⚠️ Tên này được plugin chuyển sang Java `ZoneId.of(...)`. Tên tự đặt
        // (ví dụ "device") làm `zonedSchedule` ném `ZoneRulesException: Unknown
        // time-zone ID` -> lịch nhắc KHÔNG được đặt. Dùng tên Java hiểu được.
        zoneIdForOffset(offset),
        [tz.minTime],
        [0],
        [
          tz.TimeZone(
            offset.inMilliseconds,
            isDst: false,
            abbreviation: 'LOCAL',
          ),
        ],
      ),
    );
  }

  /// Lập lịch nhắc cho một [MedicationReminder]: đúng `hour:minute` đã chọn,
  /// lặp mỗi tuần vào các ngày trong `daysOfWeek`.
  Future<void> scheduleReminder(MedicationReminder reminder) async {
    if (!_initialized || reminder.id == null || !reminder.isActive) return;

    const title = 'Nhắc uống thuốc';
    final body = reminder.dosage != null
        ? '${reminder.medicineName} — ${reminder.dosage}'
        : reminder.medicineName;

    const androidDetails = AndroidNotificationDetails(
      'med_reminder',
      'Nhắc uống thuốc',
      channelDescription: 'Thông báo nhắc uống thuốc theo lịch.',
      importance: Importance.high,
      priority: Priority.high,
      actions: [
        AndroidNotificationAction(actionTaken, 'Đã uống'),
        AndroidNotificationAction(actionSnooze, 'Hoãn 15 phút'),
      ],
    );
    const details = NotificationDetails(android: androidDetails);

    // Mỗi ngày trong tuần một thông báo riêng (id = reminderId * 10 + weekday)
    // để chọn được ngày; lặp theo tuần bằng dayOfWeekAndTime.
    _useDeviceOffset();
    final mode = await _scheduleMode();
    final now = DateTime.now();
    final days = reminder.daysOfWeek.isEmpty
        ? const [1, 2, 3, 4, 5, 6, 7]
        : reminder.daysOfWeek;
    for (final weekday in days) {
      final first = nextReminderTime(
        now,
        reminder.hour,
        reminder.minute,
        [weekday],
      );
      await _plugin.zonedSchedule(
        _notificationId(reminder.id!, weekday),
        title,
        body,
        tz.TZDateTime.from(first, tz.local),
        details,
        androidScheduleMode: mode,
        uiLocalNotificationDateInterpretation:
            UILocalNotificationDateInterpretation.absoluteTime,
        matchDateTimeComponents: DateTimeComponents.dayOfWeekAndTime,
        payload: '${reminder.id}',
      );
    }
  }

  /// Hoãn: một thông báo một-lần sau [minutes] phút cho [reminder].
  Future<void> snooze(MedicationReminder reminder, {int minutes = 15}) async {
    if (!_initialized || reminder.id == null) return;
    const androidDetails = AndroidNotificationDetails(
      'med_reminder',
      'Nhắc uống thuốc',
      channelDescription: 'Thông báo nhắc uống thuốc theo lịch.',
      importance: Importance.high,
      priority: Priority.high,
      actions: [
        AndroidNotificationAction(actionTaken, 'Đã uống'),
        AndroidNotificationAction(actionSnooze, 'Hoãn 15 phút'),
      ],
    );
    _useDeviceOffset();
    final mode = await _scheduleMode();
    await _plugin.zonedSchedule(
      _snoozeBase + reminder.id!,
      'Nhắc uống thuốc',
      reminder.dosage != null
          ? '${reminder.medicineName} — ${reminder.dosage}'
          : reminder.medicineName,
      tz.TZDateTime.now(tz.local).add(Duration(minutes: minutes)),
      const NotificationDetails(android: androidDetails),
      androidScheduleMode: mode,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      payload: '${reminder.id}',
    );
  }

  /// Dựng lại toàn bộ lịch (gọi mỗi lần mở app): đồng bộ với độ lệch múi giờ mới
  /// và đảm bảo lịch khớp dữ liệu đã lưu.
  Future<void> rescheduleAll(List<MedicationReminder> reminders) async {
    if (!_initialized) return;
    await _plugin.cancelAll();
    for (final r in reminders) {
      await scheduleReminder(r);
    }
  }

  /// Báo thức chính xác chỉ khi được phép; nếu không thì rơi về không-chính-xác
  /// (lệch tối đa vài phút) thay vì không nhắc.
  Future<AndroidScheduleMode> _scheduleMode() async {
    if (!Platform.isAndroid) return AndroidScheduleMode.exactAllowWhileIdle;
    try {
      final android = _plugin.resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin>();
      final canExact = await android?.canScheduleExactNotifications() ?? false;
      return canExact
          ? AndroidScheduleMode.exactAllowWhileIdle
          : AndroidScheduleMode.inexactAllowWhileIdle;
    } on Object {
      return AndroidScheduleMode.inexactAllowWhileIdle;
    }
  }

  static int _notificationId(int reminderId, int weekday) =>
      reminderId * 10 + weekday;

  /// Huỷ nhắc cho một reminder cụ thể (mọi ngày trong tuần).
  Future<void> cancelReminder(int reminderId) async {
    for (var weekday = 1; weekday <= 7; weekday++) {
      await _plugin.cancel(_notificationId(reminderId, weekday));
    }
    await _plugin.cancel(_snoozeBase + reminderId);
  }

  /// Huỷ tất cả notification (dùng khi đăng xuất hoặc xoá hồ sơ).
  Future<void> cancelAll() async {
    await _plugin.cancelAll();
  }

  /// Yêu cầu quyền thông báo (Android 13+) và báo thức chính xác. Không bao giờ
  /// ném: thiếu quyền thì nhắc vẫn được lưu, chỉ là thông báo không hiện/lệch giờ.
  Future<bool> requestPermission() async {
    if (!_initialized) return false;
    try {
      if (Platform.isAndroid) {
        final android = _plugin.resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>();
        final granted = await android?.requestNotificationsPermission();
        await android?.requestExactAlarmsPermission();
        return granted ?? false;
      }
      return true;
    } on Object {
      return false;
    }
  }
}
