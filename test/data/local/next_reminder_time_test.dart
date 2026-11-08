import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/local/notification_service.dart';

void main() {
  const all = [1, 2, 3, 4, 5, 6, 7];

  test('giờ nhắc còn ở phía sau hôm nay -> nhắc hôm nay', () {
    final now = DateTime(2026, 10, 28, 7, 0); // Thứ Tư
    expect(nextReminderTime(now, 8, 30, all), DateTime(2026, 10, 28, 8, 30));
  });

  test('giờ nhắc đã qua -> nhắc ngày mai', () {
    final now = DateTime(2026, 10, 28, 9, 0);
    expect(nextReminderTime(now, 8, 30, all), DateTime(2026, 10, 29, 8, 30));
  });

  test('đúng giờ nhắc -> sang lần kế (không bao giờ bằng hiện tại)', () {
    final now = DateTime(2026, 10, 28, 8, 30);
    expect(nextReminderTime(now, 8, 30, all), DateTime(2026, 10, 29, 8, 30));
  });

  test('chỉ chọn Thứ Hai: từ Thứ Tư nhảy tới Thứ Hai tuần sau', () {
    final now = DateTime(2026, 10, 28, 7, 0); // Thứ Tư (weekday 3)
    final next = nextReminderTime(now, 8, 0, [1]);
    expect(next, DateTime(2026, 11, 2, 8, 0));
    expect(next.weekday, DateTime.monday);
  });

  test('cùng ngày trong tuần nhưng giờ đã qua -> tuần sau', () {
    final now = DateTime(2026, 10, 28, 9, 0); // Thứ Tư
    expect(nextReminderTime(now, 8, 0, [3]), DateTime(2026, 11, 4, 8, 0));
  });

  test('danh sách ngày rỗng coi như mọi ngày', () {
    final now = DateTime(2026, 10, 28, 9, 0);
    expect(nextReminderTime(now, 8, 0, const []), DateTime(2026, 10, 29, 8, 0));
  });

  test('qua ranh giới tháng/năm', () {
    final now = DateTime(2026, 12, 31, 23, 0);
    expect(nextReminderTime(now, 6, 0, all), DateTime(2027, 1, 1, 6, 0));
  });

  // Plugin chuyển tên múi giờ sang Java `ZoneId.of(...)`; tên lạ (vd. "device") làm
  // đặt lịch ném ZoneRulesException (bắt được khi chạy máy ảo, không bắt được ở test cũ).
  group('zoneIdForOffset — tên Java ZoneId hiểu được', () {
    test('UTC', () => expect(zoneIdForOffset(Duration.zero), 'UTC'));
    test('Việt Nam +7', () {
      expect(zoneIdForOffset(const Duration(hours: 7)), 'UTC+07:00');
    });
    test('lệch âm', () {
      expect(zoneIdForOffset(const Duration(hours: -5)), 'UTC-05:00');
    });
    test('lệch có phút (Ấn Độ +5:30, Newfoundland -3:30)', () {
      expect(
          zoneIdForOffset(const Duration(hours: 5, minutes: 30)), 'UTC+05:30');
      expect(
        zoneIdForOffset(const Duration(hours: -3, minutes: -30)),
        'UTC-03:30',
      );
    });
    test('không bao giờ là "device"', () {
      expect(zoneIdForOffset(const Duration(hours: 7)), isNot('device'));
    });
  });
}
