// Mọi đơn thuốc ở đây là DỮ LIỆU GIẢ (tên/SĐT/mã thẻ bịa, không phải người thật).
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/models/medical_facility.dart';
import 'package:image_processing_app/domain/prescription_parser.dart';

void main() {
  test('foldVietnamese giữ nguyên độ dài và bỏ dấu đúng', () {
    const s = 'Sáng tối Đường ữ ỹ Ỹ đ';
    final f = foldVietnamese(s);
    expect(f.length, s.length);
    expect(f, 'Sang toi Duong u y Y d');
  });

  test('đơn in chuẩn: bỏ thông tin cá nhân, tách thuốc + cách dùng', () {
    const ocr = '''
BỆNH VIỆN ĐA KHOA GIẢ LẬP
ĐƠN THUỐC
Họ tên: Nguyễn Văn Giả    Tuổi: 99
Giới tính: Nam
Địa chỉ: 123 Đường Không Có, Phường Ảo
SĐT: 0900 000 000
Số thẻ BHYT: XX0000000000000
Mã BN: 000111
Chẩn đoán: J06 - Viêm họng cấp
1. Paracetamol 500mg   20 viên
   Ngày uống 2 lần, mỗi lần 1 viên, sau ăn
2) Amoxicillin 500mg x 14 viên
   Sáng 1 viên, tối 1 viên, trước ăn
3. Vitamin C 500mg
   Uống 1 viên x 3 lần/ngày
Lời dặn: uống nhiều nước
Bác sĩ: BS. Trần Thị Mẫu
Ngày khám: 01/01/2026
''';
    final items = parsePrescription(ocr);
    expect(items.map((i) => i.medicineName), [
      'Paracetamol 500mg',
      'Amoxicillin 500mg',
      'Vitamin C 500mg',
    ]);

    final para = items[0];
    expect(para.timesPerDay, 2);
    expect(para.times, ['08:00', '20:00']);
    expect(para.doseText, '1 viên');
    expect(para.mealRelation, MealRelation.after);
    expect(para.needsReview, isFalse);

    final amox = items[1];
    expect(amox.slots, [DaySlot.morning, DaySlot.evening]);
    expect(amox.times, ['08:00', '20:00']);
    expect(amox.doseText, '1 viên');
    expect(amox.mealRelation, MealRelation.before);

    final vitc = items[2];
    expect(vitc.timesPerDay, 3);
    expect(vitc.times, ['08:00', '12:00', '18:00']);
    expect(vitc.mealRelation, MealRelation.none);

    // Không rò thông tin cá nhân vào kết quả.
    final dump = items.map((i) => '${i.medicineName} ${i.doseText}').join(' ');
    for (final secret in ['Giả', '0900', 'XX000', '000111', 'Mẫu', '123']) {
      expect(dump.contains(secret), isFalse, reason: secret);
    }
    expect(extractDiagnosis(ocr), 'Viêm họng cấp');
  });

  test('OCR mất dấu, khoảng trắng lộn xộn, dòng nhiễu', () {
    const ocr = '''
Ho ten : Le Thi Gia   Nam sinh: 1900
Dien thoai 0911.222.333
~~ ## ..
1 .   Omeprazol   20 mg   -   uong 1 vien truoc an sang
2.Loratadin 10mg
toi 1 vien
3. Siro ho Gia Lap 100 ml   1 chai
''';
    final items = parsePrescription(ocr);
    expect(items.length, 3);

    expect(items[0].medicineName, 'Omeprazol 20 mg');
    expect(items[0].slots, [DaySlot.morning]);
    expect(items[0].times, ['08:00']);
    expect(items[0].mealRelation, MealRelation.before);
    expect(items[0].needsReview, isFalse);

    expect(items[1].medicineName, 'Loratadin 10mg');
    expect(items[1].times, ['20:00']);
    expect(items[1].doseText, '1 vien');

    // Không có hướng dẫn dùng -> mặc định 08:00 và cần xem lại.
    expect(items[2].medicineName, 'Siro ho Gia Lap 100 ml');
    expect(items[2].times, ['08:00']);
    expect(items[2].needsReview, isTrue);

    for (final i in items) {
      expect(i.medicineName.contains('0911'), isFalse);
      expect(i.medicineName.toLowerCase().contains('le thi'), isFalse);
    }
  });

  test('liều khác nhau theo buổi + 4 lần/ngày + trong bữa ăn', () {
    const ocr = '''
1. Metformin 850mg
Sáng 2 viên, chiều 1 viên, trong bữa ăn
2. Thuốc giả lập B 5mg
Ngày 4 lần, mỗi lần 1 gói
''';
    final items = parsePrescription(ocr);
    expect(items.length, 2);
    expect(items[0].slots, [DaySlot.morning, DaySlot.afternoon]);
    expect(items[0].times, ['08:00', '17:00']);
    expect(items[0].doseText, 'sáng 2 viên, chiều 1 viên');
    expect(items[0].mealRelation, MealRelation.during);

    expect(items[1].timesPerDay, 4);
    expect(items[1].times, ['08:00', '12:00', '16:00', '20:00']);
    expect(items[1].doseText, '1 gói');
  });

  test('văn bản không phải đơn thuốc -> rỗng', () {
    expect(parsePrescription('Xin chào\nHôm nay trời đẹp'), isEmpty);
    expect(parsePrescription(''), isEmpty);
    expect(extractDiagnosis('Không có gì'), isNull);
  });

  test('haversine + sắp theo khoảng cách', () {
    const a = MedicalFacility(name: 'A', address: '', lat: 10.0, lng: 106.0);
    const b = MedicalFacility(name: 'B', address: '', lat: 10.5, lng: 106.0);
    final km = haversineKm(10.0, 106.0, 10.5, 106.0);
    expect(km, closeTo(55.6, 0.5));
    final sorted = sortByDistance([b, a], 10.01, 106.0);
    expect(sorted.first.$1.name, 'A');
  });
}
