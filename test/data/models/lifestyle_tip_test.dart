// Parse hợp đồng lifestyle: kind diet, is_authoritative, unmatched_conditions.
// Dữ liệu GIẢ.
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';

void main() {
  final json = <String, dynamic>{
    'tips': [
      {
        'title': 'Ăn nhạt',
        'body': 'Giảm muối giả lập.',
        'kind': 'diet',
        'citations': [
          {'id': 'd1', 'source': 'Tài liệu giả', 'is_authoritative': true},
          {'id': 'd2', 'source': 'Mẫu giả'},
        ],
      },
      {'title': 'Lạ', 'body': 'x', 'kind': 'weird'},
    ],
    'unmatched_conditions': ['Bệnh giả A', ' ', 'Bệnh giả B'],
  };

  test('diet kind, is_authoritative mặc định false, unmatched', () {
    final s = LifestyleSnapshot.fromJson(json);
    final tip = s.tips.first;
    expect(tip.isDiet, isTrue);
    expect(tip.citations[0].isAuthoritative, isTrue);
    expect(tip.citations[1].isAuthoritative, isFalse);
    expect(tip.isUnverified, isTrue);
    expect(s.unmatchedConditions, ['Bệnh giả A', 'Bệnh giả B']);
  });

  test('kind lạ vẫn parse (không phải diet/medication)', () {
    final t = LifestyleSnapshot.fromJson(json).tips[1];
    expect(t.isDiet, isFalse);
    expect(t.isMedication, isFalse);
    expect(t.body, 'x');
  });

  test('thiếu unmatched_conditions -> rỗng', () {
    expect(
        LifestyleSnapshot.fromJson({'tips': []}).unmatchedConditions, isEmpty);
  });

  test('cache round trip giữ nguyên các trường mới', () {
    final a = LifestyleSnapshot.fromJson(json);
    final b = LifestyleSnapshot.fromJson(a.toJson());
    expect(b.unmatchedConditions, a.unmatchedConditions);
    expect(b.tips.first.isDiet, isTrue);
    expect(b.tips.first.citations.map((c) => c.isAuthoritative), [true, false]);
  });
}
