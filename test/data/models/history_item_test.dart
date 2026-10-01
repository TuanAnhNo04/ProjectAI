// Day 6 — HistoryItem: `GET /history` trả bản ghi **thiếu `id`**,
// và `search_results` rỗng ở lần quét ảnh không nhận ra thuốc.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/history_item.dart';

import 'fixtures/server_samples.dart';

List<Map<String, dynamic>> _rawHistory() =>
    ((jsonDecode(historyBody) as Map<String, dynamic>)['history']! as List)
        .cast<Map<String, dynamic>>();

final Matcher throwsParsing = throwsA(
  isA<AppException>()
      .having((e) => e.kind, 'kind', AppExceptionKind.parsing),
);

void main() {
  group('fromJson — JSON thật', () {
    test('đọc id + filepath + search_results', () {
      final item = HistoryItem.fromJson(_rawHistory().first);

      expect(item.id, 1);
      expect(item.filepath, 'C:/data/user/0/cache/photo_1.jpg');
      expect(item.searchResults, hasLength(1));
      expect(item.searchResults.first.name, 'Panadol 500mg');
      expect(item.isValid, isTrue);
    });

    test('search_results rỗng vẫn là bản ghi hợp lệ', () {
      final item = HistoryItem.fromJson(_rawHistory()[1]);

      expect(item.id, 2);
      expect(item.searchResults, isEmpty);
      expect(item.isValid, isTrue);
    });

    test('thiếu id -> isValid = false (không render nút xoá chết)', () {
      final item = HistoryItem.fromJson(_rawHistory()[2]);

      expect(item.id, isNull);
      expect(item.isValid, isFalse);
    });

    test('created_at thiếu -> null; có thì parse ISO-8601', () {
      expect(
        HistoryItem.fromJson(<String, dynamic>{'id': 1}).createdAt,
        isNull,
      );
      final withDate = HistoryItem.fromJson(<String, dynamic>{
        'id': 1,
        'created_at': '2026-09-21T08:30:00Z',
      });
      expect(withDate.createdAt, isNotNull);
      expect(withDate.createdAt!.toUtc().year, 2026);
    });

    test('created_at sai định dạng -> null, không ném', () {
      expect(
        HistoryItem.fromJson(<String, dynamic>{
          'id': 1,
          'created_at': 'khong phai ngay',
        }).createdAt,
        isNull,
      );
    });
  });

  group('HistoryResponse', () {
    test('lọc sẵn bản ghi không xoá được (thiếu id)', () {
      final res = HistoryResponse.parse(historyBody);

      // 3 bản ghi trong JSON, 1 bản ghi không có id -> chỉ còn 2.
      expect(_rawHistory(), hasLength(3));
      expect(res.items, hasLength(2));
      expect(res.isEmpty, isFalse);
    });

    test('history null / thiếu -> rỗng, KHÔNG ném', () {
      expect(
        HistoryResponse.fromJson(<String, dynamic>{'history': null}).items,
        isEmpty,
      );
      expect(
        HistoryResponse.fromJson(<String, dynamic>{}).isEmpty,
        isTrue,
      );
    });

    test('history sai kiểu -> AppException parsing', () {
      expect(
        () => HistoryResponse.fromJson(
          <String, dynamic>{'history': 5},
        ),
        throwsParsing,
      );
    });

    test('body HTML của Flask -> parsing chứ không crash', () {
      expect(() => HistoryResponse.parse('<!doctype html>'), throwsParsing);
    });
  });

  group('copyWith / == / toJson', () {
    test('copyWith đổi filepath, giữ id và kết quả', () {
      final item = HistoryItem.fromJson(_rawHistory().first);

      final moved = item.copyWith(filepath: 'D:/moved.jpg');

      expect(moved.filepath, 'D:/moved.jpg');
      expect(moved.id, item.id);
      expect(moved.searchResults, item.searchResults);
    });

    test('copyWith() không đổi gì', () {
      final item = HistoryItem.fromJson(_rawHistory().first);

      expect(item.copyWith(), item);
    });

    test('== so cả danh sách thuốc bên trong', () {
      final a = HistoryItem.fromJson(_rawHistory().first);

      expect(a, HistoryItem.fromJson(_rawHistory().first));
      expect(a.hashCode, HistoryItem.fromJson(_rawHistory().first).hashCode);
      expect(a, isNot(a.copyWith(searchResults: const [])));
    });

    test('round-trip toJson -> fromJson', () {
      final original = HistoryItem.fromJson(_rawHistory().first);

      expect(HistoryItem.fromJson(original.toJson()), original);
    });
  });
}