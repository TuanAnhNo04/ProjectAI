// Day 8 — RemoteMedicineRepository nối `/search-medicine`, `/history`,
// `DELETE /history/{id}` của Flask hiện tại.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_repository.dart';

import 'fake_http.dart';
import '../models/fixtures/server_samples.dart';

Object? _json(String body) => jsonDecode(body);

void main() {
  group('RemoteMedicineRepository.search', () {
    test('gọi đúng endpoint, encode tên, parse results', () async {
      final f = fakeClient((_) => jsonBody(_json(searchMedicineBody)));
      final repo = RemoteMedicineRepository(f.client);

      final response = await repo.search('panadol');

      final sent = f.adapter.last;
      expect(sent.method, 'GET');
      expect(sent.uri.path, '/search-medicine');
      expect(sent.uri.queryParameters['name'], 'panadol');

      expect(response.results, hasLength(2));
      expect(response.results.first.name, 'Panadol 500mg');
      // Mọt server cũ: thuốc 2 thiếu Uses -> null, Side_effects rỗng -> null.
      expect(response.results[1].uses, isNull);
      expect(response.results[1].sideEffects, isNull);
      // total mặc định bằng số kết quả khi server không echo lại.
      expect(response.total, 2);
    });

    test('tên thuốc có dấu/khoảng trắng được encode đúng', () async {
      final f = fakeClient((_) => jsonBody(_json(searchMedicineBody)));
      final repo = RemoteMedicineRepository(f.client);

      await repo.search('thuốc cảm cúm');

      // `queryParameters` đã decode lại — muốn thấy encode thì đọc raw query.
      expect(f.adapter.last.uri.query, contains('name=thu%E1%BB%91c'));
    });

    test('server 500 -> AppException server', () async {
      final f = fakeClient((_) => jsonBody({'message': 'db chết'}, 500));
      final repo = RemoteMedicineRepository(f.client);

      await expectLater(
        repo.search('panadol'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.server),
        ),
      );
    });
  });

  group('RemoteMedicineRepository.history', () {
    test('gọi GET /history, bỏ bản ghi thiếu id', () async {
      final f = fakeClient((_) => jsonBody(_json(historyBody)));
      final repo = RemoteMedicineRepository(f.client);

      final items = await repo.history();

      expect(f.adapter.last.method, 'GET');
      expect(f.adapter.last.uri.path, '/history');

      // Fixture có 3 mục nhưng 1 mục thiếu id -> còn 2.
      expect(items, hasLength(2));
      expect(items.first.id, 1);
      expect(items.first.filepath, 'C:/data/user/0/cache/photo_1.jpg');
      expect(items.first.searchResults.single.name, 'Panadol 500mg');
      expect(items[1].searchResults, isEmpty);
    });
  });

  group('RemoteMedicineRepository.deleteHistory', () {
    test('gửi DELETE /history/{id}', () async {
      final f = fakeClient((_) => jsonBody({'success': true}));
      final repo = RemoteMedicineRepository(f.client);

      await repo.deleteHistory(42);

      expect(f.adapter.last.method, 'DELETE');
      expect(f.adapter.last.uri.path, '/history/42');
    });

    test('bản ghi không còn -> notFound', () async {
      final f = fakeClient((_) => jsonBody({'message': 'not found'}, 404));
      final repo = RemoteMedicineRepository(f.client);

      await expectLater(
        repo.deleteHistory(9),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.notFound),
        ),
      );
    });
  });
}
