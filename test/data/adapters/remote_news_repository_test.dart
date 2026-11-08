import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/adapters/remote_news_repository.dart';

import 'fake_http.dart';

Object? _json(String body) => jsonDecode(body);

const _sampleBody = '''
{
  "articles": [
    {"id": "sample-1", "title": "Đọc kỹ hướng dẫn trước khi dùng thuốc",
     "summary": "Tờ hướng dẫn...", "url": "", "source": "MedAssist"}
  ],
  "total": 1,
  "is_sample": true,
  "failed_sources": [],
  "disclaimer": "Tham khảo, không thay thế bác sĩ."
}
''';

void main() {
  group('RemoteNewsRepository.fetchNews', () {
    test('gọi đúng endpoint, không kèm tham số refresh khi mặc định', () async {
      final f = fakeClient((_) => jsonBody(_json(_sampleBody)));
      final repo = RemoteNewsRepository(f.client, path: '/v1/news');

      final snapshot = await repo.fetchNews();

      final sent = f.adapter.last;
      expect(sent.method, 'GET');
      expect(sent.uri.path, '/v1/news');
      expect(sent.uri.queryParameters.containsKey('refresh'), isFalse);
      expect(snapshot.isSample, isTrue);
      expect(snapshot.articles, hasLength(1));
      expect(snapshot.articles.first.hasLink, isFalse);
    });

    test('refresh: true gửi kèm tham số refresh=true', () async {
      final f = fakeClient((_) => jsonBody(_json(_sampleBody)));
      final repo = RemoteNewsRepository(f.client, path: '/v1/news');

      await repo.fetchNews(refresh: true);

      expect(f.adapter.last.uri.queryParameters['refresh'], 'true');
    });

    test('endpoint công khai — không cần token vẫn gọi được', () async {
      final f = fakeClient(
        (_) => jsonBody(_json(_sampleBody)),
        tokenProvider: () async => null,
      );
      final repo = RemoteNewsRepository(f.client, path: '/v1/news');

      final snapshot = await repo.fetchNews();
      expect(snapshot.articles, isNotEmpty);
    });

    test('server lỗi 500 -> ném AppException', () async {
      final f = fakeClient((_) => jsonBody({'message': 'lỗi server'}, 500));
      final repo = RemoteNewsRepository(f.client, path: '/v1/news');

      expect(repo.fetchNews(), throwsA(isA<Object>()));
    });
  });
}
