import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/news_article.dart';

void main() {
  group('NewsArticle.fromJson', () {
    test('đọc đủ trường', () {
      final a = NewsArticle.fromJson({
        'id': 'a1',
        'title': 'Panadol thu hồi',
        'summary': 'Tóm tắt ngắn',
        'url': 'https://bao.example.com/bai-1',
        'source': 'Báo Giả',
        'published_at': '2026-10-28T10:00:00+00:00',
        'image_url': 'https://img.example.com/a.jpg',
      });
      expect(a.id, 'a1');
      expect(a.title, 'Panadol thu hồi');
      expect(a.source, 'Báo Giả');
      expect(a.hasLink, isTrue);
      expect(a.publishedAt, isNotNull);
    });

    test('thiếu url -> hasLink là false (bài mẫu)', () {
      final a = NewsArticle.fromJson({
        'id': 'sample-1',
        'title': 'T',
        'source': 'MedAssist',
      });
      expect(a.url, '');
      expect(a.hasLink, isFalse);
    });

    test('published_at sai định dạng -> null, không ném lỗi', () {
      final a = NewsArticle.fromJson({
        'id': '1',
        'title': 'T',
        'source': 'X',
        'published_at': 'khong-phai-ngay',
      });
      expect(a.publishedAt, isNull);
    });

    test('toJson rồi fromJson lại ra cùng dữ liệu', () {
      const a = NewsArticle(
        id: '1',
        title: 'T',
        summary: 'S',
        url: 'https://e.com',
        source: 'X',
        imageUrl: 'https://img.example.com/a.jpg',
      );
      final back = NewsArticle.fromJson(a.toJson());
      expect(back.id, a.id);
      expect(back.title, a.title);
      expect(back.url, a.url);
      expect(back.imageUrl, a.imageUrl);
    });
  });

  group('NewsSnapshot.fromJson', () {
    test('đọc is_sample, failed_sources, disclaimer', () {
      final s = NewsSnapshot.fromJson({
        'articles': [
          {'id': '1', 'title': 'T', 'source': 'X'},
        ],
        'total': 1,
        'is_sample': true,
        'failed_sources': ['Báo Hỏng'],
        'disclaimer': 'Tham khảo',
      });
      expect(s.articles, hasLength(1));
      expect(s.isSample, isTrue);
      expect(s.failedSources, ['Báo Hỏng']);
      expect(s.disclaimer, 'Tham khảo');
    });

    test('đọc verified_sources; server cũ thiếu trường -> rỗng', () {
      final withField = NewsSnapshot.fromJson({
        'articles': <Object>[],
        'is_sample': false,
        'verified_sources': ['fda.gov', '', 'who.int'],
      });
      expect(withField.verifiedSources, ['fda.gov', 'who.int']);

      final old = NewsSnapshot.fromJson({
        'articles': <Object>[],
        'is_sample': false,
      });
      expect(old.verifiedSources, isEmpty);
    });

    test('thiếu failed_sources -> danh sách rỗng, không ném lỗi', () {
      final s = NewsSnapshot.fromJson({
        'articles': [],
        'is_sample': false,
      });
      expect(s.failedSources, isEmpty);
    });

    test('articles không phải list -> ném AppException kiểu parsing', () {
      expect(
        () => NewsSnapshot.fromJson({'articles': 'không phải list'}),
        throwsA(isA<AppException>()
            .having((e) => e.kind, 'kind', AppExceptionKind.parsing)),
      );
    });
  });
}
