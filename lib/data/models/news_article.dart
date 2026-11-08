import 'package:image_processing_app/data/models/json_utils.dart';

/// Một bài tin tức thuốc/y tế — chỉ tiêu đề + trích đoạn + nguồn + link, không
/// phải toàn văn bài báo (xem lý do ở `server/medassist-core/app/news.py`).
class NewsArticle {
  const NewsArticle({
    required this.id,
    required this.title,
    this.summary = '',
    this.url = '',
    required this.source,
    this.publishedAt,
    this.imageUrl,
  });

  factory NewsArticle.fromJson(Map<String, dynamic> json) => NewsArticle(
        id: asString(json['id']),
        title: asString(json['title']),
        summary: asString(json['summary']),
        url: asString(json['url']),
        source: asString(json['source']),
        publishedAt: asDateTimeOrNull(json['published_at']),
        imageUrl: asStringOrNull(json['image_url']),
      );

  static List<NewsArticle> listFromJson(Object? value, String field) =>
      asJsonList(value, field)
          .map((e) => NewsArticle.fromJson(asJsonMap(e, field)))
          .toList(growable: false);

  final String id;
  final String title;
  final String summary;

  /// Link bài gốc; rỗng với bài mẫu kèm theo app (chưa bật nguồn tin thật).
  final String url;
  final String source;
  final DateTime? publishedAt;
  final String? imageUrl;

  /// Bài mẫu không có link để mở — dùng phân biệt ở UI thay vì so `url.isEmpty`
  /// rải rác nhiều nơi.
  bool get hasLink => url.isNotEmpty;

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'summary': summary,
        'url': url,
        'source': source,
        'published_at': publishedAt?.toIso8601String(),
        'image_url': imageUrl,
      };
}

/// Phản hồi của `GET /v1/news`.
class NewsSnapshot {
  const NewsSnapshot({
    required this.articles,
    required this.isSample,
    this.failedSources = const [],
    this.disclaimer = '',
    this.verifiedSources = const [],
  });

  factory NewsSnapshot.fromJson(Map<String, dynamic> json) => NewsSnapshot(
        articles: NewsArticle.listFromJson(json['articles'], 'articles'),
        isSample: json['is_sample'] == true,
        failedSources: asJsonList(json['failed_sources'], 'failed_sources')
            .map((e) => asString(e))
            .toList(growable: false),
        disclaimer: asString(json['disclaimer']),
        // Server cũ chưa có trường này -> danh sách rỗng, UI không hiện dòng.
        verifiedSources: json['verified_sources'] is List
            ? asJsonList(json['verified_sources'], 'verified_sources')
                .map((e) => asString(e))
                .where((s) => s.isNotEmpty)
                .toList(growable: false)
            : const [],
      );

  /// Tên miền của các nguồn tin đã kiểm chứng mà server cho phép (vd fda.gov).
  final List<String> verifiedSources;

  final List<NewsArticle> articles;

  /// `true` = server chưa bật nguồn tin thật, đây là bài mẫu đi kèm app.
  /// UI PHẢI hiện rõ điều này — không được để người dùng tưởng là tin thật.
  final bool isSample;

  /// Tên các nguồn tải hỏng lần này (rỗng nếu mọi nguồn đều ổn, hoặc đang ở
  /// chế độ mẫu).
  final List<String> failedSources;
  final String disclaimer;
}
