import 'package:image_processing_app/data/models/news_article.dart';

/// Cổng lấy tin tức thuốc/y tế. Màn hình chỉ biết cổng này, không biết HTTP
/// hay cache — cùng nguyên tắc ports & adapters như [MedicineRepository].
abstract class NewsRepository {
  /// Danh sách bài tin. [refresh] = bỏ qua cache phía server, tải lại từ nguồn.
  Future<NewsSnapshot> fetchNews({bool refresh = false});
}
