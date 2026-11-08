import 'package:image_processing_app/data/models/lifestyle_tip.dart';

/// Cổng lấy lời khuyên lối sống cá nhân hoá (rules hoặc AI đám mây nếu đã đồng ý).
abstract class LifestyleRepository {
  Future<LifestyleSnapshot> fetchTips(LifestyleRequest request);
}
