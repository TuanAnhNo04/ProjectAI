import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/news_article.dart';
import 'package:image_processing_app/domain/ports/news_repository.dart';

/// Adapter HTTP cho tin tức — gọi `GET /v1/news` (công khai, không cần token).
class RemoteNewsRepository implements NewsRepository {
  RemoteNewsRepository(this._api, {required String path}) : _path = path;

  final ApiClient _api;
  final String _path;

  @override
  Future<NewsSnapshot> fetchNews({bool refresh = false}) async {
    final response = await _api.get(
      _path,
      query: refresh ? const {'refresh': true} : null,
    );
    return NewsSnapshot.fromJson(asJsonMap(response.data, _path));
  }
}
