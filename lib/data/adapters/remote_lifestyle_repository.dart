import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/domain/ports/lifestyle_repository.dart';

/// Adapter HTTP — `POST /v1/agent/lifestyle` (Bearer qua [ApiClient]).
class RemoteLifestyleRepository implements LifestyleRepository {
  RemoteLifestyleRepository(this._api, {required String path}) : _path = path;

  final ApiClient _api;
  final String _path;

  @override
  Future<LifestyleSnapshot> fetchTips(LifestyleRequest request) async {
    final response = await _api.post(_path, data: request.toJson());
    return LifestyleSnapshot.fromJson(asJsonMap(response.data, _path));
  }
}
