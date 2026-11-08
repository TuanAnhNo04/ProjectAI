import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Vị trí thiết bị — chỉ dùng tính khoảng cách trên máy, không gửi đi đâu.
class DeviceLocation {
  const DeviceLocation({required this.lat, required this.lng});
  final double lat;
  final double lng;
}

/// Lỗi từ kênh native. [code]: `permission_denied`, `location_off`,
/// `unavailable`, `unsupported`, `no_activity`.
class DeviceChannelException implements Exception {
  const DeviceChannelException(this.code);
  final String code;

  @override
  String toString() => 'DeviceChannelException($code)';
}

/// Cầu nối tới MainActivity.kt qua MethodChannel `medassist/device`.
///
/// Không dùng package ngoài. Trên nền tảng khác Android, mọi hàm ném
/// [DeviceChannelException] mã `unsupported`.
class DeviceChannel {
  DeviceChannel({MethodChannel? channel, bool? supported})
      : _channel = channel ?? const MethodChannel('medassist/device'),
        _supported = supported ??
            (!kIsWeb && defaultTargetPlatform == TargetPlatform.android);

  final MethodChannel _channel;
  final bool _supported;

  bool get isSupported => _supported;

  Future<DeviceLocation> getLocation() async {
    final map = await _invoke<Map<Object?, Object?>>('getLocation');
    final lat = map?['lat'];
    final lng = map?['lng'];
    if (lat is! num || lng is! num) {
      throw const DeviceChannelException('unavailable');
    }
    return DeviceLocation(lat: lat.toDouble(), lng: lng.toDouble());
  }

  /// Mở trình quay số (ACTION_DIAL) — người dùng tự bấm gọi.
  Future<void> dial(String number) => _invoke<void>('dial', {'number': number});

  Future<void> openMap(double lat, double lng, String label) =>
      _invoke<void>('openMap', {'lat': lat, 'lng': lng, 'label': label});

  /// Mở trang web trong trình duyệt của máy. Chỉ nhận `https`; gọi hàm này CHỈ sau
  /// khi người dùng đã xác nhận rời app (xem news_screen.dart).
  Future<void> openUrl(String url) => _invoke<void>('openUrl', {'url': url});

  Future<void> searchMap(String query) =>
      _invoke<void>('searchMap', {'query': query});

  Future<T?> _invoke<T>(String method, [Map<String, Object?>? args]) async {
    if (!_supported) throw const DeviceChannelException('unsupported');
    try {
      return await _channel.invokeMethod<T>(method, args);
    } on PlatformException catch (e) {
      throw DeviceChannelException(e.code);
    } on MissingPluginException {
      throw const DeviceChannelException('unsupported');
    }
  }
}
