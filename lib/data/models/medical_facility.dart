import 'dart:convert';
import 'dart:math' as math;

/// Cơ sở y tế trong danh sách mẫu `assets/data/medical_facilities.json`.
/// Dữ liệu chưa kiểm chứng (`verified: false`) — xem docs/MEDICAL_FACILITIES.md.
class MedicalFacility {
  const MedicalFacility({
    required this.name,
    required this.address,
    required this.lat,
    required this.lng,
    this.city = '',
    this.verified = false,
  });

  final String name;
  final String address;
  final double lat;
  final double lng;
  final String city;
  final bool verified;

  factory MedicalFacility.fromJson(Map<String, dynamic> json) =>
      MedicalFacility(
        name: json['name'] as String? ?? '',
        address: json['address'] as String? ?? '',
        lat: (json['lat'] as num?)?.toDouble() ?? 0,
        lng: (json['lng'] as num?)?.toDouble() ?? 0,
        city: json['city'] as String? ?? '',
        verified: json['verified'] == true,
      );

  static List<MedicalFacility> listFromJsonString(String source) {
    final decoded = jsonDecode(source);
    final list = decoded is Map ? decoded['facilities'] : decoded;
    if (list is! List) return const [];
    return [
      for (final e in list)
        if (e is Map<String, dynamic>) MedicalFacility.fromJson(e),
    ];
  }
}

/// Khoảng cách đường chim bay (km) — công thức haversine, tính hoàn toàn trên máy.
double haversineKm(double lat1, double lng1, double lat2, double lng2) {
  const r = 6371.0;
  double rad(double d) => d * math.pi / 180;
  final dLat = rad(lat2 - lat1);
  final dLng = rad(lng2 - lng1);
  final a = math.pow(math.sin(dLat / 2), 2) +
      math.cos(rad(lat1)) *
          math.cos(rad(lat2)) *
          math.pow(math.sin(dLng / 2), 2);
  return 2 * r * math.asin(math.sqrt(a));
}

/// Sắp xếp theo khoảng cách tới (lat, lng); trả về cặp (cơ sở, km).
List<(MedicalFacility, double)> sortByDistance(
  List<MedicalFacility> facilities,
  double lat,
  double lng,
) {
  final list = [
    for (final f in facilities) (f, haversineKm(lat, lng, f.lat, f.lng)),
  ]..sort((a, b) => a.$2.compareTo(b.$2));
  return list;
}
