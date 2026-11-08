import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/json_utils.dart';

/// Nguồn trích dẫn của một lời khuyên.
class LifestyleCitation {
  const LifestyleCitation({
    required this.id,
    required this.medicineName,
    this.source = '',
    this.excerpt = '',
    this.isAuthoritative = false,
    this.sourceUrl = '',
    this.retrievedAt = '',
    this.license = '',
    this.reviewedBy = '',
  });

  factory LifestyleCitation.fromJson(Map<String, dynamic> j) =>
      LifestyleCitation(
        id: asString(j['id']),
        medicineName: asString(j['medicine_name']),
        source: asString(j['source']),
        excerpt: asString(j['excerpt']),
        isAuthoritative: j['is_authoritative'] == true,
        sourceUrl: asString(j['source_url']).trim(),
        retrievedAt: asString(j['retrieved_at']).trim(),
        license: asString(j['license']).trim(),
        reviewedBy: asString(j['reviewed_by']).trim(),
      );

  final String id;
  final String medicineName;
  final String source;
  final String excerpt;

  /// Nguon da kiem chung? Thieu truong = false (an toan: hien nhan canh bao).
  final bool isAuthoritative;

  /// Xuất xứ (rỗng = không rõ). `sourceUrl` phải được re-validate trước khi mở.
  final String sourceUrl;
  final String retrievedAt;
  final String license;
  final String reviewedBy;

  Map<String, dynamic> toJson() => {
        'id': id,
        'medicine_name': medicineName,
        'source': source,
        'excerpt': excerpt,
        'is_authoritative': isAuthoritative,
        'source_url': sourceUrl,
        'retrieved_at': retrievedAt,
        'license': license,
        'reviewed_by': reviewedBy,
      };
}

class LifestyleTip {
  const LifestyleTip({
    required this.title,
    required this.body,
    this.kind = 'general',
    this.citations = const [],
  });

  factory LifestyleTip.fromJson(Map<String, dynamic> j) {
    final cits = j['citations'];
    return LifestyleTip(
      title: asString(j['title']),
      body: asString(j['body']),
      kind: asString(j['kind'], fallback: 'general'),
      citations: [
        if (cits is List)
          for (final c in cits)
            if (c is Map)
              LifestyleCitation.fromJson(Map<String, dynamic>.from(c)),
      ],
    );
  }

  final String title;
  final String body;

  /// `medication` | `general` | `diet`.
  final String kind;
  final List<LifestyleCitation> citations;

  bool get isMedication => kind == 'medication';
  bool get isDiet => kind == 'diet';

  /// Co trich dan nao chua kiem chung (hoac khong co trich dan nao)?
  bool get isUnverified =>
      citations.isEmpty || citations.any((c) => !c.isAuthoritative);

  Map<String, dynamic> toJson() => {
        'title': title,
        'body': body,
        'kind': kind,
        'citations': citations.map((c) => c.toJson()).toList(),
      };
}

class LifestyleSnapshot {
  const LifestyleSnapshot({
    required this.tips,
    this.disclaimer = '',
    this.generatedBy = 'rules',
    this.unmatchedConditions = const [],
  });

  factory LifestyleSnapshot.fromJson(Map<String, dynamic> j) {
    final tips = j['tips'];
    return LifestyleSnapshot(
      tips: [
        if (tips is List)
          for (final t in tips)
            if (t is Map) LifestyleTip.fromJson(Map<String, dynamic>.from(t)),
      ],
      disclaimer: asString(j['disclaimer']),
      generatedBy: asString(j['generated_by'], fallback: 'rules'),
      unmatchedConditions: [
        if (j['unmatched_conditions'] is List)
          for (final c in j['unmatched_conditions'] as List)
            if (asString(c).trim().isNotEmpty) asString(c).trim(),
      ],
    );
  }

  final List<LifestyleTip> tips;
  final String disclaimer;

  /// `rules` | `claude`.
  final String generatedBy;

  /// Benh nen cua ho so chua co tai lieu dinh duong da kiem chung.
  final List<String> unmatchedConditions;

  bool get isAi => generatedBy == 'claude';

  Map<String, dynamic> toJson() => {
        'tips': tips.map((t) => t.toJson()).toList(),
        'disclaimer': disclaimer,
        'generated_by': generatedBy,
        'unmatched_conditions': unmatchedConditions,
      };
}

/// Body của `POST /v1/agent/lifestyle`. KHÔNG chứa lịch sử chat.
class LifestyleRequest {
  const LifestyleRequest({
    required this.currentMedications,
    required this.conditions,
    required this.recentTopics,
    required this.cloudAiConsent,
  });

  /// Dựng request: thuốc = tên trong lịch nhắc + thuốc trong hồ sơ (bỏ trùng,
  /// tối đa 20); bệnh nền CHỈ khi hồ sơ đã có consent; không gửi chủ đề chat.
  factory LifestyleRequest.build({
    required List<String> reminderMedicineNames,
    required HealthProfile? profile,
    required bool cloudAiConsent,
  }) {
    String clip(String s) => s.length > 100 ? s.substring(0, 100) : s;
    final seen = <String>{};
    final meds = <String>[];
    for (final raw in [
      ...reminderMedicineNames,
      ...?profile?.currentMedications,
    ]) {
      final name = raw.trim();
      if (name.isEmpty || !seen.add(name.toLowerCase())) continue;
      meds.add(clip(name));
      if (meds.length >= 20) break;
    }
    final conditions = (profile != null && profile.hasConsent)
        ? profile.conditions
            .map((c) => c.trim())
            .where((c) => c.isNotEmpty)
            .take(20)
            .map(clip)
            .toList()
        : <String>[];
    return LifestyleRequest(
      currentMedications: meds,
      conditions: conditions,
      recentTopics: const [],
      cloudAiConsent: cloudAiConsent,
    );
  }

  final List<String> currentMedications;
  final List<String> conditions;
  final List<String> recentTopics;
  final bool cloudAiConsent;

  Map<String, dynamic> toJson() => {
        'current_medications': currentMedications,
        'conditions': conditions,
        'recent_topics': recentTopics,
        'cloud_ai_consent': cloudAiConsent,
      };
}
