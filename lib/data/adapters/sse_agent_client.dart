import 'dart:async';
import 'dart:convert';

import 'package:dio/dio.dart';

import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';

/// Adapter SSE cho `/v1/agent/chat` (Day 25).
///
/// Parse dòng `event: <name>\ndata: <json>\n\n` từ Dio stream rồi phát
/// [AgentEvent]. Lỗi giữa chừng LUÔN đóng gói thành [Failure] — stream
/// không bao giờ ném exception (hợp đồng của [AgentClient]).
class SseAgentClient implements AgentClient {
  SseAgentClient(this._client);

  final ApiClient _client;

  @override
  Stream<AgentEvent> chat({
    required String sessionId,
    required String message,
    List<String> currentMedications = const [],
    bool cloudAiConsent = false,
    HealthContext? healthContext,
  }) async* {
    final Response<ResponseBody> response;
    try {
      response = await _client.request<ResponseBody>(
        'POST',
        '/v1/agent/chat',
        data: {
          'message': message,
          'session_id': sessionId,
          if (currentMedications.isNotEmpty)
            'current_medications': currentMedications,
          'cloud_ai_consent': cloudAiConsent,
          if (healthContext != null && !healthContext.isEmpty)
            'health_context': {
              'conditions': _clip(healthContext.conditions),
              'allergies': _clip(healthContext.allergies),
              if (healthContext.recentSymptoms.isNotEmpty)
                'recent_symptoms': _clip(healthContext.recentSymptoms, max: 5),
            },
        },
        responseType: ResponseType.stream,
        receiveTimeout: const Duration(minutes: 2),
      );
    } on AppException catch (e) {
      yield Failure(e);
      return;
    } catch (e) {
      yield Failure(AppException.from(e));
      return;
    }

    yield* _parseStream(response.data!.stream);
  }

  /// Giới hạn theo hợp đồng server: ≤20 mục, mỗi mục ≤100 ký tự.
  static List<String> _clip(List<String> items, {int max = 20}) => items
      .map((e) => e.trim())
      .where((e) => e.isNotEmpty)
      .take(max)
      .map((e) => e.length > 100 ? e.substring(0, 100) : e)
      .toList();

  Stream<AgentEvent> _parseStream(Stream<List<int>> raw) async* {
    final buffer = StringBuffer();
    String? currentEvent;
    var gotDone = false;

    try {
      // `cast` bắt buộc: Dio trả `Stream<Uint8List>`; gọi thẳng
      // `transform(utf8.decoder)` ném "Utf8Decoder is not a subtype of
      // StreamTransformer<Uint8List, String>" ở runtime — lỗi này làm chat chưa
      // từng chạy được với server thật (phát hiện khi chạy kiểm thử sống).
      await for (final chunk in raw.cast<List<int>>().transform(utf8.decoder)) {
        buffer.write(chunk);

        while (true) {
          final content = buffer.toString();
          final endIndex = content.indexOf('\n\n');
          if (endIndex < 0) break;

          final block = content.substring(0, endIndex);
          buffer
            ..clear()
            ..write(content.substring(endIndex + 2));

          for (final line in block.split('\n')) {
            if (line.startsWith('event: ')) {
              currentEvent = line.substring(7).trim();
            } else if (line.startsWith('data: ') && currentEvent != null) {
              final event = _decode(currentEvent, line.substring(6));
              if (event != null) {
                yield event;
                if (event is Done) gotDone = true;
              }
              currentEvent = null;
            }
          }
        }
      }
    } catch (e) {
      if (!gotDone) {
        yield Failure(AppException(
          kind: AppExceptionKind.network,
          message: 'Kết nối bị ngắt khi đang nhận dữ liệu.',
          cause: e,
        ));
      }
      return;
    }

    if (!gotDone) {
      yield const Done();
    }
  }

  static AgentEvent? _symptom(Map<String, dynamic> json) {
    final raw = json['terms'];
    if (raw is! List) return null;
    final terms = <String>[
      for (final t in raw)
        if (t is String && t.trim().isNotEmpty)
          t.trim().length > 60 ? t.trim().substring(0, 60) : t.trim(),
    ].take(5).toList();
    if (terms.isEmpty) return null;
    return SymptomNoted(terms: terms, recurring: json['recurring'] == true);
  }

  static String _str(Object? v) => v is String ? v.trim() : '';

  AgentEvent? _decode(String event, String data) {
    try {
      final json = jsonDecode(data) as Map<String, dynamic>;
      return switch (event) {
        'token' => TokenDelta(json['text'] as String? ?? ''),
        'tool_call' => ToolCallStarted(
            name: json['name'] as String? ?? '',
            arguments: (json['arguments'] is Map)
                ? Map<String, Object?>.from(json['arguments'] as Map)
                : const {},
          ),
        'citation' => Citation(
            documentId: json['documentId'] as String? ?? '',
            quote: json['quote'] as String? ?? '',
            title: json['title'] as String?,
            source: _str(json['source']),
            isAuthoritative: json['isAuthoritative'] == true,
            sourceUrl: _str(json['sourceUrl']),
            retrievedAt: _str(json['retrievedAt']),
            license: _str(json['license']),
            reviewedBy: _str(json['reviewedBy']),
          ),
        'emergency' => EmergencyAlert(
            matched: (json['matched'] is List)
                ? [for (final m in json['matched'] as List) m.toString()]
                : const [],
            message: json['message'] as String? ?? '',
          ),
        'symptom_noted' => _symptom(json),
        'done' => Done(
            sessionId: json['sessionId'] as String?,
            mode: json['mode'] == 'cloud' ? 'cloud' : 'extractive',
          ),
        'failure' => Failure(AppException(
            kind: AppExceptionKind.server,
            message: json['message'] as String? ?? 'Lỗi hệ thống.',
          )),
        // `refusal` chỉ mang MÃ lý do (vd `out_of_scope`); câu trả lời thân thiện
        // luôn đi kèm ngay sau bằng sự kiện `token`. Biến mã thô thành `Failure`
        // sẽ làm bong bóng chat hiện "out_of_scope" cho người dùng.
        'refusal' => null,
        _ => null,
      };
    } catch (_) {
      return null;
    }
  }
}
