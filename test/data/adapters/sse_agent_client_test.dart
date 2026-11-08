// SseAgentClient: parse luồng SSE của `/v1/agent/chat` thành AgentEvent.
//
// Hồi quy quan trọng: Dio trả `Stream<Uint8List>`; bản cũ gọi thẳng
// `transform(utf8.decoder)` và ném lỗi kiểu ở runtime nên chat KHÔNG chạy được với
// server thật (chỉ lộ ra khi chạy kiểm thử sống). Test này dùng đúng `Uint8List`.
//
// Dữ liệu GIẢ, không chạm mạng.

import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/adapters/sse_agent_client.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';

import 'fake_http.dart';

/// Tách [text] thành các mảnh [size] byte — cắt NGANG ký tự UTF-8 nhiều byte để
/// kiểm tra bộ giải mã ghép đúng giữa các mảnh.
List<Uint8List> _chunks(String text, int size) {
  final bytes = utf8.encode(text);
  return [
    for (var i = 0; i < bytes.length; i += size)
      Uint8List.fromList(
        bytes.sublist(i, i + size > bytes.length ? bytes.length : i + size),
      ),
  ];
}

ResponseBody _sse(String text, {int chunkSize = 7, int status = 200}) =>
    ResponseBody(
      Stream<Uint8List>.fromIterable(_chunks(text, chunkSize)),
      status,
      headers: {
        Headers.contentTypeHeader: ['text/event-stream'],
      },
    );

const String _stream = 'event: tool_call\n'
    'data: {"name": "explain_for_patient", "arguments": ""}\n\n'
    'event: citation\n'
    'data: {"documentId": "R1", "quote": "Hoạt chất giả", "title": "Thuốc Giả"}\n\n'
    'event: token\n'
    'data: {"text": "Tác dụng: giảm đau giả [R1]"}\n\n'
    'event: done\n'
    'data: {"sessionId": "sess1234"}\n\n';

void main() {
  test('parse symptom_noted; sự kiện lạ vô hại; terms rỗng bị bỏ', () async {
    const s = 'event: symptom_noted\n'
        'data: {"terms": ["buồn nôn", " ", "đau đầu"], "recurring": true}\n\n'
        'event: future_event\n'
        'data: {"x": 1}\n\n'
        'event: symptom_noted\n'
        'data: {"terms": []}\n\n'
        'event: token\n'
        'data: {"text": "ok"}\n\n'
        'event: done\n'
        'data: {}\n\n';
    final f = fakeClient((_) => _sse(s));
    final events = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'hi')
        .toList();
    expect(events, hasLength(3));
    final n = events[0] as SymptomNoted;
    expect(n.terms, ['buồn nôn', 'đau đầu']);
    expect(n.recurring, isTrue);
    expect(events[1], isA<TokenDelta>());
    expect(events[2], isA<Done>());
  });

  test('parse đủ tool_call/citation/token/done từ Stream<Uint8List>', () async {
    final f = fakeClient((_) => _sse(_stream));

    final events = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'hi')
        .toList();

    expect(events[0], isA<ToolCallStarted>());
    expect((events[1] as Citation).documentId, 'R1');
    expect((events[2] as TokenDelta).text, 'Tác dụng: giảm đau giả [R1]');
    expect((events[3] as Done).sessionId, 'sess1234');
    expect(events, hasLength(4));
  });

  test('chữ tiếng Việt nhiều byte bị cắt giữa mảnh vẫn ghép đúng', () async {
    for (final size in [1, 2, 3, 5]) {
      final f = fakeClient((_) => _sse(_stream, chunkSize: size));
      final events = await SseAgentClient(f.client)
          .chat(sessionId: 'sess1234', message: 'hi')
          .toList();
      expect((events[2] as TokenDelta).text, 'Tác dụng: giảm đau giả [R1]',
          reason: 'chunkSize=$size');
    }
  });

  test('refusal KHÔNG thành Failure (mã thô không được hiện cho người dùng)',
      () async {
    const refused = 'event: refusal\n'
        'data: {"reason": "out_of_scope"}\n\n'
        'event: token\n'
        'data: {"text": "Ứng dụng không tư vấn liều dùng."}\n\n'
        'event: done\n'
        'data: {"sessionId": "sess1234"}\n\n';
    final f = fakeClient((_) => _sse(refused));

    final events = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'liều dùng')
        .toList();

    expect(events.whereType<Failure>(), isEmpty);
    expect(events.whereType<TokenDelta>().single.text,
        'Ứng dụng không tư vấn liều dùng.');
    expect(events.last, isA<Done>());
  });

  test('server không gửi done -> client tự đóng bằng Done (UI không treo)',
      () async {
    final f = fakeClient(
      (_) => _sse('event: token\ndata: {"text": "xin chào"}\n\n'),
    );

    final events = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'hi')
        .toList();

    expect(events.last, isA<Done>());
  });

  test('sự kiện failure của server -> Failure mang thông điệp', () async {
    final f = fakeClient(
      (_) => _sse(
        'event: failure\ndata: {"message": "Lỗi hệ thống.", "reason": "internal"}\n\n',
      ),
    );

    final events = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'hi')
        .toList();

    expect((events.first as Failure).error.message, 'Lỗi hệ thống.');
  });

  test('HTTP lỗi (401) -> đúng 1 Failure, stream không ném', () async {
    final f = fakeClient(
      (_) => jsonBody({'code': 'unauthorized'}, 401),
    );

    final events = await SseAgentClient(f.client)
        .chat(sessionId: 'sess1234', message: 'hi')
        .toList();

    expect(events, hasLength(1));
    expect(events.single, isA<Failure>());
  });

  test('gửi đúng body: message, session_id, current_medications', () async {
    final f = fakeClient((_) => _sse(_stream));

    await SseAgentClient(f.client).chat(
      sessionId: 'sess1234',
      message: 'hi',
      currentMedications: const ['Thuốc Giả'],
    ).toList();

    expect(f.adapter.last.uri.path, '/v1/agent/chat');
    expect(f.adapter.last.data, {
      'message': 'hi',
      'session_id': 'sess1234',
      'current_medications': ['Thuốc Giả'],
      'cloud_ai_consent': false,
    });
  });

  test('done có mode cloud -> Done.mode == cloud', () async {
    final f = fakeClient((_) =>
        _sse('event: done\ndata: {"sessionId": "s", "mode": "cloud"}\n\n'));
    final events = await SseAgentClient(f.client)
        .chat(sessionId: 's', message: 'hi')
        .toList();
    expect((events.last as Done).mode, 'cloud');
  });

  test('done không có mode (server cũ) -> extractive', () async {
    final f = fakeClient((_) => _sse(_stream));
    final events = await SseAgentClient(f.client)
        .chat(sessionId: 's', message: 'hi')
        .toList();
    expect((events.last as Done).mode, 'extractive');
  });
}
