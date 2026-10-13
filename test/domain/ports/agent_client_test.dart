// Day 7 — AgentEvent: 5 biến thể `sealed` của luồng SSE.
//
// Test chứng minh điều quan trọng nhất: `Failure` mang **AppException** nên tầng
// UI hiển thị được lỗi tiếng Việt mà không cần `try/catch`.

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';

void main() {
  group('TokenDelta', () {
    test('giữ nguyên chữ, kể cả khoảng trắng và xuống dòng', () {
      const e = TokenDelta(' paracetamol ');

      expect(e.text, ' paracetamol ');
      expect(e.toString(), contains('paracetamol'));
    });
  });

  group('ToolCallStarted', () {
    test('arguments mặc định rỗng, không null', () {
      const e = ToolCallStarted(name: 'check_interaction');

      expect(e.name, 'check_interaction');
      expect(e.arguments, isEmpty);
    });

    test('giữ tham số đã che', () {
      const e = ToolCallStarted(
        name: 'search_drug',
        arguments: <String, Object?>{'query': 'panadol', 'token': '***'},
      );

      expect(e.arguments['query'], 'panadol');
      expect(e.arguments['token'], '***');
      expect(e.toString(), contains('search_drug'));
    });
  });

  group('Citation', () {
    test('bắt buộc có documentId + quote', () {
      const e = Citation(
        documentId: 'leaflet-01',
        quote: 'Paracetamol 500mg',
        title: 'Tờ hướng dẫn',
      );

      expect(e.documentId, 'leaflet-01');
      expect(e.quote, 'Paracetamol 500mg');
      expect(e.title, 'Tờ hướng dẫn');
    });

    test('title có thể null', () {
      expect(const Citation(documentId: 'a', quote: 'b').title, isNull);
    });
  });

  group('Done', () {
    test('sessionId có thể null', () {
      expect(const Done().sessionId, isNull);
      expect(const Done(sessionId: 's1').sessionId, 's1');
    });
  });

  group('Failure', () {
    test('mang AppException — UI hiện được message tiếng Việt', () {
      const e = Failure(
        AppException(
          kind: AppExceptionKind.server,
          message: 'Máy chủ đang gặp lỗi, vui lòng thử lại sau.',
        ),
      );

      expect(e.error.kind, AppExceptionKind.server);
      expect(e.error.message, contains('Máy chủ'));
      expect(e.error.isRetryable, isTrue);
      expect(e.toString(), contains('server'));
    });

    test('request bị huỷ vẫn là Failure nhưng isSilent để UI im', () {
      const e = Failure(
        AppException(
            kind: AppExceptionKind.canceled, message: 'Yêu cầu đã bị huỷ.'),
      );

      expect(e.error.isSilent, isTrue);
    });
  });

  test('switch trên AgentEvent xử lý đủ 5 nhánh', () {
    // Nhờ `sealed`, thêm biến thể mới sẽ khiến test này KHÔNG còn biên dịch
    // — đúng ý muốn (bắt buộc UI xử lý đủ).
    final events = <AgentEvent>[
      const TokenDelta('a'),
      const ToolCallStarted(name: 'search_drug'),
      const Citation(documentId: 'd', quote: 'q'),
      const Done(sessionId: 's'),
      const Failure(
        AppException(kind: AppExceptionKind.network, message: 'mất mạng'),
      ),
    ];

    final seen = <String>[];
    for (final e in events) {
      seen.add(
        switch (e) {
          TokenDelta() => 'token',
          ToolCallStarted() => 'tool',
          Citation() => 'citation',
          Done() => 'done',
          Failure() => 'failure',
        },
      );
    }

    expect(seen, <String>['token', 'tool', 'citation', 'done', 'failure']);
  });

  group('AgentClient — hợp đồng của adapter', () {
    test('adapter giả minh hoạ luồng đầy đủ', () async {
      final client = _FakeAgentClient([
        const TokenDelta('Xin'),
        const ToolCallStarted(name: 'check_interaction'),
        const Citation(documentId: 'l1', quote: 'không nên dùng chung'),
        const Done(sessionId: 's1'),
      ]);

      final got = await client
          .chat(sessionId: 's1', message: 'warfarin + panadol?')
          .toList();

      expect(got, hasLength(4));
      expect(got.first, isA<TokenDelta>());
      expect(got.last, isA<Done>());
      expect(client.lastSession, 's1');
      expect(client.lastMessage, 'warfarin + panadol?');
    });
  });
}

/// Adapter giả — đồng thời là **khuôn mẫu** cho adapter thật ở Day 22.
class _FakeAgentClient implements AgentClient {
  _FakeAgentClient(this.events);

  final List<AgentEvent> events;
  String? lastSession;
  String? lastMessage;

  @override
  Stream<AgentEvent> chat({
    required String sessionId,
    required String message,
  }) {
    lastSession = sessionId;
    lastMessage = message;
    return Stream<AgentEvent>.fromIterable(events);
  }
}
