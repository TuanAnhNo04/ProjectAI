import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/data/models/chat_message.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';

/// Ngữ cảnh gửi kèm một lượt hỏi — luôn cần hồ sơ có consent.
///
/// - Bệnh nền/dị ứng: thêm điều kiện đã đồng ý AI đám mây (quy tắc cũ).
/// - `recent_symptoms`: CHỈ khi bật trí nhớ trợ lý (mặc định tắt); máy chủ dùng
///   cho gợi ý theo luật trong lượt này và không lưu.
HealthContext? buildHealthContext({
  required HealthProfile? profile,
  required bool cloudAiConsent,
  bool memoryEnabled = false,
  List<SymptomEpisode> episodes = const [],
  DateTime? now,
}) {
  if (profile == null || !profile.hasConsent) return null;
  final symptoms = memoryEnabled
      ? recentSymptomTerms(episodes, now ?? DateTime.now())
      : const <String>[];
  if (!cloudAiConsent && symptoms.isEmpty) return null;
  return HealthContext(
    conditions: cloudAiConsent ? profile.conditions : const [],
    allergies: cloudAiConsent ? profile.allergies : const [],
    recentSymptoms: symptoms,
  );
}

/// Controller hội thoại agent — Day 25.
///
/// State là danh sách [ChatMessage]; mỗi khi SSE phát token mới, controller
/// thay thế tin nhắn cuối bằng bản có thêm text, làm UI rebuild đúng bubble.
class ChatController extends AutoDisposeAsyncNotifier<List<ChatMessage>> {
  StreamSubscription<AgentEvent>? _sub;
  String _sessionId = '';

  @override
  FutureOr<List<ChatMessage>> build() {
    // Làm nóng trạng thái trí nhớ để lượt hỏi đầu tiên đã có đủ dữ liệu.
    ref.read(assistantMemoryControllerProvider);
    ref.read(symptomEpisodesProvider);
    return const [];
  }

  AgentClient get _client => ref.read(agentClientProvider);

  /// Gửi tin nhắn mới.
  void send(String text) {
    if (text.trim().isEmpty) return;

    // Huỷ stream cũ nếu đang chạy.
    _cancelStream();

    if (_sessionId.isEmpty) {
      _sessionId = DateTime.now().microsecondsSinceEpoch.toRadixString(36);
    }

    final current = state.valueOrNull ?? [];
    final userMsg = ChatMessage(role: ChatRole.user, text: text);
    const assistantMsg = ChatMessage(
      role: ChatRole.assistant,
      text: '',
      status: MessageStatus.streaming,
    );

    state = AsyncData([...current, userMsg, assistantMsg]);

    // Day 27: gửi danh sách thuốc đang dùng để agent biết ngữ cảnh.
    final profile = ref.read(healthProfileControllerProvider).valueOrNull;
    final currentMeds = profile?.currentMedications ?? const [];
    final cloudConsent =
        ref.read(cloudAiConsentControllerProvider).valueOrNull?.granted ??
            false;
    final memoryActive = isAssistantMemoryActive(ref);
    final healthContext = buildHealthContext(
      profile: profile,
      cloudAiConsent: cloudConsent,
      memoryEnabled: memoryActive,
      episodes: ref.read(symptomEpisodesProvider).valueOrNull ?? const [],
    );

    final stream = _client.chat(
      sessionId: _sessionId,
      message: text,
      currentMedications: currentMeds,
      cloudAiConsent: cloudConsent,
      healthContext: healthContext,
    );
    _sub = stream.listen(
      _onEvent,
      onDone: _onStreamDone,
      onError: _onStreamError,
      cancelOnError: false,
    );
  }

  /// Bản sao có kiểu rõ ràng — `[...x ?? []]` bị suy ra `List<dynamic>` ở Dart 3.3.
  List<ChatMessage> _snapshot() =>
      <ChatMessage>[...(state.valueOrNull ?? const <ChatMessage>[])];

  void _onEvent(AgentEvent event) {
    final messages = _snapshot();
    if (messages.isEmpty) return;

    final last = messages.last;
    if (last.isUser) return;

    switch (event) {
      case TokenDelta(:final text):
        messages[messages.length - 1] = last.copyWith(
          text: last.text + text,
          toolName: () => null,
        );
      case ToolCallStarted(:final name):
        messages[messages.length - 1] = last.copyWith(
          toolName: () => name,
        );
      case Citation():
        messages[messages.length - 1] = last.copyWith(
          citations: [...last.citations, event],
        );
      case EmergencyAlert():
        messages[messages.length - 1] = last.copyWith(emergency: event);
      case SymptomNoted():
        messages[messages.length - 1] = last.copyWith(symptom: event);
        _remember(event);
      case Done(:final mode):
        messages[messages.length - 1] = last.copyWith(
          status: MessageStatus.done,
          mode: mode,
          toolName: () => null,
        );
      case Failure(:final error):
        messages[messages.length - 1] = last.copyWith(
          status: MessageStatus.error,
          toolName: () => null,
          errorText: () => error.message,
        );
    }

    state = AsyncData(messages);
  }

  /// Lưu lần ghi nhớ vào két trên máy — chỉ khi trí nhớ bật VÀ hồ sơ có consent.
  /// Không bao giờ lưu nội dung câu hỏi; chỉ cụm triệu chứng + tên thuốc đang dùng.
  Future<void> _remember(SymptomNoted event) async {
    if (!isAssistantMemoryActive(ref)) return;
    final meds = ref
            .read(healthProfileControllerProvider)
            .valueOrNull
            ?.currentMedications ??
        const <String>[];
    final at = DateTime.now();
    try {
      await ref.read(symptomEpisodesProvider.notifier).add(SymptomEpisode(
            at: at,
            terms: event.terms,
            meds: meds,
            recurring: event.recurring,
          ));
    } on Object {
      return; // két không ghi được -> không báo "đã ghi nhớ"
    }
    final messages = _snapshot();
    for (var i = messages.length - 1; i >= 0; i--) {
      if (identical(messages[i].symptom, event)) {
        messages[i] = messages[i].copyWith(savedEpisodeAt: () => at);
        state = AsyncData(messages);
        break;
      }
    }
  }

  /// Nút "Xoá" dưới câu trả lời: xoá lần ghi nhớ đó khỏi két.
  Future<void> forgetSymptom(ChatMessage message) async {
    final at = message.savedEpisodeAt;
    if (at == null) return;
    try {
      await ref.read(symptomEpisodesProvider.notifier).remove(at);
    } on Object {
      return;
    }
    final messages = _snapshot();
    for (var i = 0; i < messages.length; i++) {
      if (messages[i].savedEpisodeAt == at) {
        messages[i] = messages[i].copyWith(savedEpisodeAt: () => null);
      }
    }
    state = AsyncData(messages);
  }

  void _onStreamDone() {
    final messages = _snapshot();
    if (messages.isNotEmpty && messages.last.isStreaming) {
      messages[messages.length - 1] = messages.last.copyWith(
        status: MessageStatus.done,
        toolName: () => null,
      );
      state = AsyncData(messages);
    }
  }

  void _onStreamError(Object error) {
    final messages = _snapshot();
    if (messages.isNotEmpty && messages.last.isStreaming) {
      messages[messages.length - 1] = messages.last.copyWith(
        status: MessageStatus.error,
        toolName: () => null,
        errorText: () => 'Lỗi kết nối.',
      );
      state = AsyncData(messages);
    }
  }

  /// Nút Dừng — hủy stream SSE đang chạy.
  void stop() {
    _cancelStream();
    final messages = _snapshot();
    if (messages.isNotEmpty && messages.last.isStreaming) {
      messages[messages.length - 1] = messages.last.copyWith(
        status: MessageStatus.done,
        toolName: () => null,
      );
      state = AsyncData(messages);
    }
  }

  /// Xoá toàn bộ hội thoại.
  void clear() {
    _cancelStream();
    _sessionId = '';
    state = const AsyncData([]);
  }

  void _cancelStream() {
    _sub?.cancel();
    _sub = null;
  }
}
