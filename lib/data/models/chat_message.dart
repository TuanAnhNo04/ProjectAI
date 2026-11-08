import 'package:image_processing_app/domain/ports/agent_client.dart';

/// Vai trò trong cuộc hội thoại.
enum ChatRole { user, assistant }

/// Trạng thái tin nhắn phía assistant.
enum MessageStatus { streaming, done, error }

/// Một tin nhắn hiển thị trên màn hình chat (Day 25).
///
/// Immutable; controller phát bản mới mỗi lần có token/citation tới.
class ChatMessage {
  const ChatMessage({
    required this.role,
    required this.text,
    this.citations = const [],
    this.status = MessageStatus.done,
    this.toolName,
    this.errorText,
    this.emergency,
    this.mode,
    this.symptom,
    this.savedEpisodeAt,
  });

  final ChatRole role;
  final String text;
  final List<Citation> citations;
  final MessageStatus status;

  /// Tên tool đang chạy ("đang tra cứu…"); `null` khi không có tool.
  final String? toolName;

  /// Thông báo lỗi tiếng Việt nếu stream thất bại.
  final String? errorText;

  /// Cảnh báo khẩn cấp kèm theo (hiện thẻ đỏ); `null` nếu không có.
  final EmergencyAlert? emergency;

  /// `extractive` | `cloud` tu su kien `done`; null = chua biet.
  final String? mode;

  /// Triệu chứng server vừa nhận ra (hiện ghi chú "Triệu chứng lặp lại").
  final SymptomNoted? symptom;

  /// Mốc của lần ghi nhớ đã lưu vào két trên máy; `null` = chưa/không lưu.
  final DateTime? savedEpisodeAt;

  bool get isUser => role == ChatRole.user;
  bool get isStreaming => status == MessageStatus.streaming;
  bool get isError => status == MessageStatus.error;

  ChatMessage copyWith({
    String? text,
    List<Citation>? citations,
    MessageStatus? status,
    String? Function()? toolName,
    String? Function()? errorText,
    EmergencyAlert? emergency,
    String? mode,
    SymptomNoted? symptom,
    DateTime? Function()? savedEpisodeAt,
  }) =>
      ChatMessage(
        role: role,
        text: text ?? this.text,
        citations: citations ?? this.citations,
        status: status ?? this.status,
        toolName: toolName != null ? toolName() : this.toolName,
        errorText: errorText != null ? errorText() : this.errorText,
        emergency: emergency ?? this.emergency,
        mode: mode ?? this.mode,
        symptom: symptom ?? this.symptom,
        savedEpisodeAt:
            savedEpisodeAt != null ? savedEpisodeAt() : this.savedEpisodeAt,
      );
}
