import 'package:image_processing_app/core/network/api_exception.dart';

/// Một sự kiện trong luồng trả lời của agent (SSE — Day 25 hiển thị).
///
/// `sealed` (Dart 3) để trình biên dịch **bắt buộc** xử lý đủ nhánh khi UI
/// `switch` trên event — thêm biến thể mới sau này sẽ sinh cảnh báo compile
/// thay vì rơi nhánh im lặng.
sealed class AgentEvent {
  const AgentEvent();
}

/// Chunk chữ vừa tới — hiển thị ngay để cảm giác như đang gõ.
final class TokenDelta extends AgentEvent {
  const TokenDelta(this.text);

  final String text;

  @override
  String toString() => 'TokenDelta($text)';
}

/// Agent bắt đầu gọi tool — UI hiện "đang tra cứu…".
final class ToolCallStarted extends AgentEvent {
  const ToolCallStarted({required this.name, this.arguments = const {}});

  /// Tên tool, vd `check_interaction`.
  final String name;

  /// Tham số đã **che** (không gửi mật khẩu/token ra log).
  final Map<String, Object?> arguments;

  @override
  String toString() => 'ToolCallStarted($name)';
}

/// Nguồn tham chiếu cho một câu trong câu trả lời — bắt buộc có (RAG Day 20).
final class Citation extends AgentEvent {
  const Citation({
    required this.documentId,
    required this.quote,
    this.title,
  });

  /// Mã tài liệu nguồn, vd `leaflet-paracetamol-01`.
  final String documentId;

  /// Đoạn trích — nên có highlight đúng chỗ khi bấm vào (Day 25).
  final String quote;

  final String? title;

  @override
  String toString() => 'Citation($documentId)';
}

/// Luồng kết thúc bình thường.
final class Done extends AgentEvent {
  const Done({this.sessionId});

  final String? sessionId;

  @override
  String toString() => 'Done($sessionId)';
}

/// Lỗi giữa chừng — **mang [AppException] đã có thông báo tiếng Việt**.
///
/// Vì lỗi nằm trong *stream* chứ không phải ngoại lệ ném ra, adapter vẫn phải
/// đóng vai người gánh: mọi lỗi **phải** được đóng gói thành event này chứ
/// không được làm stream văng exception (UI sẽ mất toàn bộ tin nhắn đã nhận).
final class Failure extends AgentEvent {
  const Failure(this.error);

  final AppException error;

  @override
  String toString() => 'Failure(${error.kind.name}: ${error.message})';
}

/// Cổng trừu tượng cho hội thoại với agent.
abstract interface class AgentClient {
  /// Mở luồng trả lời cho [message] trong [sessionId].
  ///
  /// `sessionId` giữ ngữ cảnh giữa các lượt — đổi sessionId là bắt đầu hội
  /// thoại mới. Sự kiện luôn kết thúc bằng [Done] hoặc [Failure].
  ///
  /// Lỗi **trước khi** mở được stream (mất mạng, 401) được ném ra như
  /// `AppException`; lỗi **sau** đó thì đến dưới dạng event [Failure].
  Stream<AgentEvent> chat({
    required String sessionId,
    required String message,
  });
}