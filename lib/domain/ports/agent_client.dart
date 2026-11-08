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
    this.source = '',
    this.isAuthoritative = false,
    this.sourceUrl = '',
    this.retrievedAt = '',
    this.license = '',
    this.reviewedBy = '',
  });

  /// Mã tài liệu nguồn, vd `leaflet-paracetamol-01`.
  final String documentId;

  /// Đoạn trích — nên có highlight đúng chỗ khi bấm vào (Day 25).
  final String quote;

  final String? title;

  /// Tên nguồn + metadata xuất xứ. Server cũ không gửi -> chuỗi rỗng (không rõ).
  final String source;
  final bool isAuthoritative;
  final String sourceUrl;
  final String retrievedAt;
  final String license;
  final String reviewedBy;

  @override
  String toString() => 'Citation($documentId)';
}

/// Cảnh báo khẩn cấp — server khớp triệu chứng nguy hiểm; luôn tới TRƯỚC token.
final class EmergencyAlert extends AgentEvent {
  const EmergencyAlert({required this.matched, required this.message});

  /// Các cụm triệu chứng đã khớp (vd `đau ngực`).
  final List<String> matched;

  /// Lời khuyên tiếng Việt do server soạn.
  final String message;

  @override
  String toString() => 'EmergencyAlert(${matched.length})';
}

/// Ngữ cảnh sức khoẻ gửi kèm — CHỈ khi hồ sơ có consent VÀ đã đồng ý AI đám mây.
class HealthContext {
  const HealthContext({
    this.conditions = const [],
    this.allergies = const [],
    this.recentSymptoms = const [],
  });

  final List<String> conditions;
  final List<String> allergies;

  /// Triệu chứng 14 ngày gần nhất (≤5) — CHỈ khi bật trí nhớ + hồ sơ có consent.
  /// Máy chủ chỉ dùng trong lượt hỏi này, không lưu.
  final List<String> recentSymptoms;

  bool get isEmpty =>
      conditions.isEmpty && allergies.isEmpty && recentSymptoms.isEmpty;
}

/// Server vừa nhận ra triệu chứng trong câu hỏi (đến trước token, chỉ với ca
/// không khẩn cấp). [recurring] = lặp lại so với `recent_symptoms` gửi lên.
final class SymptomNoted extends AgentEvent {
  const SymptomNoted({required this.terms, this.recurring = false});

  final List<String> terms;
  final bool recurring;

  @override
  String toString() => 'SymptomNoted(${terms.length}, $recurring)';
}

/// Luồng kết thúc bình thường.
final class Done extends AgentEvent {
  const Done({this.sessionId, this.mode = 'extractive'});

  final String? sessionId;

  /// `extractive` (trich nguyen van) | `cloud` (Claude dien dat lai, da qua
  /// kiem tra bam nguon). Server cu khong gui -> `extractive`.
  final String mode;

  bool get isCloud => mode == 'cloud';

  @override
  String toString() => 'Done($sessionId, $mode)';
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
  /// [currentMedications] (Day 27): danh sách thuốc đang dùng từ hồ sơ local —
  /// gửi kèm để agent cảnh báo tương tác. Không lưu server.
  ///
  /// [cloudAiConsent]: người dùng đã đồng ý dùng AI đám mây (mặc định tắt).
  /// [healthContext]: bệnh nền/dị ứng, `null` = không gửi.
  ///
  /// Lỗi **trước khi** mở được stream (mất mạng, 401) được ném ra như
  /// `AppException`; lỗi **sau** đó thì đến dưới dạng event [Failure].
  Stream<AgentEvent> chat({
    required String sessionId,
    required String message,
    List<String> currentMedications,
    bool cloudAiConsent,
    HealthContext? healthContext,
  });
}
