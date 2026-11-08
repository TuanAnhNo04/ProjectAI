/// Cổng (port) cho trợ lý giọng nói: nhập giọng nói (STT) và đọc to (TTS).
///
/// QUY TẮC RIÊNG TƯ (AGENTS.md mục 2): giọng nói và bản ghi (transcript) là dữ liệu
/// sức khỏe. Mọi adapter gửi âm thanh hoặc transcript ra khỏi thiết bị BẮT BUỘC có
/// đủ: (a) đồng ý rõ ràng của người dùng trong UI, (b) khử định danh trước khi gửi,
/// (c) tài liệu hóa trong `docs/PRIVACY.md`. Ưu tiên nhận dạng ngay trên thiết bị
/// (on-device). Không được ghi log nội dung transcript.
library;

/// Sự kiện phát ra trong lúc nghe.
sealed class VoiceInputEvent {
  const VoiceInputEvent();
}

/// Bản nhận dạng tạm thời (có thể thay đổi).
final class VoicePartial extends VoiceInputEvent {
  const VoicePartial(this.text);
  final String text;
}

/// Bản nhận dạng cuối cùng cho một lượt nói.
final class VoiceFinal extends VoiceInputEvent {
  const VoiceFinal(this.text);
  final String text;
}

/// Lỗi khi nhận dạng (thông điệp không chứa dữ liệu người dùng).
final class VoiceError extends VoiceInputEvent {
  const VoiceError(this.message);
  final String message;
}

/// Đã bắt đầu nghe.
final class VoiceListening extends VoiceInputEvent {
  const VoiceListening();
}

/// Nhập giọng nói. Ưu tiên on-device; xem quy tắc riêng tư ở đầu file.
abstract interface class VoiceInput {
  /// `false` nếu thiết bị/bản dựng chưa hỗ trợ — UI phải ẩn nút mic.
  bool get isAvailable;

  /// Bắt đầu nghe, trả về luồng sự kiện; luồng kết thúc khi phiên nghe xong.
  Stream<VoiceInputEvent> listen({String localeId = 'vi_VN'});

  /// Dừng nghe.
  Future<void> stop();
}

/// Đọc to văn bản. Nếu adapter gửi văn bản ra ngoài thiết bị, áp dụng quy tắc riêng
/// tư ở đầu file.
abstract interface class VoiceOutput {
  bool get isAvailable;

  Future<void> speak(String text, {String localeId = 'vi_VN'});

  Future<void> stop();
}
