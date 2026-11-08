import 'package:image_processing_app/domain/ports/voice_assistant.dart';

/// Adapter mặc định: chưa có nhận dạng giọng nói (cần package + quyền RECORD_AUDIO
/// do chủ dự án duyệt). Không thu âm, không gọi mạng.
class UnavailableVoiceInput implements VoiceInput {
  const UnavailableVoiceInput();

  @override
  bool get isAvailable => false;

  @override
  Stream<VoiceInputEvent> listen({String localeId = 'vi_VN'}) =>
      const Stream<VoiceInputEvent>.empty();

  @override
  Future<void> stop() async {}
}

/// Adapter mặc định cho đọc to: không làm gì.
class UnavailableVoiceOutput implements VoiceOutput {
  const UnavailableVoiceOutput();

  @override
  bool get isAvailable => false;

  @override
  Future<void> speak(String text, {String localeId = 'vi_VN'}) async {}

  @override
  Future<void> stop() async {}
}
