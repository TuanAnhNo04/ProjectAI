import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/data/adapters/unavailable_voice.dart';
import 'package:image_processing_app/domain/ports/voice_assistant.dart';

/// Nhập giọng nói. Hiện là adapter "không khả dụng"; test/override có thể thay thế.
final voiceInputProvider = Provider<VoiceInput>(
  (ref) => const UnavailableVoiceInput(),
);

/// Đọc to. Hiện là adapter "không khả dụng".
final voiceOutputProvider = Provider<VoiceOutput>(
  (ref) => const UnavailableVoiceOutput(),
);
