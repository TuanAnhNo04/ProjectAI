import 'package:flutter_test/flutter_test.dart';

import 'package:image_processing_app/data/adapters/unavailable_voice.dart';
import 'package:image_processing_app/domain/ports/voice_assistant.dart';

String _describe(VoiceInputEvent e) => switch (e) {
      VoicePartial() => 'partial',
      VoiceFinal() => 'final',
      VoiceError() => 'error',
      VoiceListening() => 'listening',
    };

void main() {
  test('Unavailable adapters report not available', () async {
    const input = UnavailableVoiceInput();
    const output = UnavailableVoiceOutput();
    expect(input.isAvailable, isFalse);
    expect(output.isAvailable, isFalse);
    expect(await input.listen().toList(), isEmpty);
    await input.stop();
    await output.speak('xin chao');
    await output.stop();
  });

  test('switch over VoiceInputEvent handles every variant', () {
    expect(_describe(const VoicePartial('a')), 'partial');
    expect(_describe(const VoiceFinal('a')), 'final');
    expect(_describe(const VoiceError('e')), 'error');
    expect(_describe(const VoiceListening()), 'listening');
  });
}
