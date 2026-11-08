import 'package:shared_preferences/shared_preferences.dart';

/// Cờ "cho trợ lý nhớ triệu chứng" — chỉ là công tắc (không nhạy cảm), mặc định TẮT.
/// Dữ liệu được nhớ nằm trong `HealthVault` (mã hoá), không nằm ở đây.
abstract interface class AssistantMemoryStore {
  Future<bool> read();
  Future<void> write(bool enabled);
}

class PreferencesAssistantMemoryStore implements AssistantMemoryStore {
  static const String key = 'assistant_memory_enabled';

  @override
  Future<bool> read() async {
    try {
      return (await SharedPreferences.getInstance()).getBool(key) ?? false;
    } on Object {
      return false;
    }
  }

  @override
  Future<void> write(bool enabled) async {
    try {
      final p = await SharedPreferences.getInstance();
      enabled ? await p.setBool(key, true) : await p.remove(key);
    } on Object {
      // Không ghi được -> giữ mặc định TẮT ở lần đọc sau.
    }
  }
}

class InMemoryAssistantMemoryStore implements AssistantMemoryStore {
  InMemoryAssistantMemoryStore({this.value = false});
  bool value;

  @override
  Future<bool> read() async => value;

  @override
  Future<void> write(bool enabled) async => value = enabled;
}
