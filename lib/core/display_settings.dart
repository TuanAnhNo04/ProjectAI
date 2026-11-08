import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Cỡ chữ trong app — nhân thêm vào cỡ chữ hệ thống (người lớn tuổi cần chữ lớn).
enum TextSizeOption {
  normal(1.0),
  large(1.15),
  extraLarge(1.3);

  const TextSizeOption(this.factor);
  final double factor;
}

/// Lựa chọn hiển thị của người dùng. Không nhạy cảm nên lưu `SharedPreferences`.
@immutable
class DisplaySettings {
  const DisplaySettings({
    this.themeMode = ThemeMode.system,
    this.textSize = TextSizeOption.normal,
  });

  final ThemeMode themeMode;
  final TextSizeOption textSize;

  DisplaySettings copyWith({ThemeMode? themeMode, TextSizeOption? textSize}) =>
      DisplaySettings(
        themeMode: themeMode ?? this.themeMode,
        textSize: textSize ?? this.textSize,
      );
}

class DisplaySettingsController extends Notifier<DisplaySettings> {
  static const String themeKey = 'display_theme_mode';
  static const String textKey = 'display_text_size';

  @override
  DisplaySettings build() {
    _load();
    return const DisplaySettings();
  }

  Future<void> _load() async {
    try {
      final p = await SharedPreferences.getInstance();
      state = DisplaySettings(
        themeMode: _byName(ThemeMode.values, p.getString(themeKey)) ??
            ThemeMode.system,
        textSize: _byName(TextSizeOption.values, p.getString(textKey)) ??
            TextSizeOption.normal,
      );
    } on Object {
      // Không đọc được (vd. test không có plugin) -> giữ mặc định.
    }
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    state = state.copyWith(themeMode: mode);
    await _save(themeKey, mode.name);
  }

  Future<void> setTextSize(TextSizeOption size) async {
    state = state.copyWith(textSize: size);
    await _save(textKey, size.name);
  }

  Future<void> _save(String key, String value) async {
    try {
      await (await SharedPreferences.getInstance()).setString(key, value);
    } on Object {
      // Không ghi được -> lần mở sau quay về mặc định, không ảnh hưởng gì khác.
    }
  }

  static T? _byName<T extends Enum>(List<T> values, String? name) {
    for (final v in values) {
      if (v.name == name) return v;
    }
    return null;
  }
}

final displaySettingsProvider =
    NotifierProvider<DisplaySettingsController, DisplaySettings>(
  DisplaySettingsController.new,
  name: 'displaySettings',
);
