import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/display_settings.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('defaults to system theme and normal text', () async {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    final c = ProviderContainer();
    addTearDown(c.dispose);
    final s = c.read(displaySettingsProvider);
    expect(s.themeMode, ThemeMode.system);
    expect(s.textSize, TextSizeOption.normal);
  });

  test('saves and restores choices', () async {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    final c1 = ProviderContainer();
    c1.read(displaySettingsProvider);
    await c1
        .read(displaySettingsProvider.notifier)
        .setThemeMode(ThemeMode.dark);
    await c1
        .read(displaySettingsProvider.notifier)
        .setTextSize(TextSizeOption.extraLarge);
    c1.dispose();

    final c2 = ProviderContainer();
    addTearDown(c2.dispose);
    c2.read(displaySettingsProvider);
    await Future<void>.delayed(Duration.zero);
    final s = c2.read(displaySettingsProvider);
    expect(s.themeMode, ThemeMode.dark);
    expect(s.textSize, TextSizeOption.extraLarge);
  });

  test('unknown stored values fall back to defaults', () async {
    SharedPreferences.setMockInitialValues(<String, Object>{
      DisplaySettingsController.themeKey: 'purple',
      DisplaySettingsController.textKey: 'huge',
    });
    final c = ProviderContainer();
    addTearDown(c.dispose);
    c.read(displaySettingsProvider);
    await Future<void>.delayed(Duration.zero);
    expect(c.read(displaySettingsProvider).themeMode, ThemeMode.system);
    expect(c.read(displaySettingsProvider).textSize, TextSizeOption.normal);
  });
}
