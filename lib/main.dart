import 'package:flutter/material.dart';

import 'package:image_processing_app/core/config/app_config.dart';
import 'screens/login_screen.dart';

void main() {
  // Verify --dart-define: in ra server mà app đang trỏ tới (Day 5).
  debugPrint('MedAssist → API: ${AppConfig.describe()}');
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(title: 'Flutter Demo', home: LoginScreen());
  }
}
