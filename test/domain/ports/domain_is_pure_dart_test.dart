// Day 7 — DoD: trong `lib/domain/**` KHÔNG được có `import 'package:flutter/...'`.
//
// Đây là test **canh kiến trúc**, không phải test hành vi. Nó biến một quy tắc
// chỉ nằm trong tài liệu (dễ vi phạm lúc vội) thành CI đỏ.
//
// Vì sao quan trọng: nếu domain import Flutter thì `lib/domain/` không còn
// chạy được trên Dart VM thuần — mất đúng lợi thế kiểm thử nhanh nhất của dự án.

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Các import bị cấm trong tầng domain (không phụ thuộc nền tảng).
const List<String> forbidden = <String>[
  'package:flutter/',
  'dart:ui',
  'dart:html',
];

List<File> _domainFiles() {
  final domain = Directory('lib${Platform.pathSeparator}domain');
  if (!domain.existsSync()) {
    fail(
      'Không thấy thư mục lib/domain tại "${Directory.current.path}" — '
      'chạy test từ gốc project.',
    );
  }
  return domain
      .listSync(recursive: true)
      .whereType<File>()
      .where((f) => f.path.endsWith('.dart'))
      .toList();
}

void main() {
  test('lib/domain/** không import package:flutter/ (DoD Day 7)', () {
    final offenders = <String>[];

    for (final file in _domainFiles()) {
      final lines = file.readAsLinesSync();
      for (var i = 0; i < lines.length; i++) {
        final line = lines[i].trim();
        if (!line.startsWith('import') && !line.startsWith('export')) {
          continue;
        }
        for (final bad in forbidden) {
          if (line.contains(bad)) {
            offenders.add('${file.path}:${i + 1}: $line');
          }
        }
      }
    }

    expect(
      offenders,
      isEmpty,
      reason: 'Tầng domain phải là Dart thuần (không Flutter). Sửa:\n'
          '${offenders.join('\n')}',
    );
  });

  test('lib/domain/** có ít nhất 1 file — test không im lặng pass', () {
    // Chặn trường hợp đường dẫn sai khiến vòng lặp rỗng và DoD "xanh" giả.
    expect(_domainFiles(), isNotEmpty);
  });

  test('tầng domain không import adapter cụ thể (tránh phụ thuộc vòng)', () {
    final offenders = <String>[];

    for (final file in _domainFiles()) {
      for (final line in file.readAsLinesSync()) {
        final trimmed = line.trim();
        if (!trimmed.startsWith('import')) continue;
        // Adapter/screen là chi tiết cài đặt, domain không được biết tới.
        if (trimmed.contains('/data/adapters/') ||
            trimmed.contains('/screens/') ||
            trimmed.contains('/core/providers')) {
          offenders.add('${file.path}: $trimmed');
        }
      }
    }

    expect(
      offenders,
      isEmpty,
      reason: 'Domain chỉ được biết model + port, không biết adapter/UI.\n'
          '${offenders.join('\n')}',
    );
  });
}
