import 'package:flutter/material.dart';

import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Banner "đang ngoại tuyến" dùng chung cho các màn có cache (Day 12).
///
/// Mục tiêu: người dùng **vẫn thấy dữ liệu** thay vì màn trắng, nhưng luôn
/// biết đó là dữ liệu cũ và có nút thử lại. Không tự xưng "Đang ngoại tuyến"
/// khi [cachedAt] không có — lúc đó hiển thị chung chung, tránh bịa thời gian.
class OfflineBanner extends StatelessWidget {
  const OfflineBanner({super.key, required this.onRetry, this.cachedAt});

  /// Gọi lại dữ liệu từ server (nút "Thử lại").
  final VoidCallback onRetry;

  /// Mốc thời gian dữ liệu cache được ghi; `null` = không rõ.
  final DateTime? cachedAt;

  /// "05/10/2026 14:30" — không giây/phút vì người dùng chỉ cần biết
  /// "cũ bao lâu rồi", và định dạng ngắn gọn khỏi vỡ bố cục trên máy nhỏ.
  static String formatCachedAt(DateTime? at) {
    if (at == null) return '';
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(at.day)}/${two(at.month)}/${at.year} '
        '${two(at.hour)}:${two(at.minute)}';
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final stamp = formatCachedAt(cachedAt);
    final message = stamp.isEmpty ? l10n.offline : l10n.offlineUpdated(stamp);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8.0),
      padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8.0),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF3CD),
        borderRadius: BorderRadius.circular(8.0),
        border: Border.all(color: const Color(0xFFFFC107)),
      ),
      child: Row(
        children: [
          const Icon(
            Icons.wifi_off,
            size: 18,
            color: Color(0xFF3E2723),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(
                fontSize: 13,
                color: Color(0xFF3E2723),
              ),
            ),
          ),
          TextButton(
            onPressed: onRetry,
            child: Text(l10n.retry),
          ),
        ],
      ),
    );
  }
}
