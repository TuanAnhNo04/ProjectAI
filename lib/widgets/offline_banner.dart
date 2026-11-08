import 'package:flutter/material.dart';

import 'package:image_processing_app/core/theme/app_theme.dart';
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
    final neu = NeuTheme.of(context);
    final stamp = formatCachedAt(cachedAt);
    final message = stamp.isEmpty ? l10n.offline : l10n.offlineUpdated(stamp);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.fromLTRB(14, 8, 8, 8),
      decoration: BoxDecoration(
        color: neu.accentSoft,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        children: [
          Icon(Icons.wifi_off, size: 20, color: neu.accent),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: neu.textStrong,
                  ),
            ),
          ),
          const SizedBox(width: 8),
          TextButton(
            onPressed: onRetry,
            style: TextButton.styleFrom(
              foregroundColor: neu.accent,
              minimumSize: const Size(48, 48),
              shape: const StadiumBorder(),
              textStyle: const TextStyle(fontWeight: FontWeight.w700),
            ),
            child: Text(l10n.retry),
          ),
        ],
      ),
    );
  }
}
