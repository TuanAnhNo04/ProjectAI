import 'package:flutter/material.dart';

import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Cảnh báo y khoa bắt buộc, hiển thị ở **mọi** màn cho ra kết quả (Day 13).
///
/// Theo quy tắc an toàn của dự án: thông tin trong app là **tham khảo**, không
/// thay thế chẩn đoán/đơn thuốc của bác sĩ hay dược sĩ. Vì vậy câu này:
/// - **luôn** hiển thị, không có tùy chọn tắt;
/// - **không** dùng từ khẳng định ("chính xác", "đảm bảo") để tránh tạo cảm giác
///   chẩn đoán;
/// - nội dung đưa vào file `.arb` để dịch được, không hardcode trong widget.
class MedicalDisclaimer extends StatelessWidget {
  const MedicalDisclaimer({super.key, this.compact = false});

  /// Bản nhỏ gọn (1–2 dòng) cho danh sách dài.
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;

    return Container(
      width: double.infinity,
      padding: EdgeInsets.symmetric(
        horizontal: 12,
        vertical: compact ? 8 : 12,
      ),
      decoration: BoxDecoration(
        color: scheme.surfaceVariant.withOpacity(0.6),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: scheme.outlineVariant),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, size: 18, color: scheme.onSurfaceVariant),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              // Lấy từ AppLocalizations; fallback là chuỗi tiếng Việt để widget
              // vẫn dùng được trong test không bọc Localizations.
              _text(context),
              style: theme.textTheme.bodySmall?.copyWith(
                color: scheme.onSurfaceVariant,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _text(BuildContext context) {
    final localizations = Localizations.of<AppLocalizations>(
      context,
      AppLocalizations,
    );
    return localizations?.medicalDisclaimer ?? _fallback;
  }

  /// Nội dung dự phòng khi không có `Localizations` (ví dụ test đơn vị).
  /// Phải **khớp** với `app_vi.arb` để không lệch về ngôn ngữ.
  static const String _fallback =
      'Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán hoặc đơn thuốc '
      'của bác sĩ.';
}
