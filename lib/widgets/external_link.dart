import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/adapters/device_channel.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Parse + kiểm tra lại một URL nguồn: chỉ `https`, có host, KHÔNG userinfo,
/// KHÔNG cổng khác 443. Trả `null` nếu không an toàn (không bao giờ mở).
Uri? safeExternalUri(String? url) {
  final raw = url?.trim() ?? '';
  if (raw.isEmpty) return null;
  final uri = Uri.tryParse(raw);
  if (uri == null) return null;
  if (uri.scheme != 'https' || uri.host.isEmpty) return null;
  if (uri.userInfo.isNotEmpty) return null;
  if (uri.hasPort && uri.port != 443) return null;
  return uri;
}

/// Hỏi xác nhận rồi mở [url] trong trình duyệt của máy.
///
/// Không dùng `url_launcher` (AGENTS.md: gói mới phải được duyệt): dùng kênh gốc
/// `medassist/device`. Trang bên ngoài sẽ thấy IP/trình duyệt của người dùng nên
/// PHẢI báo trước; URL là link công khai, không chứa dữ liệu của người dùng.
/// URL không hợp lệ hoặc kênh native lỗi -> chép link vào clipboard.
Future<void> openExternalLink(
    BuildContext context, WidgetRef ref, String url) async {
  final l10n = AppLocalizations.of(context);
  final messenger = ScaffoldMessenger.of(context);
  final uri = safeExternalUri(url);
  if (uri == null) {
    await Clipboard.setData(ClipboardData(text: url));
    messenger.showSnackBar(SnackBar(content: Text(l10n.newsOpenSource)));
    return;
  }
  final choice = await showDialog<String>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: Text(l10n.newsLeaveTitle),
      content: Text(l10n.newsLeaveBody(uri.host)),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(ctx),
          child: Text(l10n.cancel),
        ),
        TextButton(
          onPressed: () => Navigator.pop(ctx, 'copy'),
          child: Text(l10n.newsLeaveCopy),
        ),
        FilledButton(
          onPressed: () => Navigator.pop(ctx, 'open'),
          child: Text(l10n.newsLeaveOpen),
        ),
      ],
    ),
  );
  if (choice == 'copy') {
    await Clipboard.setData(ClipboardData(text: url));
    messenger.showSnackBar(SnackBar(content: Text(l10n.newsOpenSource)));
  } else if (choice == 'open') {
    try {
      await ref.read(deviceChannelProvider).openUrl(url);
    } on DeviceChannelException {
      await Clipboard.setData(ClipboardData(text: url));
      messenger.showSnackBar(SnackBar(content: Text(l10n.newsOpenSource)));
    }
  }
}

/// `dd/MM/yyyy` theo giờ máy; chuỗi rỗng/không phân tích được -> `null`.
String? formatSourceDate(String? raw) {
  final s = raw?.trim() ?? '';
  if (s.isEmpty) return null;
  final d = DateTime.tryParse(s);
  if (d == null) return null;
  final l = d.toLocal();
  String two(int n) => n.toString().padLeft(2, '0');
  return '${two(l.day)}/${two(l.month)}/${l.year}';
}

const Color _okGreen = Color(0xFF2E7D32);
const Color _warnAmber = Color(0xFF7A5300);

/// Khối "Nguồn" gọn: tên nguồn, ngày lấy, giấy phép, người kiểm duyệt và nút
/// "Mở nguồn gốc" (chỉ khi URL qua [safeExternalUri]; không thì chữ mờ).
class SourceBlock extends ConsumerWidget {
  const SourceBlock({
    super.key,
    this.source = '',
    this.isAuthoritative,
    this.retrievedAt = '',
    this.license = '',
    this.reviewedBy = '',
    this.sourceUrl = '',
  });

  final String source;

  /// `null` = không hiện huy hiệu (ngữ cảnh không có cờ này).
  final bool? isAuthoritative;
  final String retrievedAt;
  final String license;
  final String reviewedBy;
  final String sourceUrl;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final muted = theme.textTheme.labelMedium?.copyWith(color: neu.textMuted);
    final date = formatSourceDate(retrievedAt);
    final uri = safeExternalUri(sourceUrl);
    final reviewer = reviewedBy.trim();
    final lic = license.trim();

    Widget line(IconData icon, Color color, String text, {TextStyle? style}) =>
        Padding(
          padding: const EdgeInsets.only(top: 2),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, size: 14, color: color),
              const SizedBox(width: 4),
              Expanded(child: Text(text, style: style ?? muted)),
            ],
          ),
        );

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: neu.accentSoft.withOpacity(0.35),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            source.trim().isEmpty
                ? l10n.srcJTitle
                : '${l10n.srcJTitle}: ${source.trim()}',
            style: theme.textTheme.labelLarge
                ?.copyWith(color: neu.textStrong, fontWeight: FontWeight.w700),
          ),
          if (isAuthoritative != null) ...[
            const SizedBox(height: 4),
            isAuthoritative!
                ? line(Icons.verified_outlined, _okGreen, l10n.srcJVerified,
                    style: muted?.copyWith(color: _okGreen))
                : line(Icons.warning_amber_rounded, _warnAmber,
                    l10n.modeGUnverified,
                    style: muted?.copyWith(color: _warnAmber)),
          ],
          if (date != null)
            line(Icons.event_outlined, neu.textMuted, l10n.srcJRetrieved(date)),
          if (lic.isNotEmpty)
            line(Icons.gavel_outlined, neu.textMuted, l10n.srcJLicense(lic)),
          if (reviewer.isNotEmpty)
            line(Icons.check_circle, _okGreen, l10n.srcJReviewed(reviewer),
                style: muted?.copyWith(color: _okGreen))
          else
            line(Icons.info_outline, _warnAmber, l10n.srcJNotReviewed,
                style: muted?.copyWith(color: _warnAmber)),
          const SizedBox(height: 4),
          if (uri != null)
            TextButton.icon(
              style: TextButton.styleFrom(
                minimumSize: const Size(48, 48),
                padding: const EdgeInsets.symmetric(horizontal: 8),
                alignment: Alignment.centerLeft,
              ),
              onPressed: () => openExternalLink(context, ref, sourceUrl.trim()),
              icon: Icon(Icons.open_in_new, size: 16, color: neu.accent),
              label: Text(
                l10n.srcJOpenHost(uri.host),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style:
                    TextStyle(color: neu.accent, fontWeight: FontWeight.w700),
              ),
            )
          else
            Text(l10n.srcJNoLink,
                style: muted?.copyWith(fontStyle: FontStyle.italic)),
        ],
      ),
    );
  }
}
