import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Chặn cửa vào app cho tới khi người dùng đồng ý chính sách dữ liệu.
///
/// - Đang đọc trạng thái -> spinner (không nháy màn đồng ý).
/// - Chưa đồng ý -> [ConsentScreen].
/// - Đã đồng ý -> [child] (màn Login).
class ConsentGate extends ConsumerWidget {
  const ConsentGate({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final consent = ref.watch(consentControllerProvider);
    return consent.when(
      data: (accepted) => accepted ? child : const ConsentScreen(),
      loading: () => const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      ),
      // Đọc kho lỗi -> hỏi lại cho chắc, không bỏ qua đồng ý.
      error: (_, __) => const ConsentScreen(),
    );
  }
}

/// Màn đồng ý chính sách dữ liệu (nội dung đầy đủ: `docs/PRIVACY.md`).
class ConsentScreen extends ConsumerWidget {
  const ConsentScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final points = <(IconData, String)>[
      (Icons.phone_android, l10n.consentPointLocal),
      (Icons.cloud_outlined, l10n.consentPointServer),
      (Icons.medical_information_outlined, l10n.consentPointMedical),
      (Icons.delete_outline, l10n.consentPointDelete),
    ];

    return Scaffold(
      appBar: AppBar(title: Text(l10n.consentTitle)),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  Text(l10n.consentIntro),
                  const SizedBox(height: 16),
                  for (final (icon, text) in points)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 14),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(icon),
                          const SizedBox(width: 12),
                          Expanded(child: Text(text)),
                        ],
                      ),
                    ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  FilledButton(
                    onPressed: () =>
                        ref.read(consentControllerProvider.notifier).accept(),
                    child: Text(l10n.consentAccept),
                  ),
                  TextButton(
                    // Không đồng ý thì không dùng app: thoát thay vì vào màn khác.
                    onPressed: () => SystemNavigator.pop(),
                    child: Text(l10n.consentDecline),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
