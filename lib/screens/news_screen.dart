import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/news_article.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/external_link.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Màn hình Tin tức thuốc & y tế (Day 31).
///
/// Nguồn dữ liệu do **server** quyết định (`GET /v1/news`) — app không bao giờ
/// gọi RSS trực tiếp, xem lý do ở `server/medassist-core/app/news.py`. Khi chủ
/// dự án chưa bật nguồn thật, server trả bài mẫu và màn này hiện rõ banner
/// "đang dùng nội dung mẫu" — không bao giờ để người dùng tưởng là tin thật.
class NewsScreen extends ConsumerWidget {
  const NewsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final scheme = Theme.of(context).colorScheme;
    final newsAsync = ref.watch(newsProvider);

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(title: Text(l10n.newsTitle)),
      body: RefreshIndicator(
        onRefresh: () => ref.refresh(newsProvider.future),
        child: newsAsync.when(
          loading: () => ListView(
            // ListView (không Center) để RefreshIndicator kéo-để-làm-mới hoạt
            // động ngay cả lúc đang tải lần đầu.
            children: const [
              SizedBox(height: 160),
              Center(child: CircularProgressIndicator()),
            ],
          ),
          error: (err, _) => ListView(
            padding: const EdgeInsets.all(24),
            children: [
              const SizedBox(height: 40),
              Icon(Icons.error_outline, size: 40, color: scheme.error),
              const SizedBox(height: 8),
              Text(
                err is AppException ? err.message : l10n.newsLoadFailed,
                textAlign: TextAlign.center,
                style: TextStyle(color: scheme.error),
              ),
              const SizedBox(height: 16),
              Center(
                child: ElevatedButton(
                  onPressed: () => ref.invalidate(newsProvider),
                  child: Text(l10n.retry),
                ),
              ),
            ],
          ),
          data: (snapshot) => _NewsList(
            snapshot: snapshot,
            l10n: l10n,
            onOpenSource: (url) => openExternalLink(context, ref, url),
          ),
        ),
      ),
    );
  }
}

class _NewsList extends StatelessWidget {
  const _NewsList({
    required this.snapshot,
    required this.l10n,
    required this.onOpenSource,
  });

  final NewsSnapshot snapshot;
  final AppLocalizations l10n;
  final ValueChanged<String> onOpenSource;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);

    if (snapshot.articles.isEmpty) {
      return ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const SizedBox(height: 40),
          Icon(Icons.newspaper_outlined, size: 40, color: neu.textMuted),
          const SizedBox(height: 8),
          Center(
            child: Text(
              l10n.newsEmpty,
              style: theme.textTheme.bodyLarge?.copyWith(color: neu.textMuted),
            ),
          ),
        ],
      );
    }

    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
      children: [
        if (snapshot.isSample) ...[
          _Banner(
            icon: Icons.science_outlined,
            text: l10n.newsSampleBanner,
          ),
          const SizedBox(height: 12),
        ],
        if (snapshot.failedSources.isNotEmpty) ...[
          _Banner(
            icon: Icons.wifi_off,
            // Nói thật khi một nguồn tải hỏng, thay vì lặng lẽ thiếu tin.
            text:
                '${l10n.newsLoadFailed} (${snapshot.failedSources.join(", ")})',
          ),
          const SizedBox(height: 12),
        ],
        if (!snapshot.isSample && snapshot.verifiedSources.isNotEmpty) ...[
          _Banner(
            icon: Icons.verified_outlined,
            text: l10n.newsVerifiedSources(snapshot.verifiedSources.join(', ')),
          ),
          const SizedBox(height: 12),
        ],
        for (final article in snapshot.articles)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: _ArticleCard(
              article: article,
              openLabel: l10n.newsOpenSource,
              onOpen: article.hasLink ? () => onOpenSource(article.url) : null,
            ),
          ),
        const SizedBox(height: 8),
        const MedicalDisclaimer(compact: true),
      ],
    );
  }
}

class _Banner extends StatelessWidget {
  const _Banner({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: neu.accentSoft.withOpacity(0.6),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: neu.accent),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: neu.textStrong),
            ),
          ),
        ],
      ),
    );
  }
}

class _ArticleCard extends StatelessWidget {
  const _ArticleCard({
    required this.article,
    required this.openLabel,
    required this.onOpen,
  });

  final NewsArticle article;
  final String openLabel;
  final VoidCallback? onOpen;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final meta = [
      article.source,
      if (article.publishedAt != null) _formatDate(article.publishedAt!),
    ].join(' · ');

    return NeuCard(
      onTap: onOpen,
      semanticLabel: '${article.title}. $meta',
      padding: const EdgeInsets.all(14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (article.imageUrl != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.network(
                article.imageUrl!,
                width: 76,
                height: 76,
                fit: BoxFit.cover,
                // Ảnh từ báo bên ngoài có thể lỗi hoặc hết hạn — không được làm
                // vỡ cả thẻ tin.
                errorBuilder: (_, __, ___) => Container(
                  width: 76,
                  height: 76,
                  color: neu.accentSoft,
                  child: Icon(Icons.image_outlined, color: neu.accent),
                ),
              ),
            ),
          if (article.imageUrl != null) const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  article.title,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: neu.textStrong,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (article.summary.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  Text(
                    article.summary,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: neu.textMuted),
                  ),
                ],
                const SizedBox(height: 6),
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        meta,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.labelSmall
                            ?.copyWith(color: neu.textMuted),
                      ),
                    ),
                    if (onOpen != null)
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            openLabel,
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: neu.accent,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          Icon(Icons.open_in_new, size: 14, color: neu.accent),
                        ],
                      ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Ngày đăng ngắn gọn kiểu `28/10`. Không cần giờ/phút/năm ở danh sách tin.
  static String _formatDate(DateTime d) =>
      '${d.day.toString().padLeft(2, '0')}/${d.month.toString().padLeft(2, '0')}';
}
