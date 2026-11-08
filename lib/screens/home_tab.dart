import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/history_screen.dart';
import 'package:image_processing_app/screens/medicine_search_screen.dart';
import 'package:image_processing_app/screens/nearest_facility_screen.dart';
import 'package:image_processing_app/screens/news_screen.dart';
import 'package:image_processing_app/screens/progress_screen.dart';
import 'package:image_processing_app/widgets/external_link.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/dose_actions.dart';
import 'package:image_processing_app/widgets/low_stock_banner.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Tab "Trang chủ" — lời chào, lịch uống thuốc hôm nay, lối tắt.
class HomeTab extends ConsumerStatefulWidget {
  const HomeTab({
    super.key,
    required this.onOpenReminders,
    required this.onOpenAssistant,
  });

  /// Chuyển sang tab Nhắc thuốc / Trợ lý của [AppShell].
  final VoidCallback onOpenReminders;
  final VoidCallback onOpenAssistant;

  @override
  ConsumerState<HomeTab> createState() => _HomeTabState();
}

class _HomeTabState extends ConsumerState<HomeTab> {
  @override
  void initState() {
    super.initState();
    Future.microtask(() {
      if (!mounted) return;
      ref.read(reminderControllerProvider.notifier).load();
      ref.read(adherenceControllerProvider.notifier).load();
    });
  }

  void _openProgress() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const ProgressScreen()),
    );
  }

  void _openSearch() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const MedicineSearchPage()),
    );
  }

  void _openHistory() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const HistoryScreen()),
    );
  }

  void _openFacilities() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const NearestFacilityScreen()),
    );
  }

  void _openNews() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => const NewsScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final name =
        ref.watch(authControllerProvider).valueOrNull?.user.username.trim() ??
            '';
    final remindersAsync = ref.watch(reminderControllerProvider);

    return Scaffold(
      backgroundColor: neu.background,
      body: SafeArea(
        bottom: false,
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            _GradientHero(
              name: name,
              greeting: name.isNotEmpty
                  ? l10n.homeGreetingNamed(name)
                  : l10n.homeGreeting,
              subtitle: l10n.homeSubtitle,
              searchTooltip: l10n.search,
              searchHint: l10n.homeSearchHint,
              onSearch: _openSearch,
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const LowStockBanner(),
                  const SizedBox(height: 24),
                  NeuSectionTitle(
                    title: l10n.homeTodayTitle,
                    actionLabel: l10n.homeSeeAll,
                    onAction: widget.onOpenReminders,
                  ),
                  const SizedBox(height: 8),
                  _buildReminders(remindersAsync, l10n),
                  const SizedBox(height: 24),
                  const _LifestyleSection(),
                  const SizedBox(height: 24),
                  NeuSectionTitle(title: l10n.homeShortcuts),
                  const SizedBox(height: 12),
                  _ShortcutGrid(
                    tiles: [
                      _ShortcutTile(
                        icon: Icons.chat_bubble_outline,
                        label: l10n.chatTitle,
                        onTap: widget.onOpenAssistant,
                      ),
                      _ShortcutTile(
                        icon: Icons.search,
                        label: l10n.search,
                        onTap: _openSearch,
                      ),
                      _ShortcutTile(
                        icon: Icons.newspaper_outlined,
                        label: l10n.newsTitle,
                        onTap: _openNews,
                      ),
                      _ShortcutTile(
                        icon: Icons.history,
                        label: l10n.history,
                        onTap: _openHistory,
                      ),
                      _ShortcutTile(
                        icon: Icons.local_fire_department_outlined,
                        label: l10n.progressShortcut,
                        onTap: _openProgress,
                      ),
                      _ShortcutTile(
                        icon: Icons.local_hospital_outlined,
                        label: l10n.facilityTitle,
                        onTap: _openFacilities,
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  const MedicalDisclaimer(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildReminders(
    AsyncValue<List<MedicationReminder>> async,
    AppLocalizations l10n,
  ) {
    final empty = _EmptyReminders(
      message: l10n.homeNoReminders,
      onTap: widget.onOpenReminders,
    );
    return async.when(
      loading: () => const Padding(
        padding: EdgeInsets.symmetric(vertical: 24),
        child: Center(
          child: SizedBox(
            width: 24,
            height: 24,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
        ),
      ),
      error: (_, __) => _ErrorReminders(
        // TODO(l10n): thay bằng key homeRemindersLoadFailed khi có.
        message: Localizations.localeOf(context).languageCode == 'vi'
            ? 'Không tải được lịch nhắc thuốc.'
            : 'Could not load reminders.',
        retryLabel: l10n.retry,
        onTap: () => ref.read(reminderControllerProvider.notifier).load(),
      ),
      data: (list) {
        if (list.isEmpty) return empty;
        final sorted = [...list]..sort((a, b) {
            final byHour = a.hour.compareTo(b.hour);
            return byHour != 0 ? byHour : a.minute.compareTo(b.minute);
          });
        final shown = sorted.take(3).toList();
        final takenToday = ref.watch(takenTodayProvider);
        final nextIdx =
            shown.indexWhere((r) => r.id == null || !takenToday.contains(r.id));
        return Column(
          children: [
            for (var i = 0; i < shown.length; i++)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: _ReminderCard(
                  reminder: shown[i],
                  highlight: i == nextIdx,
                  onToggle: (value) => ref
                      .read(reminderControllerProvider.notifier)
                      .toggleActive(shown[i], value),
                ),
              ),
          ],
        );
      },
    );
  }
}

/// Banner gradient trên cùng (lời chào + avatar) với thanh tìm kiếm "nổi" đè
/// lên mép dưới — chỉ đổi cách trình bày, dữ liệu/điều hướng giữ nguyên như
/// `_Header` trước đây.
class _GradientHero extends StatelessWidget {
  const _GradientHero({
    required this.name,
    required this.greeting,
    required this.subtitle,
    required this.searchTooltip,
    required this.searchHint,
    required this.onSearch,
  });

  final String name;
  final String greeting;
  final String subtitle;
  final String searchTooltip;
  final String searchHint;
  final VoidCallback onSearch;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                neu.accent.withOpacity(0.92),
                neu.accent,
                neu.accent.withOpacity(0.78),
              ],
            ),
            borderRadius: const BorderRadius.vertical(
              bottom: Radius.circular(32),
            ),
            boxShadow: [
              BoxShadow(
                color: neu.accent.withOpacity(0.22),
                offset: const Offset(0, 6),
                blurRadius: 16,
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: const BorderRadius.vertical(
              bottom: Radius.circular(32),
            ),
            child: Stack(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 34),
                  child: SafeArea(
                    bottom: false,
                    child: Row(
                      children: [
                        ExcludeSemantics(
                          child: Container(
                            width: 52,
                            height: 52,
                            alignment: Alignment.center,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.white.withOpacity(0.18),
                            ),
                            child: name.isEmpty
                                ? const Icon(Icons.person_outline,
                                    color: Colors.white)
                                : Text(
                                    name.substring(0, 1).toUpperCase(),
                                    style: theme.textTheme.titleLarge?.copyWith(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                greeting,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: theme.textTheme.titleLarge?.copyWith(
                                  color: Colors.white,
                                  fontSize: 22,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                subtitle,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: theme.textTheme.bodyMedium?.copyWith(
                                    fontSize: 15,
                                    color: Colors.white.withOpacity(0.9)),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
                        Semantics(
                          button: true,
                          label: searchTooltip,
                          excludeSemantics: true,
                          child: Tooltip(
                            message: searchTooltip,
                            child: GestureDetector(
                              onTap: onSearch,
                              child: Container(
                                width: AppTheme.minTouchTarget,
                                height: AppTheme.minTouchTarget,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.white.withOpacity(0.18),
                                ),
                                child: const Icon(Icons.search,
                                    color: Colors.white),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        Transform.translate(
          offset: const Offset(0, -22),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: NeuCard(
              radius: 24,
              onTap: onSearch,
              semanticLabel: searchHint,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
              child: ConstrainedBox(
                constraints: const BoxConstraints(minHeight: 24),
                child: Row(
                  children: [
                    Icon(Icons.search, color: neu.accent),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        searchHint,
                        style: theme.textTheme.bodyLarge
                            ?.copyWith(color: neu.textMuted),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// "Lời khuyên cho bạn" — gợi ý lối sống từ `/v1/agent/lifestyle` (tối đa 3).
class _LifestyleSection extends ConsumerWidget {
  const _LifestyleSection();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final async = ref.watch(lifestyleProvider);

    Widget body;
    if (async.isLoading && !async.hasValue) {
      body = const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Center(
          child: SizedBox(
            width: 24,
            height: 24,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
        ),
      );
    } else if (async.hasError && !async.hasValue) {
      body = NeuCard(
        onTap: () => ref.invalidate(lifestyleProvider),
        semanticLabel: l10n.lifestyleError,
        child: Row(
          children: [
            Icon(Icons.error_outline, color: theme.colorScheme.error),
            const SizedBox(width: 12),
            Expanded(child: Text(l10n.lifestyleError)),
            Icon(Icons.refresh, color: neu.accent),
          ],
        ),
      );
    } else {
      final snap = async.requireValue;
      final tips = snap.tips.take(3).toList();
      if (tips.isEmpty) {
        body = NeuCard(
          child: Row(
            children: [
              Icon(Icons.lightbulb_outline, color: neu.accent),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  l10n.lifestyleEmpty,
                  style:
                      theme.textTheme.bodyLarge?.copyWith(color: neu.textMuted),
                ),
              ),
            ],
          ),
        );
      } else {
        body = Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (var i = 0; i < tips.length; i++) ...[
              if (i > 0) const SizedBox(height: 10),
              // Chỉ gợi ý chung (không trích dẫn) được Claude diễn đạt lại; lưu ý theo thuốc
              // và gợi ý ăn uống là nguyên văn tài liệu -> không gắn nhãn "AI" cho chúng.
              _TipTile(
                tip: tips[i],
                isAi: snap.isAi && tips[i].kind == 'general',
              ),
            ],
            if (snap.unmatchedConditions.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                l10n.modeGUnmatched(snap.unmatchedConditions.join(', ')),
                style: theme.textTheme.bodySmall?.copyWith(
                    color: neu.textMuted, height: 1.35, fontSize: 12.5),
              ),
            ],
            if (snap.disclaimer.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                snap.disclaimer,
                style: theme.textTheme.bodySmall?.copyWith(
                    color: neu.textMuted, height: 1.35, fontSize: 12.5),
              ),
            ],
          ],
        );
      }
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(child: NeuSectionTitle(title: l10n.lifestyleTitle)),
            IconButton(
              tooltip: l10n.lifestyleRefresh,
              icon: const Icon(Icons.refresh),
              onPressed: async.isLoading
                  ? null
                  : () => ref.invalidate(lifestyleProvider),
            ),
          ],
        ),
        const SizedBox(height: 8),
        body,
        const SizedBox(height: 8),
        Text(
          l10n.fixELifestyleNote,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

class _TipTile extends StatelessWidget {
  const _TipTile({required this.tip, required this.isAi});

  final LifestyleTip tip;
  final bool isAi;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final source = tip.isMedication && tip.citations.isNotEmpty
        ? tip.citations.first.medicineName
        : '';
    return NeuCard(
      radius: 20,
      onTap: () => _showTipSheet(context, tip),
      semanticLabel: tip.title,
      padding: const EdgeInsets.fromLTRB(14, 14, 8, 14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: tip.isDiet ? _dietGreenSoft : neu.accentSoft),
            child: Icon(
              tip.isMedication
                  ? Icons.medication_outlined
                  : tip.isDiet
                      ? Icons.restaurant_outlined
                      : Icons.lightbulb_outline,
              color: tip.isDiet ? _dietGreen : neu.accent,
              size: 22,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        tip.title,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontSize: 17,
                          color: neu.textStrong,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                    if (isAi) ...[
                      const SizedBox(width: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: neu.accentSoft,
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Text(
                          l10n.lifestyleAiBadge,
                          style: theme.textTheme.labelSmall?.copyWith(
                            color: neu.accent,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  tip.body,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(fontSize: 15, color: neu.textMuted),
                ),
                if (tip.isDiet) ...[
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: [
                      if (tip.citations.isNotEmpty &&
                          tip.citations.first.source.isNotEmpty)
                        Text(
                          l10n.lifestyleSource(tip.citations.first.source),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: theme.textTheme.labelMedium
                              ?.copyWith(color: neu.textMuted),
                        ),
                      if (tip.isUnverified) const _UnverifiedBadge(),
                    ],
                  ),
                ],
                if (source.isNotEmpty) ...[
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: neu.accentSoft,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      l10n.lifestyleSource(source),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.labelMedium?.copyWith(
                        fontSize: 13,
                        color: neu.accent,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Icon(Icons.chevron_right, color: neu.textMuted),
          ),
        ],
      ),
    );
  }
}

const Color _dietGreen = Color(0xFF2E7D32);
const Color _dietGreenSoft = Color(0xFFDCF1DD);

/// Amber badge: sample data source, not verified.
class _UnverifiedBadge extends StatelessWidget {
  const _UnverifiedBadge();

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF0C2),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: const Color(0xFFE0A100)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.warning_amber_rounded,
              size: 14, color: Color(0xFF7A5300)),
          const SizedBox(width: 4),
          Flexible(
            child: Text(
              l10n.modeGUnverified,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: Color(0xFF7A5300),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

void _showTipSheet(BuildContext context, LifestyleTip tip) {
  showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
    ),
    builder: (sheetCtx) {
      final l10n = AppLocalizations.of(sheetCtx);
      final neu = NeuTheme.of(sheetCtx);
      final theme = Theme.of(sheetCtx);
      final cits = tip.citations;
      return SafeArea(
        child: ConstrainedBox(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(sheetCtx).size.height * 0.85,
          ),
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  tip.title,
                  style: theme.textTheme.titleLarge?.copyWith(
                      color: neu.textStrong, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 10),
                Text(
                  tip.body,
                  style: theme.textTheme.bodyLarge
                      ?.copyWith(color: neu.textStrong, height: 1.45),
                ),
                if (tip.isDiet && tip.isUnverified && cits.isEmpty) ...[
                  const SizedBox(height: 12),
                  const _UnverifiedBadge(),
                ],
                for (final c in cits) ...[
                  const SizedBox(height: 18),
                  if (c.medicineName.isNotEmpty)
                    Text(
                      c.medicineName,
                      style: theme.textTheme.titleMedium?.copyWith(
                          color: neu.textStrong, fontWeight: FontWeight.w700),
                    ),
                  if (c.excerpt.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: neu.accentSoft.withOpacity(0.6),
                        borderRadius: BorderRadius.circular(12),
                        border: Border(
                          left: BorderSide(color: neu.accent, width: 4),
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            c.excerpt,
                            style: theme.textTheme.bodyMedium?.copyWith(
                                color: neu.textStrong,
                                fontStyle: FontStyle.italic,
                                height: 1.4),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            [
                              if (c.id.isNotEmpty) l10n.polishDQuoteLabel(c.id),
                              if (c.source.isNotEmpty) c.source,
                            ].join(' - '),
                            style: theme.textTheme.labelMedium
                                ?.copyWith(color: neu.textMuted),
                          ),
                        ],
                      ),
                    ),
                  ],
                  const SizedBox(height: 8),
                  SourceBlock(
                    source: c.source,
                    isAuthoritative: c.isAuthoritative,
                    retrievedAt: c.retrievedAt,
                    license: c.license,
                    reviewedBy: c.reviewedBy,
                    sourceUrl: c.sourceUrl,
                  ),
                ],
                const SizedBox(height: 16),
                const MedicalDisclaimer(compact: true),
                if (tip.isMedication &&
                    cits.isNotEmpty &&
                    cits.first.medicineName.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      icon: const Icon(Icons.medication_outlined),
                      label: Text(l10n.polishDViewMedicine),
                      onPressed: () {
                        final name = cits.first.medicineName;
                        Navigator.of(sheetCtx).pop();
                        Navigator.of(context).push(
                          MaterialPageRoute<void>(
                            builder: (_) =>
                                MedicineSearchPage(initialQuery: name),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      );
    },
  );
}

class _ReminderCard extends StatelessWidget {
  const _ReminderCard({
    required this.reminder,
    required this.onToggle,
    this.highlight = false,
  });

  final MedicationReminder reminder;
  final ValueChanged<bool> onToggle;

  /// Liều tiếp theo chưa uống: hiện giờ lớn + nhãn "Liều tiếp theo".
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final l10n = AppLocalizations.of(context);
    final dosage = reminder.dosage?.trim() ?? '';
    return NeuCard(
      radius: highlight ? 24 : 20,
      color: highlight ? neu.accentSoft : null,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (highlight) ...[
            Row(
              children: [
                Icon(Icons.alarm_on_rounded, color: neu.accent),
                const SizedBox(width: 6),
                Text(
                  l10n.doseANextDose,
                  style: theme.textTheme.titleSmall?.copyWith(
                      color: neu.accent, fontWeight: FontWeight.w800),
                ),
                const Spacer(),
                Text(
                  reminder.timeDisplay,
                  style: theme.textTheme.headlineMedium?.copyWith(
                      fontSize: 36,
                      color: neu.accent,
                      fontWeight: FontWeight.w700),
                ),
              ],
            ),
            const SizedBox(height: 10),
          ],
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color:
                      highlight ? neu.accent.withOpacity(0.16) : neu.accentSoft,
                ),
                child: Icon(Icons.medication_outlined, color: neu.accent),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      reminder.medicineName,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: neu.textStrong,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    if (dosage.isNotEmpty)
                      Text(
                        dosage,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodyMedium
                            ?.copyWith(color: neu.textMuted),
                      ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  if (!highlight)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: neu.accentSoft,
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        reminder.timeDisplay,
                        style: theme.textTheme.labelLarge?.copyWith(
                          color: neu.accent,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  Semantics(
                    label: '${reminder.medicineName} ${reminder.timeDisplay}',
                    child: Switch(
                      value: reminder.isActive,
                      onChanged: onToggle,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          DoseActions(reminder: reminder),
        ],
      ),
    );
  }
}

class _EmptyReminders extends StatelessWidget {
  const _EmptyReminders({required this.message, required this.onTap});

  final String message;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: NeuCard(
        onTap: onTap,
        semanticLabel: message,
        child: Row(
          children: [
            Icon(Icons.alarm, color: neu.accent),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message,
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(color: neu.textMuted),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorReminders extends StatelessWidget {
  const _ErrorReminders({
    required this.message,
    required this.retryLabel,
    required this.onTap,
  });

  final String message;
  final String retryLabel;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: NeuCard(
        onTap: onTap,
        semanticLabel: '$message $retryLabel',
        child: Row(
          children: [
            Icon(Icons.error_outline, color: theme.colorScheme.error),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message,
                style:
                    theme.textTheme.bodyLarge?.copyWith(color: neu.textStrong),
              ),
            ),
            const SizedBox(width: 8),
            Icon(Icons.refresh, color: neu.accent),
            const SizedBox(width: 4),
            Text(
              retryLabel,
              style: theme.textTheme.labelLarge?.copyWith(
                color: neu.accent,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ShortcutGrid extends StatelessWidget {
  const _ShortcutGrid({required this.tiles});

  final List<Widget> tiles;

  @override
  Widget build(BuildContext context) {
    // Chiều cao ô theo cỡ chữ để không tràn khi người dùng phóng chữ.
    final scale = MediaQuery.textScalerOf(context).scale(14) / 14;
    final extent = 20 + 48 + 8 + 2 * 14 * 1.25 * scale + 4;
    return GridView(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 4,
        mainAxisSpacing: 12,
        crossAxisSpacing: 8,
        mainAxisExtent: extent,
      ),
      children: tiles,
    );
  }
}

class _ShortcutTile extends StatelessWidget {
  const _ShortcutTile({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return NeuCard(
      onTap: onTap,
      semanticLabel: label,
      radius: 16,
      padding: const EdgeInsets.fromLTRB(4, 10, 4, 8),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: neu.accentSoft,
            ),
            child: Icon(icon, size: 28, color: neu.accent),
          ),
          const SizedBox(height: 6),
          Flexible(
            child: Text(
              label,
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    fontSize: 13,
                    height: 1.2,
                    color: neu.textStrong,
                    fontWeight: FontWeight.w600,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}
