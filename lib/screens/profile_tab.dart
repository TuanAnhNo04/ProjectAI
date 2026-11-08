import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/display_settings.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/health_profile_screen.dart';
import 'package:image_processing_app/screens/privacy_screen.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Tab "Hồ sơ" — hồ sơ sức khoẻ, quyền riêng tư, đăng xuất.
class ProfileTab extends ConsumerWidget {
  const ProfileTab({super.key});

  Future<void> _confirmSignOut(
      BuildContext context, WidgetRef ref, AppLocalizations l10n) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l10n.signOutConfirmTitle),
        content: Text(l10n.signOutConfirmBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l10n.cancel),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(l10n.signOut),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      // AuthGate tự chuyển về màn Login, không điều hướng thủ công.
      await ref.read(authControllerProvider.notifier).logout();
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;
    final username =
        ref.watch(authControllerProvider).valueOrNull?.user.username;
    final hasName = username != null && username.trim().isNotEmpty;
    final initial =
        hasName ? username.trim().substring(0, 1).toUpperCase() : null;

    return Scaffold(
      appBar: AppBar(title: Text(l10n.profileTitle)),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
        children: [
          NeuCard(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                ExcludeSemantics(
                  child: Container(
                    width: 72,
                    height: 72,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: neu.accentSoft,
                      boxShadow: neu.soft(),
                    ),
                    child: initial != null
                        ? Text(
                            initial,
                            style: textTheme.headlineMedium?.copyWith(
                              color: neu.accent,
                              fontWeight: FontWeight.w800,
                            ),
                          )
                        : Icon(Icons.person, size: 36, color: neu.accent),
                  ),
                ),
                if (hasName) ...[
                  const SizedBox(width: 16),
                  Expanded(
                    child: Text(
                      username,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: textTheme.titleLarge?.copyWith(
                        color: neu.textStrong,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 24),
          NeuSectionTitle(title: l10n.profileAccountSection),
          const SizedBox(height: 12),
          _ProfileTile(
            icon: Icons.favorite_outline,
            title: l10n.healthProfileTitle,
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const HealthProfileScreen()),
            ),
          ),
          const SizedBox(height: 24),
          NeuSectionTitle(title: l10n.displaySection),
          const SizedBox(height: 12),
          const _DisplayCard(),
          const SizedBox(height: 24),
          NeuSectionTitle(title: l10n.profileDataSection),
          const SizedBox(height: 12),
          _ProfileTile(
            icon: Icons.privacy_tip_outlined,
            title: l10n.privacyTitle,
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const PrivacyScreen()),
            ),
          ),
          const SizedBox(height: 24),
          _ProfileTile(
            icon: Icons.logout,
            title: l10n.signOut,
            iconColor: scheme.error,
            showChevron: false,
            onTap: () => _confirmSignOut(context, ref, l10n),
          ),
        ],
      ),
    );
  }
}

/// Chế độ sáng/tối + cỡ chữ (lưu trên máy, áp dụng ngay).
class _DisplayCard extends ConsumerWidget {
  const _DisplayCard();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final textTheme = Theme.of(context).textTheme;
    final settings = ref.watch(displaySettingsProvider);
    final ctrl = ref.read(displaySettingsProvider.notifier);
    return NeuCard(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(l10n.displayTheme, style: textTheme.titleSmall),
          const SizedBox(height: 8),
          SegmentedButton<ThemeMode>(
            key: const ValueKey('themeModeSelector'),
            segments: [
              ButtonSegment(
                  value: ThemeMode.system, label: Text(l10n.displayThemeSystem)),
              ButtonSegment(
                  value: ThemeMode.light, label: Text(l10n.displayThemeLight)),
              ButtonSegment(
                  value: ThemeMode.dark, label: Text(l10n.displayThemeDark)),
            ],
            selected: {settings.themeMode},
            showSelectedIcon: false,
            onSelectionChanged: (s) => ctrl.setThemeMode(s.first),
          ),
          const SizedBox(height: 16),
          Text(l10n.displayTextSize, style: textTheme.titleSmall),
          const SizedBox(height: 8),
          SegmentedButton<TextSizeOption>(
            key: const ValueKey('textSizeSelector'),
            segments: [
              ButtonSegment(
                  value: TextSizeOption.normal,
                  label: Text(l10n.displayTextNormal)),
              ButtonSegment(
                  value: TextSizeOption.large,
                  label: Text(l10n.displayTextLarge)),
              ButtonSegment(
                  value: TextSizeOption.extraLarge,
                  label: Text(l10n.displayTextExtraLarge)),
            ],
            selected: {settings.textSize},
            showSelectedIcon: false,
            onSelectionChanged: (s) => ctrl.setTextSize(s.first),
          ),
        ],
      ),
    );
  }
}

class _ProfileTile extends StatelessWidget {
  const _ProfileTile({
    required this.icon,
    required this.title,
    required this.onTap,
    this.iconColor,
    this.showChevron = true,
  });

  final IconData icon;
  final String title;
  final VoidCallback onTap;
  final Color? iconColor;
  final bool showChevron;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return NeuCard(
      onTap: onTap,
      semanticLabel: title,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: ConstrainedBox(
        constraints: const BoxConstraints(minHeight: 32),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: neu.accentSoft,
              ),
              child: Icon(icon, color: iconColor ?? neu.accent),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                title,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: iconColor ?? neu.textStrong,
                      fontWeight: FontWeight.w700,
                    ),
              ),
            ),
            if (showChevron) Icon(Icons.chevron_right, color: neu.textMuted),
          ],
        ),
      ),
    );
  }
}
