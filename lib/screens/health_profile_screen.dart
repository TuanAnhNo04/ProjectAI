import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/reminder_screen.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Màn hình quản lý hồ sơ sức khoẻ — Day 27.
///
/// Yêu cầu consent rõ ràng trước khi lưu; xoá được toàn bộ.
class HealthProfileScreen extends ConsumerStatefulWidget {
  const HealthProfileScreen({super.key});

  @override
  ConsumerState<HealthProfileScreen> createState() =>
      _HealthProfileScreenState();
}

class _HealthProfileScreenState extends ConsumerState<HealthProfileScreen> {
  final _allergiesController = TextEditingController();
  final _conditionsController = TextEditingController();
  final _medsController = TextEditingController();

  @override
  void initState() {
    super.initState();
    Future.microtask(_load);
  }

  @override
  void dispose() {
    _allergiesController.dispose();
    _conditionsController.dispose();
    _medsController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    final profile =
        await ref.read(healthProfileControllerProvider.notifier).load();
    if (profile != null && mounted) {
      _allergiesController.text = profile.allergies.join(', ');
      _conditionsController.text = profile.conditions.join(', ');
      _medsController.text = profile.currentMedications.join(', ');
    }
  }

  @override
  Widget build(BuildContext context) {
    final profileAsync = ref.watch(healthProfileControllerProvider);
    final profile = profileAsync.valueOrNull;
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l10n.healthProfileTitle),
        actions: [
          IconButton(
            icon: const Icon(Icons.alarm),
            tooltip: l10n.reminderTitle,
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const ReminderScreen()),
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Consent banner
              if (profile == null || !profile.hasConsent) ...[
                NeuCard(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.verified_user_outlined, color: neu.accent),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              l10n.healthConsentTitle,
                              style: theme.textTheme.titleMedium?.copyWith(
                                color: neu.textStrong,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        l10n.healthConsentBody,
                        style: TextStyle(color: neu.textStrong, height: 1.4),
                      ),
                      const SizedBox(height: 16),
                      FilledButton(
                        onPressed: _giveConsent,
                        child: Text(l10n.healthConsentAccept),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
              ],

              // Form fields
              _buildField(
                label: l10n.healthAllergies,
                hint: l10n.healthAllergiesHint,
                controller: _allergiesController,
                icon: Icons.warning_amber,
              ),
              const SizedBox(height: 16),
              _buildField(
                label: l10n.healthConditions,
                hint: l10n.healthConditionsHint,
                controller: _conditionsController,
                icon: Icons.local_hospital,
              ),
              const SizedBox(height: 16),
              _buildField(
                label: l10n.healthCurrentMeds,
                hint: l10n.healthCurrentMedsHint,
                controller: _medsController,
                icon: Icons.medication,
              ),
              const SizedBox(height: 24),

              // Save + Delete
              FilledButton.icon(
                onPressed: (profile?.hasConsent ?? false) ? _save : null,
                icon: const Icon(Icons.save),
                label: Text(l10n.healthSave),
              ),
              if (profile != null) ...[
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed: () => _confirmDelete(l10n),
                  icon: Icon(Icons.delete_outline,
                      color: theme.colorScheme.error),
                  label: Text(
                    l10n.healthDelete,
                    style: TextStyle(color: theme.colorScheme.error),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildField({
    required String label,
    required String hint,
    required TextEditingController controller,
    required IconData icon,
  }) {
    return TextField(
      controller: controller,
      maxLines: 2,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon: Icon(icon),
      ),
    );
  }

  Future<void> _giveConsent() async {
    await ref.read(healthProfileControllerProvider.notifier).giveConsent();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text(AppLocalizations.of(context).healthConsentGiven)),
      );
    }
  }

  Future<void> _save() async {
    final profile = HealthProfile(
      allergies: _parseList(_allergiesController.text),
      conditions: _parseList(_conditionsController.text),
      currentMedications: _parseList(_medsController.text),
      consentGivenAt:
          ref.read(healthProfileControllerProvider).valueOrNull?.consentGivenAt,
    );
    await ref.read(healthProfileControllerProvider.notifier).save(profile);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(AppLocalizations.of(context).healthSaved)),
      );
    }
  }

  List<String> _parseList(String input) =>
      input.split(',').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();

  void _confirmDelete(AppLocalizations l10n) {
    showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l10n.healthDeleteTitle),
        content: Text(l10n.healthDeleteBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l10n.cancel),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(l10n.delete),
          ),
        ],
      ),
    ).then((confirmed) async {
      if (confirmed == true) {
        await ref
            .read(healthProfileControllerProvider.notifier)
            .deleteProfile();
        if (mounted) {
          _allergiesController.clear();
          _conditionsController.clear();
          _medsController.clear();
        }
      }
    });
  }
}
