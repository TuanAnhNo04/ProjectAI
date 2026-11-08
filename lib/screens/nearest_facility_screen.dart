import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/adapters/device_channel.dart';
import 'package:image_processing_app/data/models/medical_facility.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Cơ sở y tế gần nhất: nút gọi 115, lấy vị trí (chỉ dùng trên máy), sắp danh
/// sách mẫu theo khoảng cách haversine. Không gọi mạng.
class NearestFacilityScreen extends ConsumerStatefulWidget {
  const NearestFacilityScreen({super.key, this.emergencyMessage});

  /// Thông điệp cảnh báo khẩn (vd. từ thẻ cấp cứu trong chat) hiển thị trên cùng.
  final String? emergencyMessage;

  @override
  ConsumerState<NearestFacilityScreen> createState() =>
      _NearestFacilityScreenState();
}

enum _LocState { idle, locating, ready, error }

class _NearestFacilityScreenState extends ConsumerState<NearestFacilityScreen> {
  _LocState _state = _LocState.idle;
  String? _errorCode;
  DeviceLocation? _location;

  static const _emergencyRed = Color(0xFFD32F2F);

  Future<void> _locate() async {
    setState(() {
      _state = _LocState.locating;
      _errorCode = null;
    });
    try {
      final loc = await ref.read(deviceChannelProvider).getLocation();
      if (!mounted) return;
      setState(() {
        _location = loc;
        _state = _LocState.ready;
      });
    } on DeviceChannelException catch (e) {
      if (!mounted) return;
      setState(() {
        _errorCode = e.code;
        _state = _LocState.error;
      });
    }
  }

  Future<void> _run(Future<void> Function(DeviceChannel ch) action) async {
    final messenger = ScaffoldMessenger.of(context);
    final l10n = AppLocalizations.of(context);
    try {
      await action(ref.read(deviceChannelProvider));
    } on DeviceChannelException catch (e) {
      messenger.showSnackBar(SnackBar(
        content: Text(e.code == 'unsupported'
            ? l10n.facilityUnsupported
            : l10n.facilityOpenFailed),
      ));
    }
  }

  String _errorText(AppLocalizations l10n) => switch (_errorCode) {
        'permission_denied' => l10n.facilityPermissionDenied,
        'location_off' => l10n.facilityLocationOff,
        'unsupported' => l10n.facilityUnsupported,
        _ => l10n.facilityLocationUnavailable,
      };

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final facilitiesAsync = ref.watch(medicalFacilitiesProvider);
    final message = widget.emergencyMessage?.trim();

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(title: Text(l10n.facilityTitle)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
          children: [
            if (message != null && message.isNotEmpty) ...[
              _InfoBox(
                icon: Icons.emergency_outlined,
                text: message,
                color: _emergencyRed,
              ),
              const SizedBox(height: 12),
            ],
            FilledButton.icon(
              onPressed: () => _run((ch) => ch.dial('115')),
              style: FilledButton.styleFrom(
                backgroundColor: _emergencyRed,
                foregroundColor: Colors.white,
                minimumSize: const Size.fromHeight(64),
                shape: const StadiumBorder(),
                textStyle: theme.textTheme.titleLarge
                    ?.copyWith(fontWeight: FontWeight.w800),
              ),
              icon: const Icon(Icons.phone_in_talk, size: 28),
              label: Text(l10n.facilityCall115),
            ),
            const SizedBox(height: 16),
            _InfoBox(
              icon: Icons.info_outline,
              text: l10n.facilitySampleBanner,
              color: Colors.orange.shade800,
            ),
            const SizedBox(height: 12),
            _buildLocationSection(l10n, neu, theme),
            const SizedBox(height: 12),
            facilitiesAsync.when(
              loading: () => const Padding(
                padding: EdgeInsets.all(24),
                child: Center(child: CircularProgressIndicator()),
              ),
              error: (_, __) => Text(l10n.facilityLoadFailed),
              data: (facilities) => _buildList(facilities, l10n, neu, theme),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () =>
                  _run((ch) => ch.searchMap(l10n.facilitySearchQuery)),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size.fromHeight(48),
                shape: const StadiumBorder(),
              ),
              icon: const Icon(Icons.map_outlined),
              label: Text(l10n.facilitySearchMore),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLocationSection(
    AppLocalizations l10n,
    NeuTheme neu,
    ThemeData theme,
  ) {
    return NeuCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.lock_outline, color: neu.accent, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  l10n.facilityLocationRationale,
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(color: neu.textStrong),
                ),
              ),
            ],
          ),
          if (_state == _LocState.error) ...[
            const SizedBox(height: 8),
            Text(
              _errorText(l10n),
              style: theme.textTheme.bodyMedium
                  ?.copyWith(color: theme.colorScheme.error),
            ),
          ],
          const SizedBox(height: 12),
          FilledButton.tonalIcon(
            onPressed: _state == _LocState.locating ? null : _locate,
            icon: _state == _LocState.locating
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2.5),
                  )
                : const Icon(Icons.my_location),
            label: Text(_state == _LocState.locating
                ? l10n.facilityLocating
                : l10n.facilityUseLocation),
          ),
        ],
      ),
    );
  }

  Widget _buildList(
    List<MedicalFacility> facilities,
    AppLocalizations l10n,
    NeuTheme neu,
    ThemeData theme,
  ) {
    final loc = _location;
    final rows = loc == null
        ? [for (final f in facilities) (f, null as double?)]
        : [
            for (final r in sortByDistance(facilities, loc.lat, loc.lng))
              (r.$1, r.$2 as double?),
          ];
    return Column(
      children: [
        for (final (facility, km) in rows)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: NeuCard(
              child: Row(
                children: [
                  const Icon(Icons.local_hospital, color: _emergencyRed),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          facility.name,
                          style: theme.textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w700,
                            color: neu.textStrong,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          facility.address,
                          style: theme.textTheme.bodySmall
                              ?.copyWith(color: neu.textMuted),
                        ),
                        if (km != null) ...[
                          const SizedBox(height: 2),
                          Text(
                            l10n.facilityDistanceKm(km.toStringAsFixed(1)),
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: neu.accent,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  TextButton.icon(
                    onPressed: () => _run(
                      (ch) =>
                          ch.openMap(facility.lat, facility.lng, facility.name),
                    ),
                    icon: const Icon(Icons.directions),
                    label: Text(l10n.facilityDirections),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }
}

class _InfoBox extends StatelessWidget {
  const _InfoBox({required this.icon, required this.text, required this.color});

  final IconData icon;
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.45)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color),
          const SizedBox(width: 8),
          Expanded(child: Text(text)),
        ],
      ),
    );
  }
}
