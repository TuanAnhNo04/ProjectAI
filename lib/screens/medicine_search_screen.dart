import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/neu.dart';
import 'package:image_processing_app/widgets/offline_banner.dart';

class MedicineSearchPage extends ConsumerStatefulWidget {
  const MedicineSearchPage({super.key, this.initialQuery});

  /// Nếu có: điền sẵn vào ô tìm và tìm một lần khi mở.
  final String? initialQuery;

  @override
  ConsumerState<MedicineSearchPage> createState() => _MedicineSearchPageState();
}

class _MedicineSearchPageState extends ConsumerState<MedicineSearchPage> {
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    final q = widget.initialQuery?.trim() ?? '';
    if (q.isNotEmpty) {
      _searchController.text = q;
      Future.microtask(() {
        if (mounted) _searchMedicine();
      });
    }
  }

  /// Cảnh báo "chưa tìm" do người dùng gõ — tách khỏi lỗi mạng trong provider
  /// để không mất kết quả đang xem khi báo lỗi.
  String _inputError = '';

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _searchMedicine() async {
    final l10n = AppLocalizations.of(context);
    final name = _searchController.text.trim();
    if (name.isEmpty) {
      setState(() => _inputError = l10n.searchEmptyQuery);
      return;
    }

    setState(() => _inputError = '');
    await ref.read(searchControllerProvider.notifier).search(name);
  }

  @override
  Widget build(BuildContext context) {
    final search = ref.watch(searchControllerProvider);
    final medicines = search.valueOrNull?.results ?? const <Medicine>[];
    final isStale = search.valueOrNull?.isStale ?? false;
    final cachedAt = search.valueOrNull?.cachedAt;
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(
        title: Text(l10n.searchTitle),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
          child: Column(
            children: [
              // Ô tìm kiếm kiểu lõm: nền nhạt của màu nhấn, không viền.
              DecoratedBox(
                decoration: BoxDecoration(
                  color: neu.accentSoft.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: TextField(
                  controller: _searchController,
                  textInputAction: TextInputAction.search,
                  onSubmitted: (_) => _searchMedicine(),
                  style: TextStyle(color: neu.textStrong),
                  decoration: InputDecoration(
                    hintText: l10n.searchHint,
                    hintStyle: TextStyle(color: neu.textMuted),
                    prefixIcon: Icon(Icons.search, color: neu.accent),
                    suffixIcon: IconButton(
                      tooltip: l10n.search,
                      icon: Icon(Icons.arrow_forward, color: neu.accent),
                      onPressed: search.isLoading ? null : _searchMedicine,
                    ),
                    border: InputBorder.none,
                    enabledBorder: InputBorder.none,
                    focusedBorder: InputBorder.none,
                    filled: false,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 16,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: search.isLoading ? null : _searchMedicine,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: neu.accent,
                    foregroundColor: scheme.onPrimary,
                    shape: const StadiumBorder(),
                    minimumSize: const Size.fromHeight(52),
                  ),
                  child: search.isLoading
                      ? SizedBox(
                          height: 22,
                          width: 22,
                          child: CircularProgressIndicator(
                            strokeWidth: 2.5,
                            color: scheme.onPrimary,
                          ),
                        )
                      : Text(l10n.search),
                ),
              ),
              const SizedBox(height: 16),
              // Banner ngoại tuyến: dữ liệu đang xem là bản cache (Day 12).
              if (isStale)
                OfflineBanner(cachedAt: cachedAt, onRetry: _searchMedicine),
              if (_inputError.isNotEmpty)
                _MessageRow(text: _inputError, color: scheme.error),
              if (search.hasError) ...[
                _MessageRow(
                  text: search.error is AppException
                      ? (search.error! as AppException).message
                      : l10n.noResults,
                  color: scheme.error,
                ),
                TextButton(
                  onPressed: _searchMedicine,
                  child: Text(l10n.retry),
                ),
              ],
              Expanded(
                child: search.isLoading
                    ? const Center(child: CircularProgressIndicator())
                    : medicines.isEmpty
                        ? Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.medication_outlined,
                                    size: 48,
                                    color: neu.textMuted.withOpacity(0.6)),
                                const SizedBox(height: 8),
                                Text(
                                  l10n.noResults,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(color: neu.textMuted),
                                ),
                              ],
                            ),
                          )
                        : ListView.builder(
                            padding: const EdgeInsets.only(
                                top: 4, bottom: 24, left: 4, right: 4),
                            itemCount: medicines.length,
                            itemBuilder: (context, index) => Padding(
                              padding: const EdgeInsets.symmetric(vertical: 8),
                              child: _MedicineCard(medicine: medicines[index]),
                            ),
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Dòng thông báo kèm icon — không chỉ dựa vào màu để báo lỗi.
class _MessageRow extends StatelessWidget {
  const _MessageRow({required this.text, required this.color});

  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, size: 20, color: color),
          const SizedBox(width: 8),
          Expanded(child: Text(text, style: TextStyle(color: color))),
        ],
      ),
    );
  }
}

/// Một thẻ thuốc — dựng từ model [Medicine] đã kiểu hoá, không phải `Map`.
class _MedicineCard extends StatelessWidget {
  const _MedicineCard({required this.medicine});

  final Medicine medicine;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    return NeuCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            medicine.name.isEmpty ? l10n.noInformation : medicine.name,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w800,
                  color: neu.textStrong,
                ),
          ),
          const SizedBox(height: 8),
          _row(context, l10n.composition, medicine.composition, l10n),
          _row(context, l10n.uses, medicine.uses, l10n),
          _row(context, l10n.sideEffects, medicine.sideEffects, l10n),
          _row(context, l10n.manufacturer, medicine.manufacturer, l10n),
          _row(context, l10n.reviewExcellent, medicine.excellentReviewPercent,
              l10n),
          _row(
              context, l10n.reviewAverage, medicine.averageReviewPercent, l10n),
          _row(context, l10n.reviewPoor, medicine.poorReviewPercent, l10n),
          const SizedBox(height: 12),
          // Bắt buộc: mọi màn ra kết quả đều phải có cảnh báo y khoa.
          const MedicalDisclaimer(compact: true),
        ],
      ),
    );
  }

  /// `null` -> "Không có thông tin"; số -> "42%". Không còn `?? '0%'` ở mọi
  /// chỗ nên trường thiếu không bị hiển thị thành 0%.
  Widget _row(
    BuildContext context,
    String label,
    Object? value,
    AppLocalizations l10n,
  ) {
    final neu = NeuTheme.of(context);
    final text = switch (value) {
      null => l10n.noInformation,
      final double percent => '${percent.round()}%',
      final String s when s.isEmpty => l10n.noInformation,
      final other => '$other',
    };
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: '$label: ',
              style: TextStyle(color: neu.textMuted),
            ),
            TextSpan(text: text, style: TextStyle(color: neu.textStrong)),
          ],
        ),
      ),
    );
  }
}
