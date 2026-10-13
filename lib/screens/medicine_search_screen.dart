import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/offline_banner.dart';

class MedicineSearchPage extends ConsumerStatefulWidget {
  const MedicineSearchPage({super.key});

  @override
  ConsumerState<MedicineSearchPage> createState() => _MedicineSearchPageState();
}

class _MedicineSearchPageState extends ConsumerState<MedicineSearchPage> {
  final TextEditingController _searchController = TextEditingController();

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

    return Scaffold(
      appBar: AppBar(
        title: Text(l10n.searchTitle),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              TextField(
                controller: _searchController,
                textInputAction: TextInputAction.search,
                onSubmitted: (_) => _searchMedicine(),
                decoration: InputDecoration(
                  labelText: l10n.searchHint,
                  prefixIcon: const Icon(Icons.search),
                  border: const OutlineInputBorder(),
                  filled: true,
                  fillColor: Colors.white,
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                onPressed: search.isLoading ? null : _searchMedicine,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  padding: const EdgeInsets.symmetric(
                    vertical: 20,
                    horizontal: 40,
                  ),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                child: search.isLoading
                    ? const SizedBox(
                        height: 22,
                        width: 22,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.5,
                          color: Colors.white,
                        ),
                      )
                    : Text(l10n.search),
              ),
              const SizedBox(height: 20),
              // Banner ngoại tuyến: dữ liệu đang xem là bản cache (Day 12).
              if (isStale)
                OfflineBanner(cachedAt: cachedAt, onRetry: _searchMedicine),
              if (_inputError.isNotEmpty)
                Text(_inputError, style: const TextStyle(color: Colors.red)),
              if (search.hasError) ...[
                Text(
                  search.error is AppException
                      ? (search.error! as AppException).message
                      : 'Không tìm thấy thông tin thuốc!',
                  style: const TextStyle(color: Colors.red),
                ),
                const SizedBox(height: 8),
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
                            child: Text(
                              l10n.noResults,
                              style: const TextStyle(color: Colors.grey),
                            ),
                          )
                        : ListView.builder(
                            itemCount: medicines.length,
                            itemBuilder: (context, index) =>
                                _MedicineCard(medicine: medicines[index]),
                          ),
              ),
            ],
          ),
        ),
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
    return Card(
      elevation: 4,
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              medicine.name.isEmpty ? l10n.noInformation : medicine.name,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
            const SizedBox(height: 8),
            _row(l10n.composition, medicine.composition, l10n),
            _row(l10n.uses, medicine.uses, l10n),
            _row(l10n.sideEffects, medicine.sideEffects, l10n),
            _row(l10n.manufacturer, medicine.manufacturer, l10n),
            _row(l10n.reviewExcellent, medicine.excellentReviewPercent, l10n),
            _row(l10n.reviewAverage, medicine.averageReviewPercent, l10n),
            _row(l10n.reviewPoor, medicine.poorReviewPercent, l10n),
            const SizedBox(height: 8),
            // Bắt buộc: mọi màn ra kết quả đều phải có cảnh báo y khoa.
            const MedicalDisclaimer(compact: true),
          ],
        ),
      ),
    );
  }

  /// `null` -> "Không có thông tin"; số -> "42%". Không còn `?? '0%'` ở mọi
  /// chỗ nên trường thiếu không bị hiển thị thành 0%.
  Widget _row(String label, Object? value, AppLocalizations l10n) {
    final text = switch (value) {
      null => l10n.noInformation,
      final double percent => '${percent.round()}%',
      final String s when s.isEmpty => l10n.noInformation,
      final other => '$other',
    };
    return Text('$label: $text');
  }
}
