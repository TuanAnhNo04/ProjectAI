import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/domain/prescription_parser.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/prescription_review_screen.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/neu.dart';

class ImageProcessingScreen extends ConsumerStatefulWidget {
  const ImageProcessingScreen({super.key});

  @override
  ConsumerState<ImageProcessingScreen> createState() =>
      _ImageProcessingScreenState();
}

class _ImageProcessingScreenState extends ConsumerState<ImageProcessingScreen> {
  File? _image;

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    // Làm nóng danh mục thuốc cho OCR trên máy (tải nền, có cache).
    ref.read(medicineCatalogProvider);
  }

  Future<void> _pickImage() async {
    final messenger = ScaffoldMessenger.of(context);
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (!mounted) return;

    if (pickedFile == null) return;
    setState(() => _image = File(pickedFile.path));
    // Kết quả cũ không còn đúng với ảnh mới -> xoá để không hiển thị nhầm.
    ref.read(analysisControllerProvider.notifier).reset();
    messenger.hideCurrentSnackBar();
  }

  Future<void> _processImage() async {
    final l10n = AppLocalizations.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final scheme = Theme.of(context).colorScheme;
    final accent = NeuTheme.of(context).accent;
    final image = _image;
    if (image == null) return;

    final ok = await ref.read(analysisControllerProvider.notifier).analyze(
          image.path,
        );
    if (!mounted) return;

    final error = ref.read(analysisControllerProvider).error;
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          ok
              ? l10n.processSuccess
              : (error is AppException ? error.message : l10n.processFailed),
        ),
        backgroundColor: ok ? accent : scheme.error,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  /// Đọc đơn thuốc trên máy (không gửi đi đâu) rồi mở màn xác nhận.
  /// Text OCR chỉ nằm trong bộ nhớ; chỉ dòng thuốc người dùng xác nhận mới được lưu.
  void _openPrescriptionReview(String ocrText) {
    final items = parsePrescription(ocrText);
    if (items.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(AppLocalizations.of(context).rxNothingFound),
          behavior: SnackBarBehavior.floating,
        ),
      );
      return;
    }
    Navigator.of(context).push(
      MaterialPageRoute<int>(
        builder: (_) => PrescriptionReviewScreen(
          items: items,
          diagnosis: extractDiagnosis(ocrText),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final analysis = ref.watch(analysisControllerProvider);
    final medicines = analysis.valueOrNull?.medicines ?? const <Medicine>[];
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(title: Text(l10n.scanMedicine)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          children: [
            _buildImageCard(),
            const SizedBox(height: 20),
            _buildButtons(),
            if ((analysis.valueOrNull?.text ?? '').trim().isNotEmpty) ...[
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () =>
                    _openPrescriptionReview(analysis.valueOrNull!.text!),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size.fromHeight(48),
                  shape: const StadiumBorder(),
                ),
                icon: const Icon(Icons.alarm_add_outlined),
                label: Text(l10n.rxCreateFromScan),
              ),
            ],
            if (analysis.valueOrNull != null) ...[
              const SizedBox(height: 28),
              _buildSearchResultsList(medicines),
              const SizedBox(height: 12),
              // Bắt buộc: màn phân tích ảnh là màn ra kết quả y tế.
              const MedicalDisclaimer(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildImageCard() {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    // `AspectRatio`: ảnh co giãn theo bề rộng màn hình nên không bị méo.
    return NeuCard(
      padding: const EdgeInsets.all(12),
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: _image == null
            ? CustomPaint(
                painter: _DashedRRectPainter(
                  color: neu.textMuted.withOpacity(0.5),
                  radius: 14,
                ),
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.add_a_photo_outlined,
                            size: 36, color: neu.accent),
                        const SizedBox(height: 8),
                        Flexible(
                          child: Text(
                            l10n.noImagePicked,
                            textAlign: TextAlign.center,
                            style: TextStyle(color: neu.textMuted),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              )
            : ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: Image.file(
                  _image!,
                  fit: BoxFit.cover,
                  width: double.infinity,
                ),
              ),
      ),
    );
  }

  Widget _buildButtons() {
    final l10n = AppLocalizations.of(context);
    final isLoading = ref.watch(analysisControllerProvider).isLoading;
    final neu = NeuTheme.of(context);
    final scheme = Theme.of(context).colorScheme;
    const shape = StadiumBorder();
    const minSize = Size.fromHeight(52);
    return Row(
      children: [
        Expanded(
          child: FilledButton.tonalIcon(
            onPressed: _pickImage,
            style: FilledButton.styleFrom(
              shape: shape,
              minimumSize: minSize,
              backgroundColor: neu.accentSoft,
              foregroundColor: neu.textStrong,
            ),
            icon: const Icon(Icons.photo_library_outlined),
            label: Text(l10n.pickImage, textAlign: TextAlign.center),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: FilledButton.icon(
            onPressed: isLoading ? null : _processImage,
            style: FilledButton.styleFrom(
              shape: shape,
              minimumSize: minSize,
              backgroundColor: neu.accent,
              foregroundColor: scheme.onPrimary,
            ),
            icon: isLoading
                ? SizedBox(
                    height: 18,
                    width: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      color: scheme.onPrimary,
                    ),
                  )
                : const Icon(Icons.document_scanner_outlined),
            label: Text(l10n.processImage, textAlign: TextAlign.center),
          ),
        ),
      ],
    );
  }

  Widget _buildSearchResultsList(List<Medicine> medicines) {
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        NeuSectionTitle(title: l10n.searchResultsLabel),
        const SizedBox(height: 8),
        if (medicines.isEmpty)
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              l10n.noResults,
              style: theme.textTheme.bodyLarge?.copyWith(color: neu.textMuted),
            ),
          )
        else
          for (final medicine in medicines)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: NeuCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      medicine.name,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: neu.textStrong,
                      ),
                    ),
                    const SizedBox(height: 4),
                    _optionalLine(l10n.composition, medicine.composition),
                    _optionalLine(l10n.uses, medicine.uses),
                    _optionalLine(l10n.sideEffects, medicine.sideEffects),
                  ],
                ),
              ),
            ),
      ],
    );
  }

  /// Trường `null` thì bỏ hẳn dòng — trước đây code ghép chuỗi rồi `split(': ')`
  /// nên tên thuốc chứa `: ` sẽ hiển thị sai.
  Widget _optionalLine(String label, String? value) {
    if (value == null) return const SizedBox.shrink();
    final neu = NeuTheme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 4.0),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: '$label: ',
              style: TextStyle(color: neu.textMuted),
            ),
            TextSpan(text: value, style: TextStyle(color: neu.textStrong)),
          ],
        ),
      ),
    );
  }
}

/// Viền nét đứt bo góc cho vùng gợi ý "chưa chọn ảnh".
class _DashedRRectPainter extends CustomPainter {
  _DashedRRectPainter({required this.color, required this.radius});

  final Color color;
  final double radius;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    final path = Path()
      ..addRRect(
        RRect.fromRectAndRadius(
          Offset.zero & size,
          Radius.circular(radius),
        ),
      );
    for (final metric in path.computeMetrics()) {
      double distance = 0;
      while (distance < metric.length) {
        canvas.drawPath(metric.extractPath(distance, distance + 7), paint);
        distance += 12;
      }
    }
  }

  @override
  bool shouldRepaint(_DashedRRectPainter old) =>
      old.color != color || old.radius != radius;
}
