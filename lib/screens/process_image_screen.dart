import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'history_screen.dart';
import 'medicine_search_screen.dart';

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
  void dispose() {
    super.dispose();
  }

  /// Xoá token cục bộ và quay về màn Login.
  ///
  /// Hỏi lại trước vì sau khi xoá token, người dùng phải đăng nhập lại.
  Future<void> _logout(BuildContext context, WidgetRef ref) async {
    final l10n = AppLocalizations.of(context);
    final navigator = Navigator.of(context);
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(l10n.signOutConfirmTitle),
        content: Text(l10n.signOutConfirmBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: Text(l10n.cancel),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: Text(l10n.signOut),
          ),
        ],
      ),
    );
    if (confirmed != true || !context.mounted) return;

    await ref.read(authControllerProvider.notifier).logout();
    if (!context.mounted) return;

    // `popUntil(isFirst)` đưa về Login (route gốc của MaterialApp) và xoá
    // sạch các màn đã mở — nếu không, bấm "back" sẽ vào lại màn cần token.
    navigator.popUntil((route) => route.isFirst);
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
              ? 'Xử lý thành công!'
              : (error is AppException ? error.message : l10n.processFailed),
        ),
        backgroundColor: ok ? Colors.green : Colors.red,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final analysis = ref.watch(analysisControllerProvider);
    final medicines = analysis.valueOrNull?.medicines ?? const <Medicine>[];
    final l10n = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Xử lý Ảnh',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 22,
            color: Colors.white,
            letterSpacing: 1.2,
          ),
        ),
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.deepPurple, Colors.purpleAccent],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
        elevation: 10,
        actions: [
          IconButton(
            icon: const Icon(Icons.history, size: 30),
            color: Colors.white,
            tooltip: l10n.history,
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const HistoryScreen()),
              );
            },
          ),
          IconButton(
            icon: const Icon(Icons.search, size: 30),
            color: Colors.white,
            tooltip: l10n.search,
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (context) => const MedicineSearchPage()),
              );
            },
          ),
          // Nút Đăng xuất (Day 11) — chỉ hiện khi thật sự đã đăng nhập.
          if (ref.watch(authControllerProvider).valueOrNull?.isAuthenticated ??
              false)
            IconButton(
              icon: const Icon(Icons.logout, size: 30),
              color: Colors.white,
              tooltip: l10n.signOut,
              onPressed: () => _logout(context, ref),
            ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Column(
              children: [
                _buildImageCard(),
                const SizedBox(height: 20),
                _buildButtons(),
                const SizedBox(height: 20),
                if (medicines.isNotEmpty) ...[
                  _buildSearchResultsList(medicines),
                  const SizedBox(height: 12),
                  // Bắt buộc: màn phân tích ảnh là màn ra kết quả y tế.
                  const MedicalDisclaimer(),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImageCard() {
    final l10n = AppLocalizations.of(context);
    return Card(
      elevation: 5,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      // `AspectRatio` thay cho `SizedBox(height: 200)`: ảnh co giãn theo bề rộng
      // màn hình nên không bị méo trên máy nhỏ / cỡ chữ hệ thống lớn.
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: _image == null
            ? Center(
                child: Text(l10n.noImagePicked,
                    style: const TextStyle(color: Colors.black54)))
            : ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Image.file(_image!, fit: BoxFit.cover),
              ),
      ),
    );
  }

  Widget _buildButtons() {
    final l10n = AppLocalizations.of(context);
    final isLoading = ref.watch(analysisControllerProvider).isLoading;
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        ElevatedButton(
          onPressed: _pickImage,
          child: Text(l10n.pickImage),
        ),
        ElevatedButton(
          onPressed: isLoading ? null : _processImage,
          child: isLoading
              ? const SizedBox(
                  height: 20,
                  width: 20,
                  child: CircularProgressIndicator(strokeWidth: 2.5),
                )
              : Text(l10n.processImage),
        ),
      ],
    );
  }

  Widget _buildSearchResultsList(List<Medicine> medicines) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Kết quả tìm kiếm:',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 10),
        if (medicines.isEmpty)
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'Không có kết quả.',
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
          )
        else
          // `Flexible` thay cho `SizedBox(height: 400)` cứng: không còn tràn
          // khi cỡ chữ hệ thống lớn hay màn hình thấp.
          Flexible(
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: medicines.length,
              itemBuilder: (context, index) => Card(
                margin: const EdgeInsets.symmetric(vertical: 8.0),
                elevation: 5,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Tên: ${medicines[index].name}',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 4),
                      _optionalLine('Thành phần', medicines[index].composition),
                      _optionalLine('Công dụng', medicines[index].uses),
                      _optionalLine(
                        'Tác dụng phụ',
                        medicines[index].sideEffects,
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

  /// Trường `null` thì bỏ hẳn dòng — trước đây code ghép chuỗi rồi `split(': ')`
  /// nên tên thuốc chứa `: ` sẽ hiển thị sai.
  Widget _optionalLine(String label, String? value) => value == null
      ? const SizedBox.shrink()
      : Padding(
          padding: const EdgeInsets.only(top: 4.0),
          child: Text('$label: $value'),
        );
}
