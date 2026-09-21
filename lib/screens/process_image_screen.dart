import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

import 'HistoryScreen.dart';
import 'MedicineSearchScreen.dart';

class ImageProcessingScreen extends StatefulWidget {
  const ImageProcessingScreen({super.key});

  @override
  _ImageProcessingScreenState createState() => _ImageProcessingScreenState();
}

class _ImageProcessingScreenState extends State<ImageProcessingScreen> {
  File? _image;
  bool _isLoading = false;
  List<String>? _searchResults;
  String _searchQuery = "";

  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _searchResults = null;
      });
    }
  }

  Future<void> _processImage() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('http://10.0.2.2:5000/process-image'),
      );
      request.files.add(await http.MultipartFile.fromPath('image', _image!.path));

      final response = await request.send();

      if (response.statusCode == 200) {
        final responseData = await http.Response.fromStream(response);
        final data = jsonDecode(responseData.body);

        setState(() {
          if (data['medicine_info'] is List) {
            _searchResults = (data['medicine_info'] as List).map((item) {
              return '''
                Tên: ${item['Medicine_Name']}
                Thành phần: ${item['Composition']}
                Công dụng: ${item['Uses']}
                Tác dụng phụ: ${item['Side_effects']}
              ''';
            }).toList();
          } else {
            _searchResults = [];
          }
        });

        _showSnackBar('Xử lý thành công!', Colors.green);
      } else {
        _showSnackBar('Lỗi: ${response.statusCode}', Colors.red);
      }
    } catch (e) {
      _showSnackBar('Lỗi: $e', Colors.red);
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showSnackBar(String message, Color color) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 2),
      ),
    );
  }

  List<String>? _getFilteredResults() {
    if (_searchQuery.isEmpty) {
      return _searchResults;
    } else {
      return _searchResults
          ?.where((result) => result.toLowerCase().contains(_searchQuery.toLowerCase()))
          .toList();
    }
  }

  @override
  Widget build(BuildContext context) {
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
            tooltip: 'Lịch sử',
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => HistoryScreen()),
              );
            },
          ),
          IconButton(
            icon: const Icon(Icons.search, size: 30),
            color: Colors.white,
            tooltip: 'Tìm kiếm',
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => MedicineSearchPage()),
              );
            },
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              _buildImageCard(),
              const SizedBox(height: 20),
              _buildButtons(),
              const SizedBox(height: 20),
              if (_searchResults != null) _buildSearchResultsList(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImageCard() {
    return Card(
      elevation: 5,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: SizedBox(
        height: 200,
        width: double.infinity,
        child: _image == null
            ? const Center(child: Text('Chưa chọn ảnh', style: TextStyle(color: Colors.black54)))
            : ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Image.file(_image!, fit: BoxFit.cover),
              ),
      ),
    );
  }

  Widget _buildButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        ElevatedButton(
          onPressed: _pickImage,
          child: const Text('Chọn Ảnh'),
        ),
        ElevatedButton(
          onPressed: _isLoading ? null : _processImage,
          child: _isLoading ? const CircularProgressIndicator() : const Text('Xử lý Ảnh'),
        ),
      ],
    );
  }

  Widget _buildSearchResultsList() {
    final filteredResults = _getFilteredResults();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Kết quả tìm kiếm:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        if (filteredResults != null && filteredResults.isNotEmpty)
          SizedBox(
            height: 400,
            child: ListView.builder(
              itemCount: filteredResults.length,
              itemBuilder: (context, index) {
                final details = filteredResults[index].split('\n');
                return Card(
                  margin: const EdgeInsets.symmetric(vertical: 8.0),
                  elevation: 5,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (details.length >= 4) ...[
                          Text(
                            'Tên: ${details[0].split(': ').last}',
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                          ),
                          Text(
                            'Thành phần: ${details[1].split(': ').last}',
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                          ),
                          Text(
                            'Công dụng: ${details[2].split(': ').last}',
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                          ),
                          Text(
                            'Tác dụng phụ: ${details[3].split(': ').last}',
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                          ),
                        ],
                      ],
                    ),
                  ),
                );
              },
            ),
          )
        else
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text('Không có kết quả.', style: TextStyle(fontSize: 16, color: Colors.grey)),
          ),
      ],
    );
  }

  Widget _buildHistoryButton() {
    return ElevatedButton(
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => HistoryScreen()),
        );
      },
      child: const Text('Xem Lịch sử'),
    );
  }
}
