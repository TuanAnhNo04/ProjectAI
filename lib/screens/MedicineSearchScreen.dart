import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class MedicineSearchPage extends StatefulWidget {
  @override
  _MedicineSearchPageState createState() => _MedicineSearchPageState();
}

class _MedicineSearchPageState extends State<MedicineSearchPage> {
  final TextEditingController _searchController = TextEditingController();
  List<dynamic> _results = [];
  String _errorMessage = '';

  Future<void> _searchMedicine() async {
    final String medicineName = _searchController.text.trim(); // Loại bỏ khoảng trắng

    if (medicineName.isEmpty) {
      setState(() {
        _errorMessage = "Vui lòng nhập tên thuốc!";
        _results = [];
      });
      return;
    }

    setState(() {
      _errorMessage = '';
    });

    // Tạo URL với tham số name
    final String url = 'http://10.0.2.2:5000/search-medicine?name=$medicineName';
    print("Searching for medicine: $url"); // In ra URL để kiểm tra

    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      setState(() {
        _results = json.decode(response.body)['results'];
      });
    } else {
      // In ra thông báo lỗi từ server
      print("Error response: ${response.body}");
      setState(() {
        _errorMessage = "Không tìm thấy thông tin thuốc!";
        _results = [];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Tìm kiếm thuốc'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _searchController,
              decoration: InputDecoration(
                labelText: 'Nhập tên thuốc',
                prefixIcon: Icon(Icons.search),
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white,
              ),
              onTap: () {
                // Mở bàn phím khi nhấn vào ô nhập liệu
                FocusScope.of(context).requestFocus(FocusNode());
                FocusScope.of(context).requestFocus(FocusNode());
              },
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: _searchMedicine,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                padding: EdgeInsets.symmetric(vertical: 20, horizontal: 40),
                textStyle: TextStyle(fontSize: 18),
              ),
              child: Text('Tìm kiếm'),
            ),
            SizedBox(height: 20),
            if (_errorMessage.isNotEmpty)
              Text(
                _errorMessage,
                style: TextStyle(color: Colors.red),
              ),
            Expanded(
              child: ListView.builder(
                itemCount: _results.length,
                itemBuilder: (context, index) {
                  return Card(
                    elevation: 4,
                    margin: EdgeInsets.symmetric(vertical: 8),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "Tên Thuốc: ${_results[index]['Medicine_Name']}",
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 18),
                          ),
                          SizedBox(height: 8),
                          Text("Thành Phần: ${_results[index]['Composition'] ?? 'Không có thông tin'}"),
                          Text("Công Dụng: ${_results[index]['Uses'] ?? 'Không có thông tin'}"),
                          Text("Tác Dụng Phụ: ${_results[index]['Side_effects'] ?? 'Không có thông tin'}"),
                          Text("Nhà Sản Xuất: ${_results[index]['Manufacturer'] ?? 'Không có thông tin'}"),
                          Text("Đánh Giá Xuất Sắc: ${_results[index]['Excellent Review %'] ?? '0%'}"),
                          Text("Đánh Giá Trung Bình: ${_results[index]['Average Review %'] ?? '0%'}"),
                          Text("Đánh Giá Kém: ${_results[index]['Poor Review %'] ?? '0%'}"),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
