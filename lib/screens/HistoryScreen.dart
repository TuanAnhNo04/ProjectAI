import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({Key? key}) : super(key: key);

  @override
  _HistoryScreenState createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<dynamic>? _historyData;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchHistory();
  }

  Future<void> _fetchHistory() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final response = await http.get(Uri.parse('http://10.0.2.2:5000/history'));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _historyData = data['history'] ?? [];
        });
      } else {
        _showSnackBar('Error: ${response.statusCode}', Colors.red);
      }
    } catch (e) {
      _showSnackBar('Connection Error: $e', Colors.red);
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

  Future<void> _deleteHistoryItem(dynamic item) async {
    final int? id = item['id'];
    if (id == null) {
      _showSnackBar('Invalid ID', Colors.red);
      return;
    }

    try {
      final response = await http.delete(Uri.parse('http://10.0.2.2:5000/history/$id'));

      if (response.statusCode == 200) {
        setState(() {
          _historyData!.removeWhere((historyItem) => historyItem['id'] == id);
        });
        _showSnackBar('Deleted successfully', Colors.green);
      } else {
        _showSnackBar('Error: ${response.statusCode}', Colors.red);
      }
    } catch (e) {
      _showSnackBar('Connection Error: $e', Colors.red);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60),
        child: AppBar(
          flexibleSpace: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.deepPurple, Colors.purpleAccent],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
          ),
          title: const Text(
            'Processing History',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              shadows: [
                Shadow(
                  blurRadius: 5,
                  color: Colors.black26,
                  offset: Offset(2, 2),
                ),
              ],
            ),
          ),
          centerTitle: true,
          elevation: 5,
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _historyData == null || _historyData!.isEmpty
              ? const Center(
                  child: Text(
                    'No processing history',
                    style: TextStyle(fontSize: 18, color: Colors.grey),
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16.0),
                  itemCount: _historyData!.length,
                  itemBuilder: (context, index) {
                    final item = _historyData![index];
                    return Card(
                      margin: const EdgeInsets.symmetric(vertical: 8),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15),
                      ),
                      elevation: 3,
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'File: ${item['filepath']}',
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.black87,
                                  ),
                                ),
                                IconButton(
                                  icon: const Icon(Icons.delete, color: Colors.red),
                                  onPressed: () => _deleteHistoryItem(item),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            const SizedBox(height: 4),
                            Text(
                              'Search Results:',
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.black87,
                              ),
                            ),
                            // Display the search_results in a readable format
                            ...(item['search_results'] ?? []).map<Widget>((searchResult) {
                              return Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const SizedBox(height: 4),
                                  Text(
                                    'Medicine Name: ${searchResult['Medicine_Name']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Composition: ${searchResult['Composition']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Uses: ${searchResult['Uses']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Side Effects: ${searchResult['Side_effects']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Manufacturer: ${searchResult['Manufacturer']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Excellent Review %: ${searchResult['Excellent Review %']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Average Review %: ${searchResult['Average Review %']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  Text(
                                    'Poor Review %: ${searchResult['Poor Review %']}',
                                    style: const TextStyle(color: Colors.black54),
                                  ),
                                  const Divider(),
                                ],
                              );
                            }).toList(),
                          ],
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}
