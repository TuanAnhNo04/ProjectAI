/// Hạ tầng test dùng chung cho các adapter (Day 8).
///
/// `flutter test` chặn socket thật, nên thay vì gọi Flask ta cài một
/// `HttpClientAdapter` giả để **tự quyết định từng phản hồi** và bắt lại
/// request đã gửi (URL/query/body/multipart). Đây đúng là chỗ Dio sẽ đi tới
/// khi nối server thật, nên test kiểm được cả hợp đồng HTTP.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:image_processing_app/core/network/api_client.dart';

/// Adapter trả về [responder] cho mọi request và lưu lại request đã gửi.
class FakeHttpAdapter implements HttpClientAdapter {
  FakeHttpAdapter(this.responder);

  final ResponseBody Function(RequestOptions options) responder;

  /// Bắt lại mọi request để kiểm tra method/URL/header/body.
  final List<RequestOptions> sent = [];

  RequestOptions get last => sent.last;

  @override
  Future<ResponseBody> fetch(
    RequestOptions options,
    Stream<Uint8List>? requestStream,
    Future<void>? cancelFuture,
  ) async {
    sent.add(options);
    return responder(options);
  }

  @override
  void close({bool force = false}) {}
}

/// Một [ResponseBody] chứa JSON [body] với [status] (mặc định 200).
ResponseBody jsonBody(Object? body, [int status = 200]) =>
    ResponseBody.fromString(
      jsonEncode(body),
      status,
      headers: {
        Headers.contentTypeHeader: [Headers.jsonContentType],
      },
    );

/// Kết quả dựng client giả: dùng adapter để soi request, client để gọi.
typedef FakeClient = ({ApiClient client, FakeHttpAdapter adapter});

/// Tạo [ApiClient] trỏ vào [FakeHttpAdapter].
FakeClient fakeClient(
  ResponseBody Function(RequestOptions options) responder, {
  TokenProvider? tokenProvider,
}) {
  final adapter = FakeHttpAdapter(responder);
  final client = ApiClient(
    dio: Dio()..httpClientAdapter = adapter,
    tokenProvider: tokenProvider,
    logTraffic: false,
  );
  return (client: client, adapter: adapter);
}
