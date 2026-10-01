/// Cấu hình runtime của app — **chỗ duy nhất** trong `lib/` chứa địa chỉ server.
///
/// Mọi giá trị lấy từ `--dart-define` (compile-time, không cần dependency):
/// ```powershell
/// flutter run   --dart-define=API_BASE_URL=http://10.0.2.2:5000
/// flutter build --dart-define=API_BASE_URL=https://api.medassist.vn
/// ```
///
/// Vì sao không hardcode: `10.0.2.2` chỉ là "máy host nhìn từ emulator";
/// điện thoại thật phải dùng IP LAN, production phải là HTTPS.
/// Xem thêm `docs/TROUBLESHOOTING.md` mục 5.
class AppConfig {
  const AppConfig._();

  /// Gốc của API, **không có dấu `/` ở cuối**.
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:5000',
  );

  /// Tiền tố đường dẫn, dùng khi backend chuyển sang `/v1/...` (Day 15+).
  /// Mặc định rỗng để khớp Flask hiện tại (`/login`, `/history`, ...).
  static const String apiPrefix = String.fromEnvironment('API_PREFIX');

  static const int connectTimeoutSeconds = int.fromEnvironment(
    'API_CONNECT_TIMEOUT_SECONDS',
    defaultValue: 15,
  );

  static const int receiveTimeoutSeconds = int.fromEnvironment(
    'API_RECEIVE_TIMEOUT_SECONDS',
    defaultValue: 60,
  );

  /// Thời gian chờ TCP/TLS bắt đầu được kết nối.
  static const Duration connectTimeout =
      Duration(seconds: connectTimeoutSeconds);

  /// Thời gian chờ server trả về dữ liệu (upload ảnh + OCR có thể chậm).
  static const Duration receiveTimeout =
      Duration(seconds: receiveTimeoutSeconds);

  /// Nhãn build, in ra log để biết app đang trỏ vào đâu.
  static const String flavor = String.fromEnvironment(
    'APP_FLAVOR',
    defaultValue: 'dev',
  );

  /// Server đang dùng HTTPS? (release bắt buộc `true` — xem
  /// `android/app/src/debug/res/xml/network_security_config.xml`).
  static bool get usesHttps => Uri.parse(baseUrl).scheme == 'https';

  /// Ghép `baseUrl` + [apiPrefix] + [path] thành `Uri` tuyệt đối, có encode
  /// `queryParameters` (sửa bug `'?name=$ten'` không escape của code cũ).
  ///
  /// [path] bắt đầu bằng `/`, vd `uri('/search-medicine', queryParameters: {'name': 'Panadol'})`.
  /// Nếu [queryParameters] khác rỗng thì nó **thay thế** query string có sẵn
  /// trong [path] — muốn dùng query sẵn thì truyền `null`.
  static Uri uri(String path, {Map<String, Object?>? queryParameters}) {
    final base = Uri.parse(_join(baseUrl, apiPrefix, path));
    if (queryParameters == null || queryParameters.isEmpty) return base;

    final cleaned = <String, dynamic>{
      for (final e in queryParameters.entries)
        if (e.value != null) e.key: '${e.value}',
    };
    return cleaned.isEmpty ? base : base.replace(queryParameters: cleaned);
  }

  /// Nối đường dẫn mà không sinh `//` thừa.
  ///
  /// Bỏ hẳn các đoạn rỗng: `apiPrefix` mặc định là `''` nếu cứ "ép dấu `/`"
  /// thì `baseUrl + '/' + '/history'` sinh ra `//history` — server sẽ 404.
  static String _join(String base, String prefix, String path) {
    String trimTrailing(String s) =>
        s.endsWith('/') ? s.substring(0, s.length - 1) : s;
    String ensureLeading(String s) {
      if (s.isEmpty) return '';
      return s.startsWith('/') ? s : '/$s';
    }

    return <String>[
      trimTrailing(base),
      ensureLeading(prefix),
      ensureLeading(path),
    ].where((s) => s.isNotEmpty).join();
  }

  /// Chuỗi ngắn gọn để log khi khởi động: `dev http://10.0.2.2:5000 (http)`.
  static String describe() =>
      '$flavor $baseUrl${apiPrefix.isEmpty ? '' : ' +$apiPrefix'} '
      '(${usesHttps ? 'https' : 'http'} · timeout '
      '${connectTimeout.inSeconds}s/${receiveTimeout.inSeconds}s)';

  /// Kiểm tra cấu hình sớm: gọi từ `main()` khi debug để bắt lỗi
  /// `--dart-define` sai chính tả (vd thiếu `http://`, còn dấu `/`).
  static List<String> validate() => [
        if (!baseUrl.startsWith('http://') && !baseUrl.startsWith('https://'))
          'API_BASE_URL phải bắt đầu bằng http:// hoặc https:// (hiện: "$baseUrl")',
        if (baseUrl.endsWith('/')) 'API_BASE_URL không được kết thúc bằng "/"',
        if (Uri.parse(baseUrl).host.isEmpty) 'API_BASE_URL thiếu host',
        if (apiPrefix.isNotEmpty && !apiPrefix.startsWith('/'))
          'API_PREFIX phải bắt đầu bằng "/" (hiện: "$apiPrefix")',
        if (receiveTimeout <= connectTimeout)
          'API_RECEIVE_TIMEOUT_SECONDS phải lớn hơn thời gian kết nối '
              '(${receiveTimeout.inSeconds}s <= ${connectTimeout.inSeconds}s)',
      ];
}
