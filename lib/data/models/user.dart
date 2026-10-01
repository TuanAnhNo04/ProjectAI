import 'package:image_processing_app/data/models/json_utils.dart';

/// Người dùng — **chỉ chứa danh tính, tuyệt đối không giữ mật khẩu**.
///
/// Mật khẩu chỉ tồn tại dưới dạng tham số của [toCredentialsJson] rồi bị bỏ
/// ngay; không có field nào trong class này mang tên `password`. Đó là cách
/// chống rò rỉ *mặc định* thay vì chỉ dựa vào kỷ luật khi viết code.
class User {
  const User({
    required this.username,
    this.id,
    this.email,
    this.displayName,
  });

  factory User.fromJson(Map<String, dynamic> json) => User(
        id: asIntOrNull(json[keyId]),
        username: asString(json[keyUsername]),
        email: asStringOrNull(json[keyEmail]),
        displayName: asStringOrNull(json[keyDisplayName]),
      );

  static const String keyId = 'id';
  static const String keyUsername = 'username';
  static const String keyEmail = 'email';
  static const String keyDisplayName = 'display_name';

  final int? id;

  /// Tên đăng nhập. Server cũ chưa bắt buộc nên có thể rỗng — UI chặn trước.
  final String username;

  final String? email;

  /// Tên hiển thị; FastAPI (Day 15) mới có, hiện để trống.
  final String? displayName;

  /// Email hợp lệ để gửi OTP/disclaimer? (Day 11 dùng cho validate form.)
  bool get hasValidEmail => email != null && _emailPattern.hasMatch(email!);

  User copyWith({
    Object? id = notProvided,
    String? username,
    Object? email = notProvided,
    Object? displayName = notProvided,
  }) =>
      User(
        id: pickChanged(id, this.id),
        username: username ?? this.username,
        email: pickChanged(email, this.email),
        displayName: pickChanged(displayName, this.displayName),
      );

  /// **Không** phát lại `password` — class này không chứa mật khẩu.
  Map<String, dynamic> toJson() => <String, dynamic>{
        keyId: id,
        keyUsername: username,
        keyEmail: email,
        keyDisplayName: displayName,
      };

  /// Body gửi lên `/login` và `/register`. Mật khẩu là **tham số bắt buộc**
  /// nên không bao giờ nằm lại trong object `User` sau khi gửi xong.
  Map<String, dynamic> toCredentialsJson({required String password}) =>
      <String, dynamic>{
        keyUsername: username,
        keyPassword: password,
        if (email != null) keyEmail: email,
      };

  static const String keyPassword = 'password';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is User &&
          other.id == id &&
          other.username == username &&
          other.email == email &&
          other.displayName == displayName;

  @override
  int get hashCode => Object.hash(id, username, email, displayName);

  @override
  String toString() => 'User($username)';
}

/// Phiên đăng nhập: người dùng + token.
///
/// Tách khỏi [User] để token có **vòng đời riêng** (xoá khi 401 ở Day 11)
/// mà không đụng vào dữ liệu danh tính.
class AuthSession {
  const AuthSession({required this.user, this.token});

  /// Đọc phản hồi `/login`. Chịu được cả 3 kiểu server:
  /// `{"user": {...}, "token": "..."}`, `{"token": "..."}`, và
  /// `{"message": "..."}` (bản cũ chưa trả token).
  factory AuthSession.fromJson(Map<String, dynamic> json) {
    final rawUser = json[keyUser];
    final user = rawUser is Map
        ? User.fromJson(asJsonMap(rawUser, keyUser))
        : User(username: asString(json[User.keyUsername]));

    return AuthSession(
      user: user,
      token: asStringOrNull(json[keyToken] ?? json[keyAccessToken]),
    );
  }

  factory AuthSession.parse(String body) =>
      AuthSession.fromJson(decodeJsonObject(body));

  static const String keyUser = 'user';
  static const String keyToken = 'token';
  static const String keyAccessToken = 'access_token';

  final User user;

  /// `null` nghĩa là server bản cũ **không xác thực bằng token** — app vẫn
  /// chạy được như màn Login cũ, chỉ là Day 11 chưa có việc gắn header.
  final String? token;

  bool get isAuthenticated => token != null && token!.isNotEmpty;

  @override
  String toString() => 'AuthSession(${user.username}, '
      '${isAuthenticated ? 'có token' : 'không token'})';
}

/// Cố ý giản dị: chỉ chặn rõ ràng là sai, không chặn nhầm email hợp lệ.
final RegExp _emailPattern =
    RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$');