import 'package:image_processing_app/data/models/user.dart';

/// Cổng trừu tượng cho phiên đăng nhập.
///
/// **Không có `password` trong bất kỳ kiểu dữ liệu nào của domain** — mật khẩu
/// chỉ tồn tại ở tham số hàm rồi bị adapter băm và xoá ngay (xem `User`).
abstract interface class AuthRepository {
  /// Đăng nhập → trả phiên (user + token nếu server có).
  ///
  /// Ném `AppExceptionKind.unauthorized` khi sai tài khoản/mật khẩu, hoặc
  /// `validation` kể cả chưa nhập gì (Day 11 chuyển validate sang tầng UI).
  Future<AuthSession> login(String username, String password);

  /// Đăng ký tài khoản mới; trả phiên nếu server đăng nhập luôn sau đăng ký.
  ///
  /// Ném `AppExceptionKind.conflict` khi tên đăng nhập đã tồn tại.
  Future<AuthSession> register(
    String username,
    String password,
    String email,
  );

  /// Xoá token cục bộ (và gọi `/logout` nếu server có). Phải **an toàn khi
  /// gọi lại nhiều lần** — dùng khi app tắt hoặc gặp 401 (Day 11).
  Future<void> logout();

  /// Làm mới phiên bằng refresh token đã lưu. Trả `true` khi có token mới (đã
  /// lưu); `false` khi không làm mới được (không có refresh token, hết hạn, bị
  /// thu hồi) — khi đó người dùng phải đăng nhập lại. Không bao giờ ném.
  Future<bool> refreshSession();

  /// Xoá tài khoản và toàn bộ dữ liệu trên server (quyền được quên). Cần mật
  /// khẩu để xác nhận. Thành công thì token cục bộ cũng bị xoá.
  ///
  /// Ném `AppExceptionKind.unauthorized` khi sai mật khẩu.
  Future<void> deleteAccount(String password);

  /// Người đang đăng nhập (để hiện tên). `null` nếu backend không có endpoint này hoặc gọi
  /// lỗi — **không bao giờ ném**. Token hết hạn thì tầng mạng tự làm mới / đăng xuất.
  Future<User?> currentUser();

  /// Token hiện tại, `null` nếu chưa đăng nhập. Đây là nguồn cho
  /// `ApiClient.tokenProvider` (Day 11), nên phải đọc **đồng bộ về mặt logic**
  /// và không được ném lỗi.
  Future<String?> currentToken();
}
