import 'app_localizations.dart';

/// The translations for Vietnamese (`vi`).
class AppLocalizationsVi extends AppLocalizations {
  AppLocalizationsVi([String locale = 'vi']) : super(locale);

  @override
  String get appTitle => 'MedAssist';

  @override
  String get loginTitle => 'Đăng nhập';

  @override
  String get loginSubtitle => 'Thông tin thuốc đáng tin cậy';

  @override
  String get usernameLabel => 'Tên đăng nhập';

  @override
  String get passwordLabel => 'Mật khẩu';

  @override
  String get emailLabel => 'Email';

  @override
  String get signIn => 'Đăng nhập';

  @override
  String get signUp => 'Đăng ký';

  @override
  String get signUpTitle => 'Đăng ký tài khoản';

  @override
  String get signUpLink => 'Đăng ký tài khoản mới';

  @override
  String get showPassword => 'Hiện mật khẩu';

  @override
  String get hidePassword => 'Ẩn mật khẩu';

  @override
  String get signInFailed => 'Đăng nhập thất bại!';

  @override
  String get signUpFailed => 'Đăng ký thất bại!';

  @override
  String get processFailed => 'Không phân tích được ảnh.';

  @override
  String get createAccount => 'Tạo tài khoản';

  @override
  String get signOut => 'Đăng xuất';

  @override
  String get signOutConfirmTitle => 'Đăng xuất?';

  @override
  String get signOutConfirmBody => 'Bạn sẽ phải đăng nhập lại để tiếp tục.';

  @override
  String get cancel => 'Huỷ';

  @override
  String get confirm => 'Xác nhận';

  @override
  String get delete => 'Xoá';

  @override
  String get processImageTitle => 'Quét thuốc';

  @override
  String get pickImage => 'Chọn ảnh';

  @override
  String get processImage => 'Phân tích';

  @override
  String get noImagePicked => 'Chưa chọn ảnh';

  @override
  String get history => 'Lịch sử';

  @override
  String get search => 'Tìm kiếm';

  @override
  String get resultsTitle => 'Kết quả:';

  @override
  String get noResults => 'Không có kết quả.';

  @override
  String get processSuccess => 'Xử lý thành công!';

  @override
  String get searchTitle => 'Tìm kiếm thuốc';

  @override
  String get searchHint => 'Nhập tên thuốc';

  @override
  String get searchEmptyQuery => 'Vui lòng nhập tên thuốc!';

  @override
  String get historyTitle => 'Lịch sử xử lý';

  @override
  String get historyEmpty => 'Chưa có lịch sử';

  @override
  String get historyLoadFailed => 'Không tải được lịch sử.';

  @override
  String get searchResultsLabel => 'Kết quả tìm kiếm:';

  @override
  String get historyFile => 'Tệp:';

  @override
  String get deleteRecordTitle => 'Xoá bản ghi này?';

  @override
  String get deleteRecordBody => 'Bản ghi sẽ bị xoá khỏi lịch sử của bạn.';

  @override
  String get deleted => 'Đã xoá';

  @override
  String get deleteFailed => 'Không xoá được bản ghi';

  @override
  String get retry => 'Thử lại';

  @override
  String get offline => 'Dữ liệu ngoại tuyến';

  @override
  String offlineUpdated(String time) {
    return 'Dữ liệu ngoại tuyến · cập nhật $time';
  }

  @override
  String get noInformation => 'Không có thông tin';

  @override
  String get medicineName => 'Tên thuốc';

  @override
  String get composition => 'Thành Phần';

  @override
  String get uses => 'Công Dụng';

  @override
  String get sideEffects => 'Tác Dụng Phụ';

  @override
  String get manufacturer => 'Nhà Sản Xuất';

  @override
  String get reviewExcellent => 'Đánh Giá Xuất Sắc';

  @override
  String get reviewAverage => 'Đánh Giá Trung Bình';

  @override
  String get reviewPoor => 'Đánh Giá Kém';

  @override
  String get validationUsernameRequired => 'Vui lòng nhập tên đăng nhập.';

  @override
  String get validationPasswordRequired => 'Vui lòng nhập mật khẩu.';

  @override
  String validationPasswordTooShort(int length) {
    return 'Mật khẩu phải có ít nhất $length ký tự.';
  }

  @override
  String get validationEmailRequired => 'Vui lòng nhập email.';

  @override
  String get validationEmailInvalid => 'Email không hợp lệ.';

  @override
  String get medicalDisclaimer =>
      'Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán hoặc đơn thuốc của bác sĩ.';
}
