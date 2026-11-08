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
  String get newsTitle => 'Tin tức thuốc & y tế';

  @override
  String get newsEmpty => 'Chưa có tin tức.';

  @override
  String get newsLoadFailed => 'Không tải được tin tức.';

  @override
  String get newsSampleBanner => 'Đang hiện nội dung mẫu — chưa bật nguồn tin thật.';

  @override
  String get newsOpenSource => 'Xem nguồn';

  @override
  String get newsNoSourcesConfigured => 'Chủ dự án chưa bật nguồn tin. Xem docs/NEWS.md.';

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
  String get medicalDisclaimer => 'Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán hoặc đơn thuốc của bác sĩ.';

  @override
  String get chatTitle => 'Hỏi đáp thuốc';

  @override
  String get chatHint => 'Nhập câu hỏi về thuốc…';

  @override
  String get chatSend => 'Gửi';

  @override
  String get chatStop => 'Dừng';

  @override
  String get chatEmpty => 'Hỏi bất kỳ thông tin nào về thuốc — thành phần, công dụng, tác dụng phụ.';

  @override
  String get chatClear => 'Xoá hội thoại';

  @override
  String get chatClearTitle => 'Xoá hội thoại?';

  @override
  String get chatClearBody => 'Toàn bộ tin nhắn sẽ bị xoá.';

  @override
  String chatToolRunning(String toolName) {
    return 'Đang tra cứu $toolName…';
  }

  @override
  String get healthProfileTitle => 'Hồ sơ sức khoẻ';

  @override
  String get healthConsentTitle => 'Đồng ý lưu hồ sơ';

  @override
  String get healthConsentBody => 'Hồ sơ sức khoẻ (dị ứng, bệnh nền, thuốc đang dùng) được lưu trên thiết bị của bạn. Hồ sơ không được tải lên máy chủ; chỉ danh sách thuốc đang dùng được gửi kèm, tạm thời, khi bạn hỏi trợ lý để cảnh báo tương tác. Bạn có thể xoá bất cứ lúc nào.';

  @override
  String get healthConsentAccept => 'Đồng ý';

  @override
  String get healthConsentGiven => 'Đã đồng ý lưu hồ sơ';

  @override
  String get healthAllergies => 'Dị ứng';

  @override
  String get healthAllergiesHint => 'VD: Penicillin, Aspirin';

  @override
  String get healthConditions => 'Bệnh nền';

  @override
  String get healthConditionsHint => 'VD: Tiểu đường, Cao huyết áp';

  @override
  String get healthCurrentMeds => 'Thuốc đang dùng';

  @override
  String get healthCurrentMedsHint => 'VD: Metformin, Amlodipine';

  @override
  String get healthSave => 'Lưu hồ sơ';

  @override
  String get healthSaved => 'Đã lưu hồ sơ';

  @override
  String get healthDelete => 'Xoá hồ sơ';

  @override
  String get healthDeleteTitle => 'Xoá hồ sơ sức khoẻ?';

  @override
  String get healthDeleteBody => 'Toàn bộ dữ liệu dị ứng, bệnh nền, thuốc đang dùng và lịch nhắc sẽ bị xoá.';

  @override
  String get reminderTitle => 'Nhắc uống thuốc';

  @override
  String get reminderEmpty => 'Chưa có lịch nhắc nào';

  @override
  String get reminderAdd => 'Thêm nhắc thuốc';

  @override
  String get reminderMedicineName => 'Tên thuốc';

  @override
  String get reminderDosage => 'Liều dùng';

  @override
  String get reminderDosageHint => 'VD: 1 viên, 5ml';

  @override
  String get consentTitle => 'Trước khi bắt đầu';

  @override
  String get consentIntro => 'MedAssist giúp tra cứu thông tin thuốc. Hãy đọc nhanh cách dữ liệu của bạn được xử lý:';

  @override
  String get consentPointLocal => 'Hồ sơ sức khoẻ, lịch nhắc và nhật ký uống thuốc chỉ lưu trên máy của bạn.';

  @override
  String get consentPointServer => 'Từ khoá tìm kiếm, câu hỏi cho trợ lý và ảnh bạn chọn để nhận diện (nếu dùng chế độ máy chủ) được gửi tới máy chủ của MedAssist. Không chia sẻ với bên thứ ba.';

  @override
  String get consentPointMedical => 'Thông tin chỉ mang tính tham khảo, không thay thế bác sĩ hoặc dược sĩ.';

  @override
  String get consentPointDelete => 'Bạn có thể xoá dữ liệu và tài khoản bất cứ lúc nào ở mục Quyền riêng tư.';

  @override
  String get consentAccept => 'Đồng ý và tiếp tục';

  @override
  String get consentDecline => 'Không đồng ý';

  @override
  String get privacyTitle => 'Quyền riêng tư';

  @override
  String get privacySummary => 'Hồ sơ sức khoẻ và lịch nhắc chỉ nằm trên máy. Câu hỏi và từ khoá tìm kiếm được gửi tới máy chủ MedAssist, không chia sẻ với bên thứ ba. Đăng xuất sẽ xoá dữ liệu ngoại tuyến đã lưu. Tên thuốc/bệnh nền dùng cho \"Lời khuyên\" được gửi tới máy chủ MedAssist cho lượt yêu cầu đó (không lưu, không ghi log), khác với việc gửi tới AI đám mây.';

  @override
  String get privacyRevokeConsent => 'Thu hồi đồng ý';

  @override
  String get privacyRevokeConsentBody => 'Lần mở app sau bạn sẽ được hỏi lại.';

  @override
  String get privacyDeleteAccount => 'Xoá tài khoản và dữ liệu';

  @override
  String get deleteAccountTitle => 'Xoá tài khoản?';

  @override
  String get deleteAccountBody => 'Xoá vĩnh viễn tài khoản, lịch sử và hội thoại trên máy chủ cùng hồ sơ sức khoẻ, lịch nhắc trên máy này. Không thể hoàn tác.';

  @override
  String get deleteAccountPasswordLabel => 'Nhập mật khẩu để xác nhận';

  @override
  String get deleteAccountConfirm => 'Xoá vĩnh viễn';

  @override
  String get deleteAccountDone => 'Đã xoá tài khoản và dữ liệu.';

  @override
  String get chatAiLabel => 'Trợ lý AI · chỉ trả lời từ nguồn đã nạp';

  @override
  String get navHome => 'Trang chủ';

  @override
  String get navReminders => 'Nhắc thuốc';

  @override
  String get navAssistant => 'Trợ lý';

  @override
  String get navProfile => 'Hồ sơ';

  @override
  String get scanMedicine => 'Quét thuốc';

  @override
  String get homeGreeting => 'Xin chào!';

  @override
  String get homeSubtitle => 'Hôm nay bạn cần uống thuốc gì?';

  @override
  String get homeTodayTitle => 'Lịch uống thuốc hôm nay';

  @override
  String get homeNoReminders => 'Chưa có lịch nhắc. Thêm lịch ở mục Nhắc thuốc.';

  @override
  String get homeSeeAll => 'Xem tất cả';

  @override
  String get homeSearchHint => 'Tìm thuốc theo tên…';

  @override
  String get homeShortcuts => 'Lối tắt';

  @override
  String get reminderSetAlarm => 'Đặt giờ nhắc';

  @override
  String get reminderSave => 'Lưu';

  @override
  String get reminderEveryDay => 'Mỗi ngày';

  @override
  String get reminderTimeLabel => 'Giờ uống';

  @override
  String get voiceMicTooltip => 'Nói với trợ lý';

  @override
  String get voiceListening => 'Đang nghe…';

  @override
  String get voiceUnavailable => 'Giọng nói chưa được bật';

  @override
  String get profileTitle => 'Hồ sơ của tôi';

  @override
  String get profileAccountSection => 'Tài khoản';

  @override
  String get profileDataSection => 'Dữ liệu & quyền riêng tư';

  @override
  String homeGreetingNamed(String name) {
    return 'Xin chào, $name!';
  }

  @override
  String get homeMarkTaken => 'Đã uống rồi';

  @override
  String get homeMarkedTaken => 'Đã uống hôm nay';

  @override
  String get doseASkip => 'Bỏ qua';

  @override
  String get doseASnooze => 'Hoãn 15 phút';

  @override
  String get doseASkippedToday => 'Đã bỏ qua hôm nay';

  @override
  String get doseANextDose => 'Liều tiếp theo';

  @override
  String get doseAMorning => 'Sáng';

  @override
  String get doseANoon => 'Trưa';

  @override
  String get doseAAfternoon => 'Chiều';

  @override
  String get doseAEvening => 'Tối';

  @override
  String get doseASnoozedMsg => 'Sẽ nhắc lại sau 15 phút';

  @override
  String get progressShortcut => 'Tiến độ';

  @override
  String get progressTitle => 'Tiến độ uống thuốc';

  @override
  String progressStreakDays(int count) {
    return '$count ngày liên tiếp!';
  }

  @override
  String get progressStreakZero => 'Uống thuốc hôm nay để bắt đầu chuỗi ngày liên tục';

  @override
  String get progressWeekTitle => '7 ngày gần đây';

  @override
  String get progressEmpty => 'Chưa có lượt uống thuốc nào được ghi nhận.';

  @override
  String get reminderDuplicateWarning => 'Bạn có vẻ đã có thuốc trùng tên này trong danh sách nhắc hoặc hồ sơ sức khoẻ — kiểm tra lại trước khi lưu.';

  @override
  String get lockSectionTitle => 'Khoá ứng dụng bằng mã PIN';

  @override
  String get lockSectionSubtitle => 'Yêu cầu nhập mã PIN khi mở lại ứng dụng sau khi rời đi một lúc.';

  @override
  String get lockEnterPinTitle => 'Nhập mã PIN để mở khoá';

  @override
  String get lockUnlock => 'Mở khoá';

  @override
  String get lockPinWrong => 'Sai mã PIN, thử lại.';

  @override
  String get lockSetNewPinTitle => 'Đặt mã PIN mới';

  @override
  String get lockPinLabel => 'Mã PIN (4-6 số)';

  @override
  String get lockPinConfirmLabel => 'Nhập lại mã PIN';

  @override
  String get lockPinTooShort => 'Mã PIN cần ít nhất 4 số.';

  @override
  String get lockPinMismatch => 'Hai mã PIN không khớp.';

  @override
  String get lockDisableTitle => 'Nhập mã PIN hiện tại để tắt khoá';

  @override
  String get lockSave => 'Lưu';

  @override
  String get lockEnabled => 'Đã bật khoá PIN';

  @override
  String get lockDisabled => 'Đã tắt khoá PIN';

  @override
  String get newsLeaveTitle => 'Rời ứng dụng để đọc bài?';

  @override
  String newsLeaveBody(String host) {
    return 'Bài viết sẽ mở trong trình duyệt tại $host. Trang này không do MedAssist quản lý và có thể thấy địa chỉ IP của bạn. Không có dữ liệu sức khoẻ nào được gửi đi.';
  }

  @override
  String newsVerifiedSources(String domains) {
    return 'Chỉ hiện link từ nguồn đã kiểm chứng: $domains';
  }

  @override
  String get newsLeaveOpen => 'Mở trình duyệt';

  @override
  String get newsLeaveCopy => 'Chép link';

  @override
  String get rxCreateFromScan => 'Tạo lịch nhắc từ đơn thuốc';

  @override
  String get rxNothingFound => 'Không tìm thấy thuốc nào trong đơn. Hãy thêm lịch nhắc thủ công.';

  @override
  String get rxReviewTitle => 'Xác nhận đơn thuốc';

  @override
  String get rxAiBanner => 'AI đã đọc đơn thuốc — kiểm tra lại trước khi lưu; đơn chữ viết tay có thể đọc sai.';

  @override
  String get rxInclude => 'Thêm vào lịch nhắc';

  @override
  String get rxMedicineName => 'Tên thuốc';

  @override
  String get rxDose => 'Liều mỗi lần';

  @override
  String get rxTimes => 'Giờ uống';

  @override
  String get rxAddTime => 'Thêm giờ';

  @override
  String get rxMealRelation => 'So với bữa ăn';

  @override
  String get rxMealNone => 'Không ghi';

  @override
  String get rxMealBefore => 'Trước ăn';

  @override
  String get rxMealAfter => 'Sau ăn';

  @override
  String get rxMealDuring => 'Trong bữa ăn';

  @override
  String get rxNeedsReview => 'Cần kiểm tra: không đọc rõ số lần dùng';

  @override
  String get rxDuplicate => 'Thuốc này đã có trong lịch nhắc';

  @override
  String get rxDiagnosisTitle => 'Chẩn đoán (đọc từ đơn)';

  @override
  String get rxAddDiagnosisToProfile => 'Thêm vào hồ sơ sức khoẻ';

  @override
  String get rxProfileConsentRequired => 'Cần đồng ý lưu hồ sơ sức khoẻ trước khi thêm chẩn đoán hoặc thuốc vào hồ sơ.';

  @override
  String get rxSave => 'Lưu và tạo lịch nhắc';

  @override
  String rxSaved(int count) {
    return 'Đã tạo $count lịch nhắc';
  }

  @override
  String get rxNoItemSelected => 'Chọn ít nhất một thuốc có tên và giờ uống.';

  @override
  String get rxSaveFailed => 'Không lưu được đơn thuốc. Vui lòng thử lại.';

  @override
  String get facilityTitle => 'Cơ sở y tế gần nhất';

  @override
  String get facilityCall115 => 'Gọi cấp cứu 115';

  @override
  String get facilitySampleBanner => 'Dữ liệu mẫu, cần kiểm chứng — hãy xác nhận địa chỉ trước khi đến.';

  @override
  String get facilityLocationRationale => 'Vị trí chỉ được dùng để tính khoảng cách ngay trên điện thoại này và không bao giờ được gửi đi đâu.';

  @override
  String get facilityUseLocation => 'Dùng vị trí của tôi';

  @override
  String get facilityLocating => 'Đang lấy vị trí…';

  @override
  String get facilityPermissionDenied => 'Bạn đã từ chối quyền vị trí. Danh sách chưa sắp theo khoảng cách; có thể cấp quyền trong Cài đặt.';

  @override
  String get facilityLocationOff => 'Định vị đang tắt. Hãy bật Vị trí rồi thử lại.';

  @override
  String get facilityLocationUnavailable => 'Không lấy được vị trí. Hãy thử lại hoặc tìm trên bản đồ.';

  @override
  String get facilityUnsupported => 'Thiết bị này không hỗ trợ tính năng vị trí.';

  @override
  String get facilityDirections => 'Chỉ đường';

  @override
  String get facilitySearchMore => 'Tìm thêm trên bản đồ';

  @override
  String get facilitySearchQuery => 'bệnh viện gần đây';

  @override
  String facilityDistanceKm(String km) {
    return 'Cách $km km';
  }

  @override
  String get facilityOpenFailed => 'Không có ứng dụng nào trên máy mở được thao tác này.';

  @override
  String get facilityLoadFailed => 'Không tải được danh sách cơ sở y tế.';

  @override
  String get cloudAiTitle => 'AI đám mây (Claude)';

  @override
  String get cloudAiSubtitle => 'Trợ lý trả lời tự nhiên hơn bằng AI của Anthropic. Mặc định tắt.';

  @override
  String get cloudAiDialogTitle => 'Dùng AI đám mây?';

  @override
  String get cloudAiDialogBody => 'Người nhận: Anthropic (Claude, AI đám mây).\n\nSẽ gửi: câu hỏi của bạn (đã xoá thông tin định danh như tên, SĐT, email, ngày tháng, số giấy tờ), tên thuốc bạn đang dùng, và bệnh nền/dị ứng bạn đã nhập trong hồ sơ.\n\nLưu ý: với \"Lời khuyên\", tên thuốc/bệnh nền được gửi tới máy chủ MedAssist cho lượt yêu cầu đó (không lưu, không ghi log) - khác với việc gửi tới AI đám mây.\n\nKhông gửi: ảnh đơn thuốc, vị trí, hồ sơ đầy đủ.\n\nBạn có thể tắt bất cứ lúc nào. Thông tin chỉ để tham khảo, không phải chẩn đoán.';

  @override
  String get cloudAiAgree => 'Đồng ý';

  @override
  String get cloudAiDecline => 'Không';

  @override
  String cloudAiGrantedAt(String when) {
    return 'Đã đồng ý lúc $when';
  }

  @override
  String get cloudAiChatHint => 'Bật \'AI đám mây\' trong Hồ sơ → Quyền riêng tư để trợ lý trả lời tự nhiên hơn';

  @override
  String get cloudAiHintDismiss => 'Ẩn gợi ý';

  @override
  String get emergencyCall => 'Gọi 115';

  @override
  String get emergencyNearest => 'Cơ sở y tế gần nhất';

  @override
  String get emergencyDialFailed => 'Không thể mở trình quay số. Hãy gọi 115 thủ công.';

  @override
  String get lifestyleTitle => 'Lời khuyên cho bạn';

  @override
  String get lifestyleEmpty => 'Chưa có lời khuyên nào lúc này.';

  @override
  String get lifestyleError => 'Không tải được lời khuyên.';

  @override
  String get lifestyleRefresh => 'Làm mới lời khuyên';

  @override
  String lifestyleSource(String name) {
    return 'Nguồn: $name';
  }

  @override
  String get lifestyleAiBadge => 'AI';

  @override
  String get polishDViewMedicine => 'Xem công dụng & tác dụng phụ';

  @override
  String polishDQuoteLabel(String id) {
    return 'Trích dẫn ($id)';
  }

  @override
  String get stockBPillsTotal => 'Số viên được cấp (không bắt buộc)';

  @override
  String get stockBPillsLeft => 'Số viên còn lại (không bắt buộc)';

  @override
  String get stockBPillsPerDose => 'Viên mỗi liều';

  @override
  String get stockBConfirmed => 'Tôi đã kiểm tra';

  @override
  String get stockBNeedsCheck => 'Hãy đối chiếu mục này với đơn thuốc và tick xác nhận.';

  @override
  String stockBLowStock(String name, int count) {
    return 'Sắp hết thuốc: $name còn $count viên — hãy mua thêm hoặc hỏi dược sĩ';
  }

  @override
  String get stockBEmptyReminderTitle => 'Chưa có lịch nhắc';

  @override
  String get stockBScanPrescription => 'Quét đơn thuốc';

  @override
  String get reportCTitle => 'Báo cáo tuân thủ';

  @override
  String get reportCRange7 => '7 ngày';

  @override
  String get reportCRange30 => '30 ngày';

  @override
  String get reportCRate => 'Tỉ lệ uống đúng';

  @override
  String get reportCTaken => 'Đã uống';

  @override
  String get reportCSkipped => 'Bỏ qua';

  @override
  String get reportCLate => 'Uống trễ';

  @override
  String get reportCSnoozed => 'Hoãn';

  @override
  String get reportCStreakNow => 'Streak hiện tại';

  @override
  String get reportCStreakBest => 'Streak dài nhất';

  @override
  String get reportCDays => 'ngày';

  @override
  String get reportCChartTitle => 'Số liều đã uống mỗi ngày';

  @override
  String get reportCNoRate => 'Chưa có dữ liệu';

  @override
  String get reportCNote => 'Chỉ để tham khảo; hãy trao đổi với bác sĩ/dược sĩ — không tự ý đổi hoặc ngưng thuốc.';

  @override
  String get reportCDoctorMode => 'Cho bác sĩ xem';

  @override
  String get reportCDoctorTitle => 'Tóm tắt cho bác sĩ';

  @override
  String get reportCBack => 'Quay lại';

  @override
  String get reportCEmpty => 'Chưa có liều nào được ghi. Hãy đánh dấu đã uống để xem báo cáo.';

  @override
  String get reportCDoctorHint => 'Chỉ xem, dữ liệu chỉ nằm trên điện thoại này.';

  @override
  String get reportCPeriod => 'Khoảng thời gian';

  @override
  String get fixELifestyleNote => 'Với \"Lời khuyên\", tên thuốc/bệnh nền được gửi tới máy chủ MedAssist cho lượt yêu cầu đó (không lưu, không ghi log). Việc này khác với gửi tới AI đám mây.';

  @override
  String fixELockTooMany(String time) {
    return 'Nhập sai quá nhiều lần. Thử lại sau $time.';
  }

  @override
  String get fixELockForgot => 'Quên PIN? Đăng xuất';

  @override
  String get fixELockUnreadable => 'Không đọc được kho bảo mật nên app giữ nguyên khoá. Hãy đăng xuất để đặt lại PIN.';

  @override
  String get modeGUnverified => 'Dữ liệu mẫu — chưa kiểm chứng';

  @override
  String modeGUnmatched(String conditions) {
    return 'Chưa có tài liệu dinh dưỡng đã kiểm chứng cho: $conditions. Hãy hỏi bác sĩ hoặc chuyên gia dinh dưỡng.';
  }

  @override
  String get modeGChipExtractive => 'Trích từ nguồn đã nạp';

  @override
  String get modeGChipCloud => 'Claude diễn đạt lại · vẫn theo nguồn';

  @override
  String get modeGSheetTitle => 'Hai chế độ trả lời';

  @override
  String get modeGOff => 'Khi tắt AI đám mây';

  @override
  String get modeGOn => 'Khi bật AI đám mây';

  @override
  String get modeGRowData => 'Dữ liệu có rời máy chủ MedAssist không?';

  @override
  String get modeGOffData => 'Không. Câu hỏi chỉ được xử lý trên máy chủ MedAssist, không gửi tới bên thứ ba.';

  @override
  String get modeGOnData => 'Có. Câu hỏi (đã xoá thông tin định danh) cùng thuốc/bệnh nền bạn đã đồng ý chia sẻ được gửi tới Anthropic (Claude).';

  @override
  String get modeGRowHow => 'Cách trả lời';

  @override
  String get modeGOffHow => 'Trích nguyên văn từ tài liệu đã nạp.';

  @override
  String get modeGOnHow => 'Claude diễn đạt lại tự nhiên hơn, nhưng chỉ được dùng nội dung của tài liệu đã nạp; không bám nguồn thì quay về bản trích nguyên văn.';

  @override
  String get modeGRowSame => 'Cái gì KHÔNG đổi';

  @override
  String get modeGSameBody => 'Nguồn trích dẫn luôn có; từ chối chẩn đoán và kê liều; cảnh báo cấp cứu vẫn hiện như cũ.';

  @override
  String get modeGWhere => 'Muốn đổi? Vào Hồ sơ → Quyền riêng tư → AI đám mây.';

  @override
  String get modeGClose => 'Đã hiểu';

  @override
  String get modeGConsentModes => 'Tắt: câu trả lời trích nguyên văn từ tài liệu đã nạp, không gửi gì tới bên thứ ba. Bật: Claude (Anthropic) diễn đạt lại tự nhiên hơn, nhưng vẫn phải bám nguồn.';

  @override
  String get memHTitle => 'Dữ liệu trợ lý nhớ về tôi';

  @override
  String get memHSubtitle => 'Xem, tắt hoặc xoá những gì trợ lý nhớ — chỉ lưu trên máy này.';

  @override
  String get memHToggle => 'Cho trợ lý nhớ triệu chứng của tôi';

  @override
  String get memHToggleBody => 'Lưu CHỈ trên máy, được mã hoá, bạn xoá được bất cứ lúc nào. Mỗi lượt hỏi chỉ gửi ngữ cảnh tối thiểu tới máy chủ MedAssist; máy chủ không lưu.';

  @override
  String get memHNeedProfile => 'Cần bật Hồ sơ sức khoẻ (đồng ý lưu) trước khi dùng tính năng này.';

  @override
  String get memHUnreadable => 'Không đọc được dữ liệu an toàn';

  @override
  String get memHUnreadableBody => 'Dữ liệu mã hoá trên máy không đọc được. Ứng dụng sẽ không ghi đè lên nó. Bạn có thể xoá toàn bộ dữ liệu trợ lý để bắt đầu lại.';

  @override
  String get memHStoredTitle => 'Đang lưu trên máy';

  @override
  String memHConditions(int count) {
    return 'Bệnh nền: $count';
  }

  @override
  String memHAllergies(int count) {
    return 'Dị ứng: $count';
  }

  @override
  String memHMeds(int count) {
    return 'Thuốc đang dùng: $count';
  }

  @override
  String memHEpisodes(int count) {
    return 'Nhật ký triệu chứng: $count lần ghi';
  }

  @override
  String get memHNone => 'Chưa có';

  @override
  String get memHEpisodesTitle => '20 lần ghi gần nhất';

  @override
  String get memHRecurringTag => 'lặp lại';

  @override
  String get memHPreviewTitle => 'Trợ lý sẽ dùng gì cho lần hỏi tiếp theo';

  @override
  String get memHPreviewConditions => 'Bệnh nền';

  @override
  String get memHPreviewAllergies => 'Dị ứng';

  @override
  String get memHPreviewSymptoms => 'Triệu chứng gần đây';

  @override
  String get memHPreviewNothing => 'Không gửi gì thêm ngoài câu hỏi của bạn.';

  @override
  String get memHCloudYes => 'AI đám mây: chỉ nhận câu hỏi đã xoá định danh (bạn đã đồng ý riêng).';

  @override
  String get memHCloudNo => 'AI đám mây: không nhận gì (bạn chưa đồng ý).';

  @override
  String get memHClearLog => 'Xoá nhật ký triệu chứng';

  @override
  String get memHClearAll => 'Xoá toàn bộ dữ liệu trợ lý';

  @override
  String get memHClearLogConfirm => 'Xoá toàn bộ nhật ký triệu chứng trên máy này?';

  @override
  String get memHClearAllConfirm => 'Xoá hồ sơ sức khoẻ, nhật ký triệu chứng và tắt trí nhớ trợ lý trên máy này? Không thể hoàn tác.';

  @override
  String get memHCleared => 'Đã xoá';

  @override
  String memHChatSaved(String terms) {
    return 'Đã ghi nhớ trên máy: $terms';
  }

  @override
  String get memHChatForget => 'Xoá';

  @override
  String get memHRecurring => 'Triệu chứng lặp lại';

  @override
  String get srcJTitle => 'Nguồn';

  @override
  String srcJRetrieved(String date) {
    return 'Lấy dữ liệu: $date';
  }

  @override
  String srcJLicense(String text) {
    return 'Giấy phép: $text';
  }

  @override
  String srcJReviewed(String name) {
    return 'Đã kiểm duyệt: $name';
  }

  @override
  String get srcJNotReviewed => 'Chưa có người kiểm duyệt';

  @override
  String get srcJOpen => 'Mở nguồn gốc';

  @override
  String srcJOpenHost(String host) {
    return 'Mở nguồn gốc ($host)';
  }

  @override
  String get srcJNoLink => 'Chưa có liên kết gốc';

  @override
  String get srcJVerified => 'Nguồn đã kiểm chứng';

  @override
  String get rxCheckCatalog => 'Đối chiếu danh mục thuốc';

  @override
  String get rxCheckCatalogNote => 'Gửi tên thuốc tới máy chủ MedAssist để tra danh mục (không lưu, không ghi log).';

  @override
  String get rxCatalogExact => 'Có trong danh mục thuốc';

  @override
  String get rxCatalogSimilar => 'Không trùng khớp hoàn toàn — xem gợi ý bên dưới và kiểm tra chính tả';

  @override
  String get rxCatalogNone => 'Không tìm thấy trong danh mục — có thể OCR đọc sai, hãy đối chiếu với đơn';

  @override
  String get rxCatalogError => 'Không đối chiếu được (mất mạng?). Hãy kiểm tra tay.';

  @override
  String get rxUseThisName => 'Dùng tên này';

  @override
  String rxCatalogStrength(String value) {
    return 'Hoạt chất · hàm lượng: $value';
  }

  @override
  String rxLowConfidence(String reasons) {
    return 'Độ chắc chắn thấp: $reasons';
  }

  @override
  String get rxIssueFrequency => 'không rõ số lần dùng';

  @override
  String get rxIssueDose => 'thiếu liều';

  @override
  String get rxIssueShortName => 'tên quá ngắn';

  @override
  String get rxIssueOddChars => 'tên có ký tự lạ';

  @override
  String get rxIssueNotInCatalog => 'không có trong danh mục';

  @override
  String get displaySection => 'Hiển thị';

  @override
  String get displayTheme => 'Giao diện';

  @override
  String get displayThemeSystem => 'Theo máy';

  @override
  String get displayThemeLight => 'Sáng';

  @override
  String get displayThemeDark => 'Tối';

  @override
  String get displayTextSize => 'Cỡ chữ';

  @override
  String get displayTextNormal => 'Chuẩn';

  @override
  String get displayTextLarge => 'Lớn';

  @override
  String get displayTextExtraLarge => 'Rất lớn';

  @override
  String get doseASnoozeMenu => 'Hoãn…';

  @override
  String doseASnoozeFor(int minutes) {
    return 'Hoãn $minutes phút';
  }

  @override
  String doseASnoozedFor(int minutes) {
    return 'Sẽ nhắc lại sau $minutes phút';
  }
}
