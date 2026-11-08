# google_mlkit_text_recognition tham chiếu tuỳ chọn nhận dạng chữ Hán/Devanagari/Nhật/Hàn.
# App chỉ dùng bộ nhận dạng chữ Latin nên các lớp này không được đóng gói -> bỏ cảnh báo R8.
-dontwarn com.google.mlkit.vision.text.chinese.**
-dontwarn com.google.mlkit.vision.text.devanagari.**
-dontwarn com.google.mlkit.vision.text.japanese.**
-dontwarn com.google.mlkit.vision.text.korean.**

# flutter_local_notifications lưu lịch nhắc bằng Gson (reflection): R8 không được bỏ/đổi tên
# các lớp này, nếu không nhắc thuốc hỏng riêng ở bản release.
-keep class com.dexterous.flutterlocalnotifications.** { *; }
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.google.gson.reflect.TypeToken { *; }
-keep class * extends com.google.gson.reflect.TypeToken
