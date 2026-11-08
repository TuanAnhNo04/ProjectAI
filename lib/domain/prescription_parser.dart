/// Bộ đọc đơn thuốc từ văn bản OCR — chạy hoàn toàn trên máy, thuần Dart.
///
/// Đầu vào là text OCR thô; đầu ra CHỈ gồm các dòng thuốc đã tách (tên, liều,
/// số lần/ngày, buổi, quan hệ bữa ăn). Mọi dòng thông tin cá nhân (họ tên, tuổi,
/// địa chỉ, SĐT, BHYT, mã BN, bác sĩ, ngày khám...) bị bỏ hẳn ngay từ đầu và
/// không bao giờ xuất hiện trong kết quả. Không lưu/ghi log text OCR.
library;

/// Quan hệ với bữa ăn.
enum MealRelation { none, before, after, during }

/// Buổi trong ngày ghi trên đơn.
enum DaySlot { morning, noon, afternoon, evening }

/// Giờ mặc định cho từng buổi.
const Map<DaySlot, String> defaultSlotTimes = {
  DaySlot.morning: '08:00',
  DaySlot.noon: '12:00',
  DaySlot.afternoon: '17:00',
  DaySlot.evening: '20:00',
};

/// Giờ mặc định theo số lần/ngày khi đơn không ghi buổi cụ thể.
const Map<int, List<String>> defaultTimesPerDay = {
  1: ['08:00'],
  2: ['08:00', '20:00'],
  3: ['08:00', '12:00', '18:00'],
  4: ['08:00', '12:00', '16:00', '20:00'],
};

class ParsedPrescriptionItem {
  const ParsedPrescriptionItem({
    required this.medicineName,
    this.doseText = '',
    this.timesPerDay,
    this.slots = const [],
    this.times = const ['08:00'],
    this.mealRelation = MealRelation.none,
    this.needsReview = false,
  });

  final String medicineName;

  /// Liều mỗi lần, ví dụ "1 viên"; rỗng nếu không đọc được.
  final String doseText;

  /// Số lần/ngày đọc được; `null` nếu không rõ.
  final int? timesPerDay;

  /// Buổi ghi rõ trên đơn (sáng/trưa/chiều/tối).
  final List<DaySlot> slots;

  /// Giờ nhắc đề xuất dạng 'HH:mm'.
  final List<String> times;

  final MealRelation mealRelation;

  /// `true` khi không đọc được tần suất → chỉ đề xuất 08:00, người dùng phải xem lại.
  final bool needsReview;

  @override
  String toString() => 'ParsedPrescriptionItem(times: ${times.length}, '
      'needsReview: $needsReview)';
}

/// Tách danh sách thuốc từ text OCR của đơn thuốc.
List<ParsedPrescriptionItem> parsePrescription(String ocrText) {
  final builders = <_ItemBuilder>[];
  for (final line in _cleanLines(ocrText)) {
    final folded = foldVietnamese(line).toLowerCase();
    if (_isPersonalInfo(folded) || _isIgnorable(folded)) continue;

    final numbered = _numbered.hasMatch(folded);
    final usageAt = _usageStart(folded);

    if (numbered || (usageAt != 0 && _looksLikeMedicine(folded))) {
      final cut = usageAt ?? line.length;
      final name = _cleanName(line.substring(0, cut));
      if (!_hasName(name)) {
        // Dòng chỉ có số thứ tự/số lượng: nếu có hướng dẫn thì gắn vào thuốc trước.
        if (usageAt != null && builders.isNotEmpty) {
          builders.last.usage.add(line);
        }
        continue;
      }
      final b = _ItemBuilder(name);
      if (usageAt != null) b.usage.add(line.substring(usageAt));
      builders.add(b);
    } else if (usageAt != null && builders.isNotEmpty) {
      builders.last.usage.add(line);
    }
  }
  return [for (final b in builders) b.build()];
}

/// Chẩn đoán ghi trên đơn ("Chẩn đoán: ..."), `null` nếu không có.
String? extractDiagnosis(String ocrText) {
  for (final line in _cleanLines(ocrText)) {
    final folded = foldVietnamese(line).toLowerCase();
    final m = RegExp(r'^\s*chan\s*doan\s*[:.\-]?\s*').firstMatch(folded);
    if (m == null) continue;
    var value = line.substring(m.end).trim();
    // Bỏ mã ICD dạng "(J06)" hoặc "J06 -" ở đầu.
    value = value
        .replaceFirst(RegExp(r'^\(?[A-Z]\d{2}(\.\d+)?\)?\s*[-:–]?\s*'), '')
        .trim();
    value = value.replaceAll(RegExp(r'[;,.\s]+$'), '');
    return value.isEmpty ? null : value;
  }
  return null;
}

// --- Chuẩn hoá -------------------------------------------------------------

const _accented =
    'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ';
const _plain =
    'aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd'
    'AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD';

final Map<int, int> _foldMap = {
  for (var i = 0; i < _accented.length; i++)
    _accented.codeUnitAt(i): _plain.codeUnitAt(i),
};

/// Bỏ dấu tiếng Việt, giữ nguyên độ dài chuỗi (mỗi ký tự → 1 ký tự) để chỉ số
/// trên chuỗi đã bỏ dấu dùng được cho chuỗi gốc.
String foldVietnamese(String s) => String.fromCharCodes(
      s.codeUnits.map((c) => _foldMap[c] ?? c),
    );

Iterable<String> _cleanLines(String text) sync* {
  // Bỏ dấu tổ hợp rời (Unicode NFD) để chỉ số giữa bản gốc/bản bỏ dấu khớp nhau.
  final normalized = text
      .replaceAll(RegExp('[\u0300-\u036f]'), '')
      .replaceAll(RegExp(r'[\t\u00a0]'), ' ');
  for (final raw in normalized.split(RegExp(r'\r?\n'))) {
    final line = raw.replaceAll(RegExp(r' {2,}'), ' ').trim();
    if (line.isNotEmpty) yield line;
  }
}

// --- Nhận dạng dòng --------------------------------------------------------

final _personal = RegExp(
  r'^[\W\d_]*(ho\s*(va\s*)?ten|nguoi\s*ke\s*don|ten\s*(benh\s*nhan|nguoi\s*benh)|benh\s*nhan|'
  r'nguoi\s*benh|tuoi|nam\s*sinh|ngay\s*sinh|gioi(\s*tinh)?|dia\s*chi|'
  r'sdt|s\.?d\.?t|so\s*dien\s*thoai|dien\s*thoai|dt\b|bhyt|so\s*the|ma\s*the|'
  r'the\s*bhyt|ma\s*bn|ma\s*benh\s*nhan|ma\s*so|bac\s*si|bs\b|y\s*si|'
  r'ngay\s*kham|nguoi\s*kham|ky\s*ten|cmnd|cccd|email)',
);
final _personalAnywhere = RegExp(
  r'\b(ho\s*ten|nam\s*sinh|gioi\s*tinh|dia\s*chi|bhyt|ma\s*bn|'
  r'ngay\s*kham|so\s*dien\s*thoai|sdt)\b',
);
final _phone =
    RegExp(r'(\+?84|\b0)[\s.\-]?\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}\b');
final _dateLine = RegExp(r'\bngay\s*\d{1,2}\s*thang\s*\d{1,2}');

bool _isPersonalInfo(String folded) =>
    _personal.hasMatch(folded) ||
    _personalAnywhere.hasMatch(folded) ||
    _phone.hasMatch(folded) ||
    _dateLine.hasMatch(folded);

final _ignorable = RegExp(
  r'^\W*(chan\s*doan|loi\s*dan|tai\s*kham|ghi\s*chu|dan\s*do|cong\s*khoan|'
  r'tong\s*(so|cong)|don\s*thuoc|benh\s*vien|phong\s*kham|khoa\b|so\s*y\s*te)',
);

bool _isIgnorable(String folded) => _ignorable.hasMatch(folded);

final _numbered = RegExp(r'^\s*\d{1,2}\s*[.):/\-]\s*\S');

final _strength = RegExp(
  r'\d+([.,]\d+)?\s*(mg|mcg|µg|ug|g|ml|iu|ui|%)(?![a-z])',
);
final _form = RegExp(r'\b(vien|goi|ong|chai|tuyp|lo|hop)\b');

bool _looksLikeMedicine(String folded) =>
    (_strength.hasMatch(folded) || _form.hasMatch(folded)) &&
    RegExp(r'[a-z]{3,}').hasMatch(folded.replaceAll(_form, ''));

/// Vị trí bắt đầu phần hướng dẫn dùng trong dòng (chuỗi đã bỏ dấu), hoặc null.
final _usageKeyword = RegExp(
  r'\b(uong|ngay\s*(uong|dung|\d)|moi\s*lan|sang|trua|chieu|toi(?!\s*(da|thieu))|'
  r'truoc\s*(an|bua)|sau\s*(an|bua)|trong\s*bua|cach\s*dung|lieu\s*dung|'
  r'\d+\s*lan)\b|[x×*]\s*\d+\s*lan',
);

int? _usageStart(String folded) => _usageKeyword.firstMatch(folded)?.start;

final _qtyTail = [
  RegExp(r'\b(sl|so\s*luong)\b.*$', caseSensitive: false),
  RegExp(r'[x×*]\s*\d+\s*(vien|goi|ong|chai|tuyp|hop|vi|lo)\b.*$'),
  RegExp(r'[\s:\-–]+\d+\s*(vien|goi|ong|chai|tuyp|hop|vi|lo)\s*$'),
];

String _cleanName(String raw) {
  var s = raw.replaceFirst(RegExp(r'^\s*\d{1,2}\s*[.):/\-]\s*'), '');
  for (final re in _qtyTail) {
    final m = re.firstMatch(foldVietnamese(s).toLowerCase());
    if (m != null) s = s.substring(0, m.start);
  }
  s = s.replaceAll(RegExp(r'[\s\-–:;,.|_]+$'), '').trim();
  return s.replaceAll(RegExp(r'\s{2,}'), ' ');
}

bool _hasName(String name) =>
    RegExp(r'[a-z]{3,}').hasMatch(foldVietnamese(name).toLowerCase());

// --- Phân tích hướng dẫn dùng ----------------------------------------------

const _doseUnits = r'(vien|goi|ong|ml|giot|nhat|xit|mieng|thia|muong|lieu|cai)';
final _doseRe =
    RegExp(r'(\d+(?:[.,/]\d+)?|nua|mot|hai)\s*' + _doseUnits + r'\b');

final _slotRe = {
  DaySlot.morning: RegExp(r'\bsang\b'),
  DaySlot.noon: RegExp(r'\btrua\b'),
  DaySlot.afternoon: RegExp(r'\bchieu\b'),
  DaySlot.evening: RegExp(r'\btoi\b(?!\s*(da|thieu))'),
};

final _timesRes = [
  RegExp(r'(\d+)\s*l\s?a\s?n\s*(/|\s|mot|moi|trong)*\s*ngay'),
  RegExp(r'ngay\s*(uong|dung|boi|nho|xit|ngam|tiem)?\s*(\d+)\s*l\s?a\s?n'),
  RegExp(r'[x×*]\s*(\d+)\s*l\s?a\s?n'),
];

class _ItemBuilder {
  _ItemBuilder(this.name);
  final String name;
  final List<String> usage = [];

  ParsedPrescriptionItem build() {
    final original = usage.join(' ');
    final u = foldVietnamese(original).toLowerCase();

    int? times;
    for (final re in _timesRes) {
      final m = re.firstMatch(u);
      if (m == null) continue;
      String? digits;
      for (var g = 1; g <= m.groupCount; g++) {
        final v = m.group(g);
        if (v != null && RegExp(r'^\d+$').hasMatch(v)) {
          digits = v;
          break;
        }
      }
      final n = int.tryParse(digits ?? '');
      if (n != null && n > 0) {
        times = n;
        break;
      }
    }

    final slots = [
      for (final e in _slotRe.entries)
        if (e.value.hasMatch(u)) e.key,
    ];

    // Liều theo từng buổi ("Sáng 2 viên, tối 1 viên").
    final slotDoses = <DaySlot, String>{};
    for (final e in _slotRe.entries) {
      final m = RegExp(
        e.value.pattern + r'\s*[:\-]?\s*' + _doseRe.pattern,
      ).firstMatch(u);
      if (m != null) {
        final d = _doseRe.firstMatch(u.substring(m.start))!;
        final start = m.start + d.start;
        slotDoses[e.key] = original.substring(start, start + d.end - d.start);
      }
    }
    String dose;
    final unique = slotDoses.values.map((d) => d.toLowerCase()).toSet();
    if (unique.length > 1) {
      dose = slotDoses.entries
          .map((e) => '${_slotLabelVi[e.key]} ${e.value}')
          .join(', ');
    } else if (unique.length == 1) {
      dose = slotDoses.values.first;
    } else {
      final m = _doseRe.firstMatch(u);
      dose = m == null ? '' : original.substring(m.start, m.end);
    }

    final meal =
        RegExp(r'truoc\s*(khi\s*)?(an|bua)|luc\s*doi|khi\s*doi').hasMatch(u)
            ? MealRelation.before
            : RegExp(r'sau\s*(khi\s*)?(an|bua)|luc\s*no').hasMatch(u)
                ? MealRelation.after
                : RegExp(r'trong\s*(khi\s*)?(an|bua)|cung\s*bua|khi\s*an')
                        .hasMatch(u)
                    ? MealRelation.during
                    : MealRelation.none;

    List<String> slotTimes;
    var review = false;
    if (slots.isNotEmpty) {
      slotTimes = [for (final s in slots) defaultSlotTimes[s]!];
      if (times != null && times != slots.length) review = true;
    } else if (times != null && defaultTimesPerDay.containsKey(times)) {
      slotTimes = defaultTimesPerDay[times]!;
    } else {
      slotTimes = const ['08:00'];
      review = true;
    }

    return ParsedPrescriptionItem(
      medicineName: name,
      doseText: dose.replaceAll(RegExp(r'\s+'), ' ').trim(),
      timesPerDay: times ?? (slots.isEmpty ? null : slots.length),
      slots: slots,
      times: List.unmodifiable(slotTimes),
      mealRelation: meal,
      needsReview: review,
    );
  }
}

const _slotLabelVi = {
  DaySlot.morning: 'sáng',
  DaySlot.noon: 'trưa',
  DaySlot.afternoon: 'chiều',
  DaySlot.evening: 'tối',
};
