// Render smoke test: mỗi màn hình dựng ở 360x800 (textScale 1.0 và 1.3) và
// khẳng định không có RenderFlex overflow / lỗi build (`takeException()` null).
//
// Toàn bộ dữ liệu là GIẢ. Đặt `QA_RENDER=1` để xuất PNG vào `docs/qa/screens/`
// (mặc định KHÔNG ghi file nào).
//
//   QA_RENDER=1 flutter test test/screens/render_smoke_test.dart

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/data/local/app_lock_store.dart';
import 'package:image_processing_app/data/local/assistant_memory_store.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/local/health_vault.dart';
import 'package:image_processing_app/data/local/cloud_ai_consent_store.dart';
import 'package:image_processing_app/data/models/chat_message.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/lifestyle_tip.dart';
import 'package:image_processing_app/data/models/news_article.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';
import 'package:image_processing_app/domain/ports/news_repository.dart';
import 'package:image_processing_app/domain/prescription_parser.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/agent_chat_screen.dart';
import 'package:image_processing_app/screens/app_shell.dart';
import 'package:image_processing_app/screens/assistant_memory_screen.dart';
import 'package:image_processing_app/screens/chat_controller.dart';
import 'package:image_processing_app/screens/home_tab.dart';
import 'package:image_processing_app/screens/lock_screen.dart';
import 'package:image_processing_app/screens/news_screen.dart';
import 'package:image_processing_app/screens/prescription_review_screen.dart';
import 'package:image_processing_app/screens/privacy_screen.dart';
import 'package:image_processing_app/screens/progress_screen.dart';
import 'package:image_processing_app/screens/reminder_screen.dart';
import 'package:image_processing_app/widgets/external_link.dart';

/// Lỗi cũ `_SummaryGrid` (Row(stretch) trong ListView -> infinite height) đã sửa bằng
/// IntrinsicHeight; các test Tiến độ có dữ liệu giờ chạy thật (cờ skip = null).
const String? _progressGridDefect = null;

final bool _qa = Platform.environment['QA_RENDER'] == '1';
const _fontDir =
    'D:/flutter_windows_3.19.4-stable/flutter/bin/cache/artifacts/material_fonts';
const _outDir = 'docs/qa/screens';

// --- Fakes -----------------------------------------------------------------

class _FakeReminders extends ReminderController {
  _FakeReminders(this.items);
  final List<MedicationReminder> items;
  @override
  Future<List<MedicationReminder>> build() => SynchronousFuture(items);
  @override
  Future<void> load() async {}
}

class _FakeAdherence extends AdherenceController {
  _FakeAdherence(this.items);
  final List<AdherenceRecord> items;
  @override
  Future<List<AdherenceRecord>> build() => SynchronousFuture(items);
  @override
  Future<void> load() async {}
}

class _FakeProfile extends HealthProfileController {
  @override
  Future<HealthProfile?> load() async => null;
}

/// Hồ sơ giả đã consent, dùng cho màn "Dữ liệu trợ lý nhớ về tôi".
class _ConsentedProfile extends HealthProfileController {
  @override
  Future<HealthProfile?> build() => SynchronousFuture(HealthProfile(
        conditions: const ['Bệnh nền giả A', 'Bệnh nền giả B'],
        allergies: const ['Dị ứng giả'],
        currentMedications: const ['Thuốc Giả 500mg'],
        consentGivenAt: DateTime(2026, 1, 1),
      ));
  @override
  Future<HealthProfile?> load() async => state.valueOrNull;
}

InMemoryHealthVault _vaultWithEpisodes() {
  final v = InMemoryHealthVault();
  final now = DateTime.now();
  v.data['medassist.vault.episodes'] = '['
      '{"at":"${now.subtract(const Duration(days: 1)).toIso8601String()}",'
      '"terms":["buồn nôn","đau đầu kéo dài và rất khó chịu về đêm"],'
      '"meds":["Thuốc Giả 500mg"],"recurring":true},'
      '{"at":"${now.subtract(const Duration(days: 3)).toIso8601String()}",'
      '"terms":["ho"],"meds":[],"recurring":false}]';
  return v;
}

class _FakeNews implements NewsRepository {
  _FakeNews(this.snap);
  final NewsSnapshot snap;
  @override
  Future<NewsSnapshot> fetchNews({bool refresh = false}) async => snap;
}

class _FakeLock implements AppLockStore {
  _FakeLock({this.left = Duration.zero});
  final Duration left;
  @override
  Future<PinState> pinState() async => PinState.hasPin;
  @override
  Future<bool> hasPin() async => true;
  @override
  Future<void> setPin(String pin) async {}
  @override
  Future<bool> verifyPin(String pin) async => false;
  @override
  Future<Duration> lockoutRemaining() async => left;
  @override
  Future<void> clearPin() async {}
}

// --- Dữ liệu giả -----------------------------------------------------------

const _reminders = <MedicationReminder>[
  MedicationReminder(
      id: 1,
      medicineName: 'FakeMed Sáng',
      hour: 6,
      minute: 30,
      dosage: '1 viên sau ăn',
      pillsLeft: 20),
  MedicationReminder(
      id: 2,
      medicineName: 'FakeMed Trưa có tên rất dài để thử cắt chữ',
      hour: 12,
      minute: 30,
      dosage: '2 viên trước bữa ăn, uống nhiều nước',
      pillsLeft: 2,
      pillsPerDose: 2),
  MedicationReminder(
      id: 3, medicineName: 'FakeMed Chiều', hour: 17, minute: 45),
  MedicationReminder(
      id: 4,
      medicineName: 'FakeMed Tối',
      hour: 21,
      minute: 0,
      dosage: '1 viên',
      pillsLeft: 30),
  MedicationReminder(
      id: 5, medicineName: 'FakeMed Khuya', hour: 23, minute: 59),
];

const _cit = LifestyleCitation(
  id: 'doc-1',
  medicineName: 'FakeMed A',
  source: 'Tờ hướng dẫn giả lập (fake) số 1',
  excerpt: 'Không dùng cùng rượu bia; uống sau ăn để giảm kích ứng dạ dày.',
);

const _tips = LifestyleSnapshot(
  generatedBy: 'claude',
  disclaimer: 'Thông tin tham khảo, không thay thế tư vấn bác sĩ.',
  tips: [
    LifestyleTip(
      title: 'Uống thuốc sau ăn để đỡ hại dạ dày',
      body: 'FakeMed A nên uống sau bữa ăn. Tránh rượu bia trong thời gian '
          'dùng thuốc. Nếu thấy đau bụng kéo dài, hãy hỏi dược sĩ.',
      kind: 'medication',
      citations: [_cit, _cit],
    ),
    LifestyleTip(
      title: 'Ngủ đủ giấc, đi bộ nhẹ mỗi ngày',
      body: 'Gợi ý chung về lối sống, không gắn với thuốc cụ thể.',
    ),
  ],
);

const _dietTips = LifestyleSnapshot(
  disclaimer: 'Thông tin tham khảo, không thay thế tư vấn bác sĩ.',
  unmatchedConditions: ['Bệnh giả A', 'Bệnh giả B'],
  tips: [
    LifestyleTip(
      title: 'Ăn nhạt, giảm muối trong bữa ăn hằng ngày',
      body: 'Nên hạn chế món mặn và đồ chế biến sẵn (nội dung giả lập).',
      kind: 'diet',
      citations: [
        LifestyleCitation(
          id: 'diet-fake-1',
          medicineName: '',
          source: 'Tài liệu dinh dưỡng giả lập số 1',
          excerpt: 'Hạn chế muối dưới 5 g mỗi ngày (fake).',
        ),
      ],
    ),
    LifestyleTip(
      title: 'Ăn nhiều rau xanh',
      body: 'Đã kiểm chứng (giả lập).',
      kind: 'diet',
      citations: [
        LifestyleCitation(
          id: 'diet-fake-2',
          medicineName: '',
          source: 'Nguồn kiểm chứng giả lập',
          isAuthoritative: true,
        ),
      ],
    ),
  ],
);

class _FakeChat extends ChatController {
  _FakeChat(this.items);
  final List<ChatMessage> items;
  @override
  List<ChatMessage> build() => items;
}

const _chatCit = Citation(
    documentId: 'doc-fake', quote: 'Trích đoạn giả', title: 'Tài liệu giả');

List<AdherenceRecord> _logs(int days) {
  final n = DateTime.now();
  const st = ['taken', 'late', 'skipped', 'snoozed'];
  return [
    for (var d = 0; d < days; d++)
      for (var r = 1; r <= 3; r++)
        AdherenceRecord(
          reminderId: r,
          takenAt: DateTime(n.year, n.month, n.day - d, 6 + r * 3, 5),
          status: st[(d + r) % 4 == 3 ? 2 : (d + r) % 2],
          skipped: (d + r) % 4 == 3,
        ),
  ];
}

NewsSnapshot _news({bool sample = false}) => NewsSnapshot(
      isSample: sample,
      verifiedSources: sample ? const [] : const ['fda.gov', 'who.int'],
      failedSources: sample ? const [] : const ['example.invalid'],
      disclaimer: 'Tin tham khảo, không phải lời khuyên y tế.',
      articles: [
        NewsArticle(
          id: 'n1',
          title: 'Tiêu đề bài viết giả lập rất dài để kiểm tra việc xuống '
              'dòng trong thẻ tin tức trên màn hình hẹp',
          summary: 'Tóm tắt giả lập: ' * 12,
          url: sample ? '' : 'https://example.invalid/a',
          source: 'fda.gov',
          publishedAt: DateTime(2026, 10, 1),
        ),
        NewsArticle(
          id: 'n2',
          title: 'Bài thứ hai',
          summary: 'Ngắn.',
          url: sample ? '' : 'https://example.invalid/b',
          source: 'who.int',
        ),
      ],
    );

// --- Harness ---------------------------------------------------------------

final _boundaryKey = GlobalKey();

Future<void> _pumpScreen(
  WidgetTester tester,
  Widget home, {
  required double scale,
  List<Override> overrides = const [],
}) async {
  if (Platform.environment['QA_DEBUG'] == '1') {
    final prev = FlutterError.onError;
    FlutterError.onError = (d) {
      debugPrint(d.toString(minLevel: DiagnosticLevel.debug));
      prev?.call(d);
    };
  }
  tester.view.physicalSize = const Size(360, 800);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(
    ProviderScope(
      overrides: [
        appLockStoreProvider.overrideWithValue(InMemoryAppLockStore()),
        cloudAiConsentStoreProvider
            .overrideWithValue(InMemoryCloudAiConsentStore()),
        cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
        healthProfileControllerProvider.overrideWith(_FakeProfile.new),
        healthVaultProvider.overrideWithValue(InMemoryHealthVault()),
        assistantMemoryStoreProvider
            .overrideWithValue(InMemoryAssistantMemoryStore()),
        ...overrides,
      ],
      child: RepaintBoundary(
        key: _boundaryKey,
        child: MaterialApp(
          debugShowCheckedModeBanner: false,
          locale: const Locale('vi'),
          theme: AppTheme.light(),
          localizationsDelegates: AppLocalizations.localizationsDelegates,
          supportedLocales: AppLocalizations.supportedLocales,
          builder: (context, child) => MediaQuery(
            data: MediaQuery.of(context).copyWith(
              textScaler: TextScaler.linear(scale),
            ),
            child: child!,
          ),
          home: home,
        ),
      ),
    ),
  );
  await tester.pump();
  await tester.pump(const Duration(seconds: 1));
}

Future<void> _snap(WidgetTester tester, String name) async {
  if (!_qa) return;
  final boundary =
      tester.renderObject<RenderRepaintBoundary>(find.byKey(_boundaryKey));
  await tester.runAsync(() async {
    final image = await boundary.toImage(pixelRatio: 2);
    final data = await image.toByteData(format: ui.ImageByteFormat.png);
    Directory(_outDir).createSync(recursive: true);
    File('$_outDir/$name.png').writeAsBytesSync(data!.buffer.asUint8List());
  });
}

Future<void> _scrollDown(WidgetTester tester, double dy) async {
  await tester.drag(find.byType(Scrollable).first, Offset(0, -dy));
  await tester.pump(const Duration(milliseconds: 500));
}

/// Mỗi kịch bản chạy ở scale 1.0 và 1.3; PNG chỉ xuất cho scale 1.0.
void _scenario(
  String name,
  Widget Function() home,
  Future<void> Function(WidgetTester t, bool shoot) body, {
  List<Override> Function()? overrides,
  String? skip,
}) {
  for (final scale in const [1.0, 1.3]) {
    testWidgets('$name @x$scale', (tester) async {
      await _pumpScreen(tester, home(),
          scale: scale, overrides: overrides?.call() ?? const []);
      await body(tester, scale == 1.0);
      expect(tester.takeException(), isNull);
    }, skip: skip != null);
  }
}

List<Override> _health(List<MedicationReminder> r, List<AdherenceRecord> a) => [
      reminderControllerProvider.overrideWith(() => _FakeReminders(r)),
      adherenceControllerProvider.overrideWith(() => _FakeAdherence(a)),
    ];

Widget _home() => HomeTab(onOpenReminders: () {}, onOpenAssistant: () {});

Future<void> _loadFonts() async {
  Future<ByteData> bytes(String f) async =>
      ByteData.view(File('$_fontDir/$f').readAsBytesSync().buffer);
  final roboto = FontLoader('Roboto')
    ..addFont(bytes('roboto-regular.ttf'))
    ..addFont(bytes('roboto-medium.ttf'))
    ..addFont(bytes('roboto-bold.ttf'));
  await roboto.load();
  final icons = FontLoader('MaterialIcons')
    ..addFont(bytes('materialicons-regular.otf'));
  await icons.load();
}

void main() {
  setUpAll(() async {
    // Ahem (font mặc định của flutter_test) rộng gấp ~2 lần chữ thật -> báo
    // overflow giả; nạp Roboto thật nếu có trong cache SDK.
    if (Directory(_fontDir).existsSync()) await _loadFonts();
  });

  // ---- Khối Nguồn (có / không có link) ----
  _scenario(
      'SourceBlock: đủ metadata + link, và không link/không kiểm duyệt',
      () => Scaffold(
            body: ListView(padding: const EdgeInsets.all(16), children: const [
              SourceBlock(
                source: 'Nguồn giả lập có tên khá dài để kiểm tra xuống dòng',
                isAuthoritative: true,
                retrievedAt: '2026-10-01',
                license: 'CC-FAKE 4.0 (giả lập)',
                reviewedBy: 'Dược sĩ Giả',
                sourceUrl: 'https://example.invalid/tai-lieu',
              ),
              SizedBox(height: 12),
              SourceBlock(
                source: 'Nguồn không link',
                isAuthoritative: false,
                retrievedAt: 'không hợp lệ',
                sourceUrl: 'http://example.invalid/x',
              ),
            ]),
          ), (t, _) async {
    expect(find.text('Mở nguồn gốc (example.invalid)'), findsOneWidget);
    expect(find.text('Chưa có liên kết gốc'), findsOneWidget);
    expect(find.text('Chưa có người kiểm duyệt'), findsOneWidget);
    expect(find.text('Lấy dữ liệu: 01/10/2026'), findsOneWidget);
  });

  // ---- Home ----
  _scenario('HomeTab: nhắc + sắp hết thuốc + mẹo (có/không trích dẫn)', _home,
      (t, shoot) async {
    if (shoot) await _snap(t, 'home_top');
    await _scrollDown(t, 600);
    if (shoot) await _snap(t, 'home_mid');
    await _scrollDown(t, 1200);
    if (shoot) await _snap(t, 'home_bottom');
  },
      overrides: () => [
            ..._health(_reminders, _logs(3)),
            lifestyleProvider.overrideWith((ref) async => _tips),
          ]);

  _scenario('HomeTab: danh sách nhắc rỗng', _home, (t, _) async {},
      overrides: () => [
            ..._health(const [], const []),
            lifestyleProvider
                .overrideWith((ref) async => const LifestyleSnapshot(tips: [])),
          ]);

  // ---- Bottom sheet mẹo ----
  for (final withCit in [true, false]) {
    _scenario('Tip sheet ${withCit ? "có" : "không"} trích dẫn', _home,
        (t, shoot) async {
      final title = _tips.tips[withCit ? 0 : 1].title;
      final f = find.text(title);
      await t.scrollUntilVisible(f, 200,
          scrollable: find.byType(Scrollable).first);
      await t.pump(const Duration(milliseconds: 300));
      await t.tap(f);
      await t.pump(const Duration(milliseconds: 600));
      await t.pump(const Duration(milliseconds: 600));
      if (shoot && withCit) await _snap(t, 'tip_sheet');
    },
        overrides: () => [
              ..._health(_reminders, const []),
              lifestyleProvider.overrideWith((ref) async => _tips),
            ]);
  }

  // ---- Diet tips ----
  _scenario('HomeTab: mẹo dinh dưỡng + dữ liệu mẫu + chưa khớp bệnh', _home,
      (t, _) async {
    final f = find.text(_dietTips.tips.first.title);
    await t.scrollUntilVisible(f, 200,
        scrollable: find.byType(Scrollable).first);
    await t.pump(const Duration(milliseconds: 300));
    expect(find.text('Dữ liệu mẫu — chưa kiểm chứng'), findsOneWidget);
    expect(find.textContaining('Bệnh giả A, Bệnh giả B'), findsOneWidget);
    await t.tap(f);
    await t.pump(const Duration(milliseconds: 600));
    await t.pump(const Duration(milliseconds: 600));
    expect(find.text('Dữ liệu mẫu — chưa kiểm chứng'), findsNWidgets(2));
    expect(find.textContaining('Hạn chế muối'), findsOneWidget);
  },
      overrides: () => [
            ..._health(_reminders, const []),
            lifestyleProvider.overrideWith((ref) async => _dietTips),
          ]);

  // ---- Chip chế độ trả lời ----
  _scenario(
      'AgentChat: chip chế độ + sheet so sánh', () => const AgentChatScreen(),
      (t, _) async {
    expect(find.text('Trích từ nguồn đã nạp'), findsOneWidget);
    expect(find.text('Claude diễn đạt lại · vẫn theo nguồn'), findsOneWidget);
    await t.tap(find.text('Claude diễn đạt lại · vẫn theo nguồn'));
    await t.pump(const Duration(milliseconds: 600));
    await t.pump(const Duration(milliseconds: 600));
    expect(find.text('Hai chế độ trả lời'), findsOneWidget);
    expect(find.textContaining('Hồ sơ → Quyền riêng tư → AI đám mây'),
        findsOneWidget);
  },
      overrides: () => [
            chatControllerProvider.overrideWith(() => _FakeChat(const [
                  ChatMessage(role: ChatRole.user, text: 'Câu hỏi giả'),
                  ChatMessage(
                      role: ChatRole.assistant,
                      text: 'Trả lời giả [doc-fake]',
                      citations: [_chatCit],
                      mode: 'extractive'),
                  ChatMessage(
                      role: ChatRole.assistant,
                      text: 'Trả lời giả hai [doc-fake]',
                      citations: [_chatCit],
                      mode: 'cloud'),
                ])),
          ]);

  // ---- Reminders ----
  _scenario('ReminderScreen: nhóm theo buổi', () => const ReminderScreen(),
      (t, shoot) async {
    if (shoot) await _snap(t, 'reminders');
    await _scrollDown(t, 700);
  }, overrides: () => _health(_reminders, _logs(1)));

  _scenario('ReminderScreen: trạng thái rỗng', () => const ReminderScreen(),
      (t, _) async {},
      overrides: () => _health(const [], const []));

  // ---- Progress ----
  _scenario('ProgressScreen: không có log', () => const ProgressScreen(),
      (t, shoot) async {
    if (shoot) await _snap(t, 'progress_empty');
  }, overrides: () => _health(_reminders, const []));

  _scenario('ProgressScreen: 7 ngày', () => const ProgressScreen(),
      skip: _progressGridDefect, (t, shoot) async {
    if (shoot) await _snap(t, 'progress_7d');
    await _scrollDown(t, 700);
  }, overrides: () => _health(_reminders, _logs(7)));

  _scenario('ProgressScreen: 30 ngày', () => const ProgressScreen(),
      skip: _progressGridDefect, (t, _) async {
    await t.tap(find.text('30 ngày'));
    await t.pump(const Duration(milliseconds: 500));
    await _scrollDown(t, 700);
  }, overrides: () => _health(_reminders, _logs(30)));

  _scenario('ProgressScreen: Cho bác sĩ xem', () => const ProgressScreen(),
      (t, shoot) async {
    final btn = find.text('Cho bác sĩ xem');
    await t.ensureVisible(btn);
    await t.tap(btn);
    await t.pump(const Duration(milliseconds: 500));
    if (shoot) await _snap(t, 'progress_doctor');
    await _scrollDown(t, 700);
  }, overrides: () => _health(_reminders, const []));

  // ---- News ----
  for (final sample in [true, false]) {
    _scenario('NewsScreen ${sample ? "mẫu" : "thật"}', () => const NewsScreen(),
        (t, shoot) async {
      if (shoot) await _snap(t, sample ? 'news_sample' : 'news_real');
    },
        overrides: () => [
              newsRepositoryProvider
                  .overrideWithValue(_FakeNews(_news(sample: sample))),
            ]);
  }

  // ---- Review đơn thuốc ----
  _scenario(
      'PrescriptionReviewScreen: needsReview -> Lưu bị khoá',
      () => const PrescriptionReviewScreen(
            diagnosis: 'Chẩn đoán giả lập',
            items: [
              ParsedPrescriptionItem(
                  medicineName: 'FakeMed A 500mg',
                  doseText: '1 viên',
                  timesPerDay: 2,
                  times: ['08:00', '20:00'],
                  needsReview: true),
              ParsedPrescriptionItem(
                  medicineName: 'FakeMed B có tên dài để thử xuống dòng',
                  doseText: '2 viên sau ăn',
                  timesPerDay: 3,
                  times: ['07:00', '12:00', '19:00']),
            ],
          ), (t, shoot) async {
    if (shoot) await _snap(t, 'review');
    await _scrollDown(t, 1500);
    final saveFinder = find.byWidgetPredicate((w) => w is FilledButton).last;
    expect(t.widget<FilledButton>(saveFinder).onPressed, isNull);
    if (shoot) await _snap(t, 'review_bottom');
    final confirm = find.byKey(const ValueKey('confirmItem'));
    await _scrollDown(t, -1500);
    await t.ensureVisible(confirm);
    await t.tap(confirm);
    await t.pump();
    await _scrollDown(t, 800);
  }, overrides: () => _health(const [], const []));

  // ---- Privacy ----
  _scenario('PrivacyScreen', () => const PrivacyScreen(), (t, _) async {
    await t.scrollUntilVisible(find.text('Dữ liệu trợ lý nhớ về tôi'), 200,
        scrollable: find.byType(Scrollable).first);
    expect(find.text('Dữ liệu trợ lý nhớ về tôi'), findsOneWidget);
    await _scrollDown(t, 800);
  });

  // ---- Dữ liệu trợ lý nhớ về tôi ----
  _scenario('AssistantMemory: có dữ liệu + trí nhớ bật',
      () => const AssistantMemoryScreen(), (t, shoot) async {
    expect(find.textContaining('Bệnh nền: 2'), findsOneWidget);
    expect(find.textContaining('buồn nôn'), findsWidgets);
    expect(
        find.text('Trợ lý sẽ dùng gì cho lần hỏi tiếp theo'), findsOneWidget);
    expect(find.textContaining('Triệu chứng gần đây'), findsOneWidget);
    if (shoot) await _snap(t, 'assistant_memory');
    await _scrollDown(t, 1500);
    expect(find.text('Xoá toàn bộ dữ liệu trợ lý'), findsOneWidget);
    await t.tap(find.text('Xoá nhật ký triệu chứng'));
    await t.pump(const Duration(milliseconds: 600));
    expect(find.text('Xoá toàn bộ nhật ký triệu chứng trên máy này?'),
        findsOneWidget);
  },
      overrides: () => [
            healthProfileControllerProvider.overrideWith(_ConsentedProfile.new),
            healthVaultProvider.overrideWithValue(_vaultWithEpisodes()),
            assistantMemoryStoreProvider
                .overrideWithValue(InMemoryAssistantMemoryStore(value: true)),
          ]);
  _scenario('AssistantMemory: chưa consent hồ sơ',
      () => const AssistantMemoryScreen(), (t, _) async {
    expect(find.textContaining('Cần bật Hồ sơ sức khoẻ'), findsOneWidget);
    expect(
        find.text('Không gửi gì thêm ngoài câu hỏi của bạn.'), findsOneWidget);
  });
  _scenario('AssistantMemory: kho không đọc được',
      () => const AssistantMemoryScreen(), (t, _) async {
    expect(find.text('Không đọc được dữ liệu an toàn'), findsOneWidget);
    expect(find.text('Xoá toàn bộ dữ liệu trợ lý'), findsOneWidget);
  },
      overrides: () => [
            healthVaultProvider
                .overrideWithValue(InMemoryHealthVault()..unreadable = true),
          ]);

  // ---- Chat: dòng "Đã ghi nhớ" + ghi chú lặp lại ----
  _scenario('AgentChat: đã ghi nhớ triệu chứng + lặp lại',
      () => const AgentChatScreen(), (t, shoot) async {
    expect(
        find.textContaining('Đã ghi nhớ trên máy: buồn nôn'), findsOneWidget);
    expect(find.text('Triệu chứng lặp lại'), findsOneWidget);
    expect(find.text('Xoá'), findsOneWidget);
    if (shoot) await _snap(t, 'chat_symptom_note');
  },
      overrides: () => [
            chatControllerProvider.overrideWith(() => _FakeChat([
                  const ChatMessage(
                      role: ChatRole.user, text: 'Tôi bị buồn nôn'),
                  ChatMessage(
                    role: ChatRole.assistant,
                    text: 'Nội dung trả lời giả lập khá dài để kiểm tra việc '
                        'xuống dòng trên màn hình hẹp.',
                    symptom: const SymptomNoted(
                        terms: ['buồn nôn', 'đau đầu kéo dài'],
                        recurring: true),
                    savedEpisodeAt: DateTime(2026, 10, 1),
                  ),
                ])),
          ]);

  // ---- Lock ----
  _scenario('LockScreen: bình thường', () => LockScreen(onUnlocked: () {}),
      (t, _) async {},
      overrides: () => [appLockStoreProvider.overrideWithValue(_FakeLock())]);
  _scenario('LockScreen: bị khoá tạm', () => LockScreen(onUnlocked: () {}),
      (t, _) async {},
      overrides: () => [
            appLockStoreProvider
                .overrideWithValue(_FakeLock(left: const Duration(seconds: 45)))
          ]);
  _scenario('LockScreen: không đọc được kho',
      () => LockScreen(onUnlocked: () {}, unreadable: true), (t, _) async {},
      overrides: () => [appLockStoreProvider.overrideWithValue(_FakeLock())]);

  // ---- AppShell ----
  _scenario('AppShell: thanh dưới', () => const AppShell(), (t, _) async {
    expect(find.byType(AppShell), findsOneWidget);
  },
      overrides: () => [
            ..._health(_reminders, const []),
            lifestyleProvider
                .overrideWith((ref) async => const LifestyleSnapshot(tips: [])),
          ]);
}
