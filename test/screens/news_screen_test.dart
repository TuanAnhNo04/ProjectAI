// Day 31 — test cho màn Tin tức: trạng thái rỗng / lỗi / bài mẫu / tin thật.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/models/news_article.dart';
import 'package:image_processing_app/domain/ports/news_repository.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/news_screen.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';

class _FakeNewsRepository implements NewsRepository {
  _FakeNewsRepository({this.snapshot, this.fail = false});

  final NewsSnapshot? snapshot;
  final bool fail;

  @override
  Future<NewsSnapshot> fetchNews({bool refresh = false}) async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.network,
        message: 'Không kết nối được máy chủ.',
      );
    }
    return snapshot ?? const NewsSnapshot(articles: [], isSample: true);
  }
}

Widget _wrap(NewsRepository? repo, {Key? key}) => ProviderScope(
      key: key,
      overrides: [
        newsRepositoryProvider.overrideWithValue(repo),
        cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
      ],
      child: const MaterialApp(
        locale: Locale('vi'),
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: NewsScreen(),
      ),
    );

void main() {
  testWidgets('đang tải -> hiện vòng xoay', (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository()));
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
  });

  testWidgets(
      'backend không có endpoint tin tức (repo null) -> rỗng, không crash',
      (tester) async {
    await tester.pumpWidget(_wrap(null));
    await tester.pumpAndSettle();
    expect(find.text('Chưa có tin tức.'), findsOneWidget);
  });

  testWidgets('danh sách rỗng -> thông báo "Chưa có tin tức."', (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(
      snapshot: const NewsSnapshot(articles: [], isSample: false),
    )));
    await tester.pumpAndSettle();
    expect(find.text('Chưa có tin tức.'), findsOneWidget);
  });

  testWidgets('lỗi mạng -> hiện thông điệp lỗi + nút Thử lại', (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(fail: true)));
    await tester.pumpAndSettle();
    expect(find.text('Không kết nối được máy chủ.'), findsOneWidget);
    expect(find.widgetWithText(ElevatedButton, 'Thử lại'), findsOneWidget);
  });

  testWidgets('bài mẫu -> hiện banner "đang dùng nội dung mẫu"',
      (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(
      snapshot: const NewsSnapshot(
        articles: [
          NewsArticle(id: '1', title: 'Bài mẫu', source: 'MedAssist'),
        ],
        isSample: true,
      ),
    )));
    await tester.pumpAndSettle();
    expect(
      find.textContaining('nội dung mẫu'),
      findsOneWidget,
    );
    expect(find.text('Bài mẫu'), findsOneWidget);
  });

  testWidgets('tin thật (không phải mẫu) -> KHÔNG hiện banner nội dung mẫu',
      (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(
      snapshot: const NewsSnapshot(
        articles: [
          NewsArticle(
            id: '1',
            title: 'Panadol thu hồi',
            summary: 'Tóm tắt',
            url: 'https://bao.example.com/1',
            source: 'Báo Giả',
          ),
        ],
        isSample: false,
      ),
    )));
    await tester.pumpAndSettle();
    expect(find.textContaining('nội dung mẫu'), findsNothing);
    expect(find.text('Panadol thu hồi'), findsOneWidget);
  });

  testWidgets('nguồn tải hỏng -> hiện banner nói rõ tên nguồn', (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(
      snapshot: const NewsSnapshot(
        articles: [
          NewsArticle(
              id: '1', title: 'T', source: 'Báo Tốt', url: 'https://e.com'),
        ],
        isSample: false,
        failedSources: ['Báo Hỏng'],
      ),
    )));
    await tester.pumpAndSettle();
    expect(find.textContaining('Báo Hỏng'), findsOneWidget);
  });

  testWidgets('luôn hiện disclaimer y khoa', (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(
      snapshot: const NewsSnapshot(
        articles: [
          NewsArticle(id: '1', title: 'T', source: 'X', url: 'https://e.com'),
        ],
        isSample: false,
      ),
    )));
    await tester.pumpAndSettle();
    expect(find.byType(MedicalDisclaimer), findsOneWidget);
  });

  testWidgets('kéo để làm mới hoạt động ngay cả khi đang tải', (tester) async {
    await tester.pumpWidget(_wrap(_FakeNewsRepository(
      snapshot: const NewsSnapshot(articles: [], isSample: true),
    )));
    expect(find.byType(RefreshIndicator), findsOneWidget);
  });
}
