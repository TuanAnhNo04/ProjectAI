import 'package:flutter/material.dart';

import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'agent_chat_screen.dart';
import 'home_tab.dart';
import 'process_image_screen.dart';
import 'profile_tab.dart';
import 'reminder_screen.dart';

/// Khung chính sau đăng nhập: 4 tab + nút giữa "Quét thuốc" (giống bố cục *remedi*).
///
///   Trang chủ · Nhắc thuốc · [Quét] · Trợ lý · Hồ sơ
///
/// Các tab dùng [IndexedStack] nên **giữ nguyên trạng thái** khi chuyển tab (đang gõ dở
/// câu hỏi cho trợ lý rồi sang xem nhắc thuốc không bị mất). Mỗi tab là một `Scaffold`
/// riêng có AppBar của nó.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;

  void _select(int index) => setState(() => _index = index);

  void _openScan() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const ImageProcessingScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    // Bàn phím mở thì ẩn thanh dưới + nút giữa: tránh chiếm chỗ ô nhập và tránh bị đẩy
    // lên trên bàn phím.
    final keyboardOpen = MediaQuery.of(context).viewInsets.bottom > 0;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      // Chừa chỗ cho nút quét "nhô lên" khỏi thanh dưới, để không che ô nhập / mục cuối.
      body: Padding(
        padding: EdgeInsets.only(bottom: keyboardOpen ? 0 : 30),
        child: IndexedStack(
          index: _index,
          children: [
            HomeTab(
              onOpenReminders: () => _select(1),
              onOpenAssistant: () => _select(2),
            ),
            const ReminderScreen(),
            const AgentChatScreen(),
            const ProfileTab(),
          ],
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: keyboardOpen
          ? null
          : FloatingActionButton(
              onPressed: _openScan,
              tooltip: l10n.scanMedicine,
              shape: const CircleBorder(),
              child: const Icon(Icons.document_scanner_outlined),
            ),
      // ⚠️ Thanh dưới "nổi": tách khỏi mép màn hình bằng margin + bo tròn toàn bộ 4
      // góc + đổ bóng, thay cho BottomAppBar phẳng dính đáy trước đây — chỉ đổi
      // cách trình bày, các `_NavItem`/điều hướng bên trong giữ nguyên.
      bottomNavigationBar: keyboardOpen
          ? null
          : Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Container(
                height: 64,
                decoration: BoxDecoration(
                  color: NeuTheme.of(context).surface,
                  borderRadius: BorderRadius.circular(32),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.08),
                      offset: const Offset(0, 4),
                      blurRadius: 16,
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    _NavItem(
                      icon: Icons.home_outlined,
                      selectedIcon: Icons.home,
                      label: l10n.navHome,
                      selected: _index == 0,
                      onTap: () => _select(0),
                    ),
                    _NavItem(
                      icon: Icons.alarm_outlined,
                      selectedIcon: Icons.alarm,
                      label: l10n.navReminders,
                      selected: _index == 1,
                      onTap: () => _select(1),
                    ),
                    // Chừa chỗ cho nút giữa (nút quét nổi đè lên phần này).
                    const Expanded(
                        child: SizedBox(height: AppTheme.minTouchTarget)),
                    _NavItem(
                      icon: Icons.chat_bubble_outline,
                      selectedIcon: Icons.chat_bubble,
                      label: l10n.navAssistant,
                      selected: _index == 2,
                      onTap: () => _select(2),
                    ),
                    _NavItem(
                      icon: Icons.person_outline,
                      selectedIcon: Icons.person,
                      label: l10n.navProfile,
                      selected: _index == 3,
                      onTap: () => _select(3),
                    ),
                  ],
                ),
              ),
            ),
    );
  }
}

/// Một mục trên thanh dưới: icon + nhãn, đủ lớn để bấm (≥48dp), đọc được bằng TalkBack.
class _NavItem extends StatelessWidget {
  const _NavItem({
    required this.icon,
    required this.selectedIcon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final IconData icon;
  final IconData selectedIcon;
  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final color = selected ? scheme.primary : scheme.onSurfaceVariant;
    return Expanded(
      child: Semantics(
        button: true,
        selected: selected,
        label: label,
        excludeSemantics: true,
        child: InkWell(
          onTap: onTap,
          child: SizedBox(
            height: 64,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 3),
                  decoration: BoxDecoration(
                    color: selected
                        ? NeuTheme.of(context).accentSoft
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Icon(selected ? selectedIcon : icon, color: color),
                ),
                const SizedBox(height: 1),
                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.labelSmall?.copyWith(
                        color: color,
                        fontWeight:
                            selected ? FontWeight.w700 : FontWeight.w500,
                      ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
