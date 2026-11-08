import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/core/theme/app_theme.dart';
import 'package:image_processing_app/core/voice_providers.dart';
import 'package:image_processing_app/data/adapters/device_channel.dart';
import 'package:image_processing_app/data/models/chat_message.dart';
import 'package:image_processing_app/domain/ports/agent_client.dart';
import 'package:image_processing_app/domain/ports/voice_assistant.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/nearest_facility_screen.dart';
import 'package:image_processing_app/widgets/external_link.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';
import 'package:image_processing_app/widgets/neu.dart';

/// Màn hình hội thoại với agent (Day 25).
///
/// Tin nhắn hiện token-by-token từ SSE; citation bấm được mở bottom-sheet.
/// Là một tab của AppShell nhưng vẫn dùng được khi được push như route.
class AgentChatScreen extends ConsumerStatefulWidget {
  const AgentChatScreen({super.key});

  @override
  ConsumerState<AgentChatScreen> createState() => _AgentChatScreenState();
}

class _AgentChatScreenState extends ConsumerState<AgentChatScreen> {
  final _inputController = TextEditingController();
  final _scrollController = ScrollController();
  StreamSubscription<VoiceInputEvent>? _voiceSub;
  bool _listening = false;

  /// `true` = người dùng đã ẩn gợi ý bật AI đám mây (lưu local, không nhạy cảm).
  bool _hintDismissed = true;
  static const String _hintKey = 'chat.cloud_ai_hint_dismissed';

  @override
  void initState() {
    super.initState();
    _loadHint();
    // Nạp hồ sơ (consent) để biết trí nhớ trợ lý có được phép dùng không.
    Future.microtask(() {
      if (!mounted) return;
      if (ref.read(healthProfileControllerProvider).valueOrNull == null) {
        ref.read(healthProfileControllerProvider.notifier).load();
      }
    });
  }

  Future<void> _loadHint() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final dismissed = prefs.getBool(_hintKey) ?? false;
      if (mounted) setState(() => _hintDismissed = dismissed);
    } on Object {
      // Không đọc được -> giữ ẩn, không làm phiền.
    }
  }

  Future<void> _dismissHint() async {
    setState(() => _hintDismissed = true);
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_hintKey, true);
    } on Object {
      // Bỏ qua.
    }
  }

  @override
  void dispose() {
    _voiceSub?.cancel();
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _send() {
    final text = _inputController.text.trim();
    if (text.isEmpty) return;
    _inputController.clear();
    ref.read(chatControllerProvider.notifier).send(text);
  }

  /// Bật/tắt nghe. Transcript cuối được nối vào ô nhập, KHÔNG tự gửi, không log.
  void _toggleListening() {
    final voice = ref.read(voiceInputProvider);
    if (_listening) {
      _voiceSub?.cancel();
      _voiceSub = null;
      voice.stop();
      setState(() => _listening = false);
      return;
    }
    setState(() => _listening = true);
    _voiceSub = voice.listen().listen(
      (event) {
        if (event is VoiceFinal) {
          final current = _inputController.text;
          final sep = current.isEmpty || current.endsWith(' ') ? '' : ' ';
          final next = '$current$sep${event.text}';
          _inputController.value = TextEditingValue(
            text: next,
            selection: TextSelection.collapsed(offset: next.length),
          );
        }
      },
      onError: (Object _) {
        if (mounted) setState(() => _listening = false);
      },
      onDone: () {
        if (mounted) setState(() => _listening = false);
      },
    );
  }

  void _stop() {
    ref.read(chatControllerProvider.notifier).stop();
  }

  void _scrollToBottom() {
    if (!_scrollController.hasClients) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final chatState = ref.watch(chatControllerProvider);
    final messages = chatState.valueOrNull ?? const <ChatMessage>[];
    final isStreaming = messages.isNotEmpty && messages.last.isStreaming;

    ref.listen(chatControllerProvider, (prev, next) {
      _scrollToBottom();
    });

    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final voiceAvailable = ref.watch(voiceInputProvider).isAvailable;
    final cloudState = ref.watch(cloudAiConsentControllerProvider);

    return Scaffold(
      backgroundColor: neu.background,
      appBar: AppBar(
        backgroundColor: neu.background,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              l10n.chatTitle,
              style: TextStyle(
                color: neu.textStrong,
                fontWeight: FontWeight.w800,
              ),
            ),
            // Nhãn AI rõ ràng ngay từ đầu để người dùng không nhầm là người thật.
            Text(
              l10n.chatAiLabel,
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    color: neu.textMuted,
                  ),
            ),
          ],
        ),
        actions: [
          if (messages.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: NeuCircleButton(
                icon: Icons.delete_outline,
                tooltip: l10n.chatClear,
                onPressed: () => _confirmClear(context, l10n),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            if (!_hintDismissed &&
                !cloudState.isLoading &&
                !cloudState.value.granted)
              _CloudAiHint(
                text: l10n.cloudAiChatHint,
                dismissTooltip: l10n.cloudAiHintDismiss,
                onDismiss: _dismissHint,
              ),
            Expanded(
              child: messages.isEmpty
                  ? _EmptyState(l10n: l10n)
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      itemCount: messages.length,
                      itemBuilder: (context, i) =>
                          _MessageBubble(message: messages[i]),
                    ),
            ),
            const MedicalDisclaimer(compact: true),
            _InputBar(
              controller: _inputController,
              isStreaming: isStreaming,
              onSend: _send,
              onStop: _stop,
              l10n: l10n,
              voiceAvailable: voiceAvailable,
              listening: _listening,
              onMic: _toggleListening,
            ),
          ],
        ),
      ),
    );
  }

  void _confirmClear(BuildContext context, AppLocalizations l10n) {
    showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l10n.chatClearTitle),
        content: Text(l10n.chatClearBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l10n.cancel),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(l10n.confirm),
          ),
        ],
      ),
    ).then((confirmed) {
      if (confirmed == true) {
        ref.read(chatControllerProvider.notifier).clear();
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.l10n});

  final AppLocalizations l10n;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.chat_bubble_outline, size: 64, color: neu.accent),
            const SizedBox(height: 16),
            Text(
              l10n.chatEmpty,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: neu.textMuted,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

class _MessageBubble extends ConsumerWidget {
  const _MessageBubble({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final isUser = message.isUser;
    final emergency = message.emergency;
    // Server gửi lại đúng lời khuyên ở token kế tiếp — đã nằm trong thẻ đỏ.
    final hideText =
        emergency != null && message.text.trim() == emergency.message.trim();
    final scheme = theme.colorScheme;
    final neu = NeuTheme.of(context);

    final textColor = isUser ? scheme.onPrimary : neu.textStrong;
    final align = isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start;
    final maxWidth = MediaQuery.sizeOf(context).width * 0.85;

    final content = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (emergency != null) _EmergencyCard(alert: emergency),
        if (message.toolName != null) _ToolChip(toolName: message.toolName!),
        if (message.text.isNotEmpty && !hideText)
          SelectableText(
            message.text,
            style: theme.textTheme.bodyMedium?.copyWith(color: textColor),
          ),
        if (message.isStreaming && message.text.isEmpty)
          _TypingIndicator(color: textColor),
        if (message.isError && message.errorText != null)
          Padding(
            padding: const EdgeInsets.only(top: 6),
            child: Text(
              message.errorText!,
              style: theme.textTheme.bodySmall?.copyWith(color: scheme.error),
            ),
          ),
      ],
    );

    const bubblePadding = EdgeInsets.symmetric(horizontal: 16, vertical: 12);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: align,
        children: [
          ConstrainedBox(
            constraints: BoxConstraints(maxWidth: maxWidth),
            child: isUser
                ? Container(
                    padding: bubblePadding,
                    decoration: BoxDecoration(
                      color: neu.accent,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: content,
                  )
                : NeuCard(padding: bubblePadding, child: content),
          ),
          if (!isUser &&
              message.mode != null &&
              !message.isStreaming &&
              !message.isError &&
              emergency == null &&
              message.citations.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: AnswerModeChip(isCloud: message.mode == 'cloud'),
            ),
          if (!isUser && message.symptom != null && !message.isStreaming)
            _SymptomNote(message: message),
          if (message.citations.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Wrap(
                spacing: 8,
                runSpacing: 4,
                children: message.citations
                    .map((c) => _CitationChip(citation: c))
                    .toList(),
              ),
            ),
        ],
      ),
    );
  }
}

/// Dưới câu trả lời: dòng mờ "Đã ghi nhớ trên máy: … · Xoá" (khi đã lưu vào két)
/// và ghi chú hổ phách "Triệu chứng lặp lại" (khi server báo lặp lại).
class _SymptomNote extends ConsumerWidget {
  const _SymptomNote({required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    final symptom = message.symptom!;
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (symptom.recurring)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFFFF3CD),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.repeat, size: 16, color: Color(0xFF7A5200)),
                  const SizedBox(width: 6),
                  Flexible(
                    child: Text(
                      l10n.memHRecurring,
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: const Color(0xFF7A5200)),
                    ),
                  ),
                ],
              ),
            ),
          if (message.savedEpisodeAt != null)
            Wrap(
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                Text(
                  '${l10n.memHChatSaved(symptom.terms.join(', '))} · ',
                  style:
                      theme.textTheme.bodySmall?.copyWith(color: neu.textMuted),
                ),
                InkWell(
                  onTap: () => ref
                      .read(chatControllerProvider.notifier)
                      .forgetSymptom(message),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Text(
                      l10n.memHChatForget,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: neu.accent,
                        decoration: TextDecoration.underline,
                      ),
                    ),
                  ),
                ),
              ],
            ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Emergency card + cloud AI hint
// ---------------------------------------------------------------------------

class _EmergencyCard extends ConsumerWidget {
  const _EmergencyCard({required this.alert});

  final EmergencyAlert alert;

  Future<void> _call(BuildContext context, WidgetRef ref) async {
    final messenger = ScaffoldMessenger.of(context);
    final failed = AppLocalizations.of(context).emergencyDialFailed;
    try {
      await ref.read(deviceChannelProvider).dial('115');
    } on DeviceChannelException {
      messenger.showSnackBar(SnackBar(content: Text(failed)));
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context);
    const red = Color(0xFFC62828);
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: red.withOpacity(0.10),
        border: Border.all(color: red, width: 1.5),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.warning_amber_rounded, color: red, size: 28),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  alert.message,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: red,
                        fontWeight: FontWeight.w700,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            style: FilledButton.styleFrom(
              backgroundColor: red,
              foregroundColor: Colors.white,
              minimumSize: const Size.fromHeight(52),
              textStyle:
                  const TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
            ),
            onPressed: () => _call(context, ref),
            icon: const Icon(Icons.call),
            label: Text(l10n.emergencyCall),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              foregroundColor: red,
              side: const BorderSide(color: red),
              minimumSize: const Size.fromHeight(44),
            ),
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) =>
                    NearestFacilityScreen(emergencyMessage: alert.message),
              ),
            ),
            icon: const Icon(Icons.local_hospital),
            label: Text(l10n.emergencyNearest),
          ),
        ],
      ),
    );
  }
}

class _CloudAiHint extends StatelessWidget {
  const _CloudAiHint({
    required this.text,
    required this.dismissTooltip,
    required this.onDismiss,
  });

  final String text;
  final String dismissTooltip;
  final VoidCallback onDismiss;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 4, 16, 0),
      padding: const EdgeInsets.only(left: 12),
      decoration: BoxDecoration(
        color: neu.accentSoft,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(Icons.cloud_outlined, size: 18, color: neu.accent),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: neu.textStrong),
            ),
          ),
          IconButton(
            tooltip: dismissTooltip,
            icon: const Icon(Icons.close, size: 18),
            onPressed: onDismiss,
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Citation chip
// ---------------------------------------------------------------------------

class _CitationChip extends StatelessWidget {
  const _CitationChip({required this.citation});

  final Citation citation;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    // Mã nguồn đứng trước tên để các chip cùng thuốc vẫn phân biệt được (R1 · Panadol…).
    final title = citation.title;
    final label = (title == null || title.isEmpty)
        ? citation.documentId
        : '${citation.documentId} · $title';
    return Semantics(
      button: true,
      label: label,
      excludeSemantics: true,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: neu.accentSoft,
          borderRadius: BorderRadius.circular(24),
          boxShadow: neu.soft(),
        ),
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            borderRadius: BorderRadius.circular(24),
            onTap: () => _showCitationSheet(context),
            child: ConstrainedBox(
              constraints: const BoxConstraints(minHeight: 48),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 14),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.menu_book, size: 16, color: neu.accent),
                    const SizedBox(width: 6),
                    Flexible(
                      child: Text(
                        label,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: neu.textStrong,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  void _showCitationSheet(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final theme = Theme.of(context);
    final neu = NeuTheme.of(context);
    showModalBottomSheet(
      context: context,
      backgroundColor: neu.background,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (_) => SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                citation.title ?? citation.documentId,
                style: theme.textTheme.titleMedium?.copyWith(
                  color: neu.textStrong,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                citation.quote,
                style:
                    theme.textTheme.bodyMedium?.copyWith(color: neu.textStrong),
              ),
              const SizedBox(height: 12),
              SourceBlock(
                source: citation.source,
                isAuthoritative:
                    citation.source.isEmpty ? null : citation.isAuthoritative,
                retrievedAt: citation.retrievedAt,
                license: citation.license,
                reviewedBy: citation.reviewedBy,
                sourceUrl: citation.sourceUrl,
              ),
              const SizedBox(height: 16),
              Text(
                l10n.medicalDisclaimer,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: neu.textMuted,
                ),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Tool running chip
// ---------------------------------------------------------------------------

class _ToolChip extends StatelessWidget {
  const _ToolChip({required this.toolName});

  final String toolName;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: neu.accent,
            ),
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              l10n.chatToolRunning(toolName),
              style: TextStyle(
                fontSize: 12,
                color: neu.textMuted,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

class _TypingIndicator extends StatelessWidget {
  const _TypingIndicator({required this.color});

  final Color color;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(
        3,
        (i) => Padding(
          padding: EdgeInsets.only(left: i > 0 ? 4 : 0),
          child: _Dot(color: color, delay: i * 150),
        ),
      ),
    );
  }
}

class _Dot extends StatefulWidget {
  const _Dot({required this.color, required this.delay});

  final Color color;
  final int delay;

  @override
  State<_Dot> createState() => _DotState();
}

class _DotState extends State<_Dot> with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _anim = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
    Future.delayed(Duration(milliseconds: widget.delay), () {
      if (mounted) _ctrl.repeat(reverse: true);
    });
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Opacity(
        opacity: 0.3 + _anim.value * 0.7,
        child: Container(
          width: 7,
          height: 7,
          decoration: BoxDecoration(
            color: widget.color,
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Input bar
// ---------------------------------------------------------------------------

class _InputBar extends StatelessWidget {
  const _InputBar({
    required this.controller,
    required this.isStreaming,
    required this.onSend,
    required this.onStop,
    required this.l10n,
    required this.voiceAvailable,
    required this.listening,
    required this.onMic,
  });

  final TextEditingController controller;
  final bool isStreaming;
  final VoidCallback onSend;
  final VoidCallback onStop;
  final AppLocalizations l10n;
  final bool voiceAvailable;
  final bool listening;
  final VoidCallback onMic;

  @override
  Widget build(BuildContext context) {
    final neu = NeuTheme.of(context);
    OutlineInputBorder border([Color? color]) => OutlineInputBorder(
          borderRadius: BorderRadius.circular(24),
          borderSide: color == null
              ? BorderSide.none
              : BorderSide(color: color, width: 1.5),
        );
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (listening)
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                children: [
                  Icon(Icons.graphic_eq, size: 18, color: neu.accent),
                  const SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      l10n.voiceListening,
                      style: TextStyle(
                        color: neu.textMuted,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: controller,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => onSend(),
                  enabled: !isStreaming,
                  maxLines: 3,
                  minLines: 1,
                  style: TextStyle(color: neu.textStrong),
                  decoration: InputDecoration(
                    hintText: l10n.chatHint,
                    hintStyle: TextStyle(color: neu.textMuted),
                    filled: true,
                    fillColor: neu.accentSoft.withOpacity(0.5),
                    border: border(),
                    enabledBorder: border(),
                    disabledBorder: border(),
                    focusedBorder: border(neu.accent),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 18,
                      vertical: 14,
                    ),
                  ),
                ),
              ),
              if (voiceAvailable && !isStreaming) ...[
                const SizedBox(width: 8),
                NeuCircleButton(
                  icon: listening ? Icons.mic : Icons.mic_none,
                  tooltip: l10n.voiceMicTooltip,
                  onPressed: onMic,
                  filled: listening,
                ),
              ],
              const SizedBox(width: 8),
              isStreaming
                  ? NeuCircleButton(
                      icon: Icons.stop,
                      tooltip: l10n.chatStop,
                      onPressed: onStop,
                      filled: true,
                    )
                  : NeuCircleButton(
                      icon: Icons.send,
                      tooltip: l10n.chatSend,
                      onPressed: onSend,
                      filled: true,
                    ),
            ],
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Answer mode chip + comparison sheet
// ---------------------------------------------------------------------------

/// Chip nho duoi cau tra loi: trich nguyen van hay Claude dien dat lai.
class AnswerModeChip extends StatelessWidget {
  const AnswerModeChip({super.key, required this.isCloud});

  final bool isCloud;

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context);
    final neu = NeuTheme.of(context);
    final theme = Theme.of(context);
    final label = isCloud ? l10n.modeGChipCloud : l10n.modeGChipExtractive;
    return Semantics(
      button: true,
      label: label,
      child: InkWell(
        borderRadius: BorderRadius.circular(999),
        onTap: () => showAnswerModeSheet(context),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: neu.accentSoft,
            borderRadius: BorderRadius.circular(999),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(isCloud ? Icons.auto_awesome : Icons.menu_book_outlined,
                  size: 16, color: neu.accent),
              const SizedBox(width: 6),
              Flexible(
                child: Text(
                  label,
                  style: theme.textTheme.labelMedium?.copyWith(
                      color: neu.accent, fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

void showAnswerModeSheet(BuildContext context) {
  showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
    ),
    builder: (ctx) {
      final l10n = AppLocalizations.of(ctx);
      final neu = NeuTheme.of(ctx);
      final theme = Theme.of(ctx);
      Widget head(String t) => Padding(
            padding: const EdgeInsets.only(top: 14, bottom: 4),
            child: Text(t,
                style: theme.textTheme.titleSmall?.copyWith(
                    color: neu.textStrong, fontWeight: FontWeight.w700)),
          );
      Widget line(IconData icon, String label, String body) => Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(icon, size: 18, color: neu.accent),
                const SizedBox(width: 8),
                Expanded(
                  child: Text.rich(TextSpan(children: [
                    TextSpan(
                        text: '$label: ',
                        style: const TextStyle(fontWeight: FontWeight.w700)),
                    TextSpan(text: body),
                  ])),
                ),
              ],
            ),
          );
      return SafeArea(
        child: ConstrainedBox(
          constraints:
              BoxConstraints(maxHeight: MediaQuery.of(ctx).size.height * 0.85),
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(l10n.modeGSheetTitle,
                    style: theme.textTheme.titleLarge?.copyWith(
                        color: neu.textStrong, fontWeight: FontWeight.w700)),
                head(l10n.modeGRowData),
                line(
                    Icons.menu_book_outlined, l10n.modeGOff, l10n.modeGOffData),
                line(Icons.auto_awesome, l10n.modeGOn, l10n.modeGOnData),
                head(l10n.modeGRowHow),
                line(Icons.menu_book_outlined, l10n.modeGOff, l10n.modeGOffHow),
                line(Icons.auto_awesome, l10n.modeGOn, l10n.modeGOnHow),
                head(l10n.modeGRowSame),
                Text(l10n.modeGSameBody, style: theme.textTheme.bodyMedium),
                const SizedBox(height: 14),
                Text(l10n.modeGWhere,
                    style: theme.textTheme.bodyMedium?.copyWith(
                        color: neu.accent, fontWeight: FontWeight.w600)),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: () => Navigator.of(ctx).pop(),
                    child: Text(l10n.modeGClose),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    },
  );
}
