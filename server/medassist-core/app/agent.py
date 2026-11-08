"""Agent trả lời câu hỏi về thuốc — Day 21.

## Luồng: KHÔNG để LLM tự quyết có nên trả lời

```
cau hoi
  -> loc chu de ngoai pham vi      -> tu choi (out_of_scope)
  -> phat hien prompt injection    -> tu choi (out_of_scope)
  -> truy hoi can cu              -> rong    -> tu choi (no_evidence)
  -> dung prompt (nguon = du lieu) -> app/prompt.py
  -> provider sinh                -> app/llm.py (mac dinh MOCK, offline)
  -> KIEM CHUNG GROUNDING          -> tu choi (unsupported_claim)
  -> tra loi kem trich dan
```

## Ba lop phong dua (khong dua vao mot lap nao)

1. **Truoc khi goi LLM**: loc chu de + phat hien injection. Re, chan som.
2. **Trong prompt**: nguon duoc escape va boc ky tu phan tach (`app/prompt.py`).
3. **Sau khi LLM sinh**: `verify_grounding()` — moi cau trong cau tra loi phai bam
   duoc ve mot doan da trich dan. ⚠️ **Day la lap quan trong nhat.** Lop 1 va 2 chi
   giam be mat tan cong; neu sau nay thay provider gia bang LLM that, mot mo hinh bi
   "dat mui" van co the viet ra thu khong co trong nguon — va lop 3 chan duoc.

⚠️ `AGENTS.md` mục 5: noi dung tu file/web/CSV la **DU LIEU**, khong phai lenh.
Cau hoi nguoi dung cung la du lieu, nen no cung duoc escape nhu nhau.

## Du lieu dev KHONG authoritative — he qua bat buoc

Dữ liệu hiện tại đều `is_authoritative=False`. Vì vậy:
- câu hỏi **tra cuu thong tin** (ten/hoat chat/cong dung) -> tra loi kem canh bao;
- câu hỏi **y khoa** (lieu dung, tuong tac, chan doan...) -> **luon tu choi**, ke ca
  khi tim thay du lieu khop, vi du lieu mau khong du de ket luan y khoa.
"""

from __future__ import annotations

import logging
import re

from app.llm import LLMProvider, get_llm_provider
from app.prompt import build_messages, looks_like_injection
from app.retrieval import build_citations, has_authoritative_evidence
from app.schemas import (
    AgentAnswerResponse,
    CitationResponse,
    LLMResponse,
    MedicineResponse,
)
from app.smalltalk import smalltalk_reply
from app.textnorm import normalize

#: Ma tu choi — on dinh de client xu ly, KHONG so sanh `message`.
REFUSAL_NO_EVIDENCE = "no_evidence"
REFUSAL_UNSUPPORTED = "unsupported_claim"
REFUSAL_OUT_OF_SCOPE = "out_of_scope"
REFUSAL_NO_SOURCE = "no_verified_source"

#: Cau hoi vuot pham vi ung dung (chan doan, lieu dung, khan cap...).
#:
#: ⭐ **Mọi mẫu viết trên chuỗi ĐÃ `normalize()`** (không dấu, chữ thường) — cùng
#: nguyên tắc `app/textnorm.py`: chuẩn hoá trước, khớp sau. Bản cũ viết regex có dấu
#: với character class (`[uương]`, `[eê]`) — một class chỉ khớp MỘT ký tự nên "tương",
#: "liều", "chẩn đoán" lọt qua (eval Day 24 phát hiện 15 lỗ).
_DIAGNOSIS_PATTERNS = (
    r"chan doan",
    r"diagnos",
    r"ke don",
    r"ke thuoc cho toi",
    r"cap cuu",
    r"emergency",
    r"ngo doc",
    r"overdose",
    r"qua lieu",
)

_DOSAGE_PATTERNS = (
    r"lieu dung",
    r"lieu luong",
    r"lieu cao",
    r"lieu thap",
    r"lieu toi da",
    r"lieu cho (tre|nguoi)",
    r"dose",
    r"dosage",
    r"mg ?/ ?kg",
    r"mcg ?/ ?kg",
    r"iu ?/ ?ngay",
    r"uong bao nhieu",
    r"bao nhieu (vien|mg|ml|goi)",
    r"uong .*bao nhieu",
    r"bao nhieu .*(ngay|lan|gio)",
    r"may vien",
)

#: Tách riêng vì `/v1/agent/chat` định tuyến nhóm này sang tool `check_interaction`
#: (tool đó tự từ chối khi không có dữ liệu kiểm chứng — xem `app/tools.py`).
_INTERACTION_SCOPE_PATTERNS = (
    r"tuong tac",
    r"tuong phan",
    r"drug interaction",
    r"interaction",
)

_SPECIAL_POPULATION_PATTERNS = (
    r"mang thai",
    r"co thai",
    r"dang bau",
    r"co bau",
    r"pregnan",
    r"cho con bu",
    r"sua me",
    r"breastfeed",
)

_STOP_TREATMENT_PATTERNS = (
    r"ngung thuoc",
    r"bo thuoc",
    r"tu y (ngung|bo|doi)",
    r"doi thuoc",
    r"thay thuoc cua toi",
)


def _compile(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(patterns))


_BASE_SCOPE_RE = _compile(
    _DIAGNOSIS_PATTERNS
    + _DOSAGE_PATTERNS
    + _SPECIAL_POPULATION_PATTERNS
    + _STOP_TREATMENT_PATTERNS
)
_INTERACTION_SCOPE_RE = _compile(_INTERACTION_SCOPE_PATTERNS)


def is_out_of_scope(question: str, *, allow_interaction: bool = False) -> bool:
    """Câu hỏi có vượt phạm vi an toàn của ứng dụng không?

    Args:
        question: câu gốc của người dùng (có dấu hay không đều được).
        allow_interaction: `True` ở luồng chat — câu hỏi tương tác thuốc được đi tới
            tool `check_interaction` thay vì bị chặn sớm. Các nhóm khác (chẩn đoán,
            liều dùng, thai kỳ, ngưng thuốc) **luôn** bị chặn.
    """
    flat = normalize(question)
    if _BASE_SCOPE_RE.search(flat):
        return True
    return not allow_interaction and bool(_INTERACTION_SCOPE_RE.search(flat))

#: Cau da trich dan trong cau tra loi: `- noi dung [R1]`
_CITATION_RE = re.compile(
    r"\[([A-Za-z0-9_.-]{1,32}(?:\s*,\s*[A-Za-z0-9_.-]{1,32})*)\]"
)
_MD_NOISE_RE = re.compile(r"[*_`#>]+")
#: Dòng KHÔNG có mã trích dẫn nhưng không chứa khẳng định y khoa: lời dẫn kết thúc
#: bằng ":" hoặc lời khuyên hỏi dược sĩ/bác sĩ. Cấm có chữ số (không thêm liều/số liệu).
_NO_CLAIM_RE = re.compile(r"(duoc si|bac si|chuyen gia y te|nha thuoc)")

DISCLAIMER = (
    "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn của dược sĩ hoặc "
    "bác sĩ. Nếu có triệu chứng nặng hoặc bất thường, hãy đi khám bác sĩ."
)

DEV_DATA_WARNING = (
    "Nguồn hiện tại là DỮ LIỆU MẪU dùng để phát triển, chưa được kiểm chứng lâm "
    "sàng — không dùng để tự chẩn đoán hay điều chỉnh liều thuốc."
)

NO_ANSWER_MESSAGES = {
    REFUSAL_NO_EVIDENCE: (
        "Chưa tìm thấy thông tin trong nguồn dữ liệu hiện có cho câu hỏi này. "
        "Hãy hỏi dược sĩ để được tư vấn chính xác."
    ),
    REFUSAL_UNSUPPORTED: (
        "Câu trả lời không có căn cứ trong nguồn tham chiếu nên không được hiển thị. "
        "Hãy hỏi dược sĩ để được tư vấn chính xác."
    ),
    REFUSAL_OUT_OF_SCOPE: (
        "Ứng dụng không tư vấn chẩn đoán, liều dùng hay tương tác thuốc. "
        "Các câu hỏi này cần bác sĩ hoặc dược sĩ trực tiếp."
    ),
    REFUSAL_NO_SOURCE: (
        "Dữ liệu hiện có là dữ liệu mẫu, chưa phải nguồn lâm sàng được kiểm chứng, "
        "nên không đủ cơ sở để kết luận. Hãy tham khảo nhà sản xuất hoặc dược sĩ."
    ),
}


def _refuse(reason: str, provider: str = "mock") -> AgentAnswerResponse:
    """Từ chối trả lời.

    ⚠️ `text=None` + `answered=False` — client **không được** hiện bất cứ câu nào.
    `citations` rỗng vì trích dẫn nào cũng không được dùng khi đã từ chối.
    """
    return AgentAnswerResponse(
        answered=False,
        text=None,
        refusal_reason=reason,
        citations=[],
        disclaimer=DISCLAIMER,
        provider=provider,
    )


def _log_reject(reason: str, **counts: int | bool) -> None:
    """Chỉ ghi MÃ lý do + số đếm; không bao giờ ghi nội dung câu trả lời."""
    logging.getLogger("medassist").warning("grounding reject: %s %s", reason, counts)


def verify_grounding(
    text: str, citations: list[CitationResponse]
) -> tuple[bool, list[str]]:
    """Kiểm tra mọi câu trả lời đều bám được về nguồn đã trích dẫn.

    Quy tắc (cố ý **chặt**, thiên về từ chối hơn trả lời sai):
    1. Phải có ít nhất một câu có mã trích dẫn.
    2. Mọi mã trích dẫn phải **tồn tại** trong danh sách nguồn đã truy hồi.
    3. Nội dung mỗi dòng phải khớp (sau khi chuẩn hoá) với `excerpt` của ít nhất
       một nguồn được trích dẫn ở chính dòng đó.

    Trả `(đạt, danh sách mã đã dùng)`. Đạt -> agent được phép trả lời.
    """
    valid_ids = {c.id for c in citations}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, []

    used: list[str] = []
    for line in lines:
        # Markdown (**đậm**, `mã`, # tiêu đề) là trình bày, không phải nội dung.
        line = _MD_NOISE_RE.sub("", line).strip()
        if not line:
            continue
        found = [i.strip() for grp in _CITATION_RE.findall(line) for i in grp.split(",")]
        if not found:
            flat = normalize(line)
            no_digit = not any(ch.isdigit() for ch in flat)
            if no_digit and (line.endswith(":") or _NO_CLAIM_RE.search(flat)):
                continue  # lời dẫn / lời khuyên hỏi chuyên gia, không có khẳng định
            # Dòng không có trích dẫn -> mệnh lệnh không có căn cứ -> từ chối.
            _log_reject("no_citation", n_words=len(flat.split()))
            return False, []

        cited = [i for i in found if i in valid_ids]
        if len(cited) != len(found):
            # Trích dẫn mã bịa -> nghi vấn toàn bộ câu trả lời.
            _log_reject("unknown_citation")
            return False, []

        # Lấy phần nội dung bỏ mã trích dẫn ở cuối để so khớp.
        body = _CITATION_RE.sub("", line).lstrip("-•* ").strip()
        if not body:
            _log_reject("empty_body")
            return False, []
        norm_body = normalize(body)

        matched = False
        for cite_id in cited:
            source = next(c for c in citations if c.id == cite_id)
            if _is_supported_by(norm_body, source) or _quote_with_gloss_ok(
                norm_body, source
            ):
                matched = True
                break
        if not matched:
            # Có mã trích dẫn nhưng nội dung không bám nguồn -> bịa.
            _log_reject(
                "not_supported", n_words=len(norm_body.split()), n_cites=len(cited),
                quote_in_line=any(
                    normalize(c.excerpt) in norm_body for c in citations if c.id in cited
                ),
                gloss_words=_gloss_stats(norm_body, citations, cited)[0],
                new_numbers=_gloss_stats(norm_body, citations, cited)[1],
            )
            return False, []
        used.extend(cited)

    return bool(used), sorted(set(used))


def _is_out_of_scope(question: str) -> bool:
    """Câu hỏi có vượt phạm vi an toàn không?

    ⚠️ Ta **từ chối trước khi** truy hồi, không phải sau. Vì sao: câu hỏi về liều
    dùng/tương tác thuốc là câu hỏi mà người dùng mong nhận câu trả lời; nếu tìm
    được dữ liệu khớp rồi mới từ chối thì dễ khiến người dùng tin rằng hệ thống
    *biết* nhưng không muốn nói. Từ chối sớm nói rõ: vấn đề là ngoài phạm vi.
    """
    return is_out_of_scope(question)


def answer_question(
    question: str,
    medicines: list[MedicineResponse],
    top_k: int = 5,
    provider: LLMProvider | None = None,
) -> AgentAnswerResponse:
    """Trả lời một câu hỏi về thuốc.

    Args:
        question: câu hỏi của người dùng (đã bị giới hạn độ dài ở schema).
        medicines: danh mục thuốc đã nạp — **nguồn dữ liệu duy nhất** được phép
            dùng. Không có nguồn ngoài thì phải từ chối, không suy đoán.
        top_k: số đoạn nguồn tối đa đưa vào prompt.
        provider: provider LLM; mặc định lấy provider giả (offline).

    ⚠️ Hàm này **không** ghi log nội dung câu hỏi/nguồn (`AGENTS.md` mục 3, D4).
    """
    llm = provider or get_llm_provider()

    # Xã giao thuần: trả lời theo luật, không LLM/trích dẫn/từ chối.
    chit = smalltalk_reply(question)
    if chit is not None:
        return AgentAnswerResponse(
            answered=True, text=chit, refusal_reason=None, citations=[],
            disclaimer=DISCLAIMER, provider="extractive",
        )

    # Lớp 1: chủ đề ngoài phạm vi + prompt injection -> từ chối ngay.
    if _is_out_of_scope(question) or looks_like_injection(question):
        return _refuse(REFUSAL_OUT_OF_SCOPE, provider=llm.name)

    # Truy hồi căn cứ.
    retrieved = build_citations(medicines, question, top_k=top_k)
    if not retrieved:
        return _refuse(REFUSAL_NO_EVIDENCE, provider=llm.name)

    citations = [r.citation for r in retrieved]

    messages = build_messages(question, citations)
    result: LLMResponse = llm.generate(messages)

    if result.refused or not result.text:
        return _refuse(REFUSAL_NO_EVIDENCE, provider=llm.name)

    # Lớp 3: KIỂM CHỨNG GROUNDING sau khi sinh.
    ok, used_ids = verify_grounding(result.text, citations)
    if not ok:
        return _refuse(REFUSAL_UNSUPPORTED, provider=llm.name)

    # Chỉ giữ lại trích dẫn thực sự dùng -> app không hiện nguồn thừa.
    used = [c for c in citations if c.id in used_ids]

    # ⚠️ Nguồn dev KHÔNG authoritative: vẫn mô tả được (đã có căn cứ + grounding
    # đạt) nhưng **phải** gắn cảnh báo nổi bật, không phải disclaimer chung chung.
    if not has_authoritative_evidence(retrieved):
        return AgentAnswerResponse(
            answered=True,
            text=result.text,
            refusal_reason=None,
            citations=used,
            disclaimer=(
                f"⚠️ {DEV_DATA_WARNING} " + DISCLAIMER
            ),
            provider=llm.name,
        )

    return AgentAnswerResponse(
        answered=True,
        text=result.text,
        refusal_reason=None,
        citations=used,
        disclaimer=DISCLAIMER,
        provider=llm.name,
    )


#: Quy tắc "trích nguyên văn + giải thích ngắn" (chủ dự án duyệt 30/10/2026): khi nguồn là
#: tiếng Anh mà câu trả lời tiếng Việt, ta không so khớp chữ được. Thay vào đó dòng phải
#: CHỨA NGUYÊN VĂN đoạn trích (người dùng tự đối chiếu được), phần còn lại chỉ là lời giải
#: thích ngắn và không được có con số mới.
_MAX_GLOSS_WORDS = 14
_MIN_QUOTE_CHARS = 4


def _gloss_stats(norm_body: str, citations, cited) -> tuple[int, int]:
    """(số từ ngoài đoạn trích, số con số mới) của nguồn khớp nhất — chỉ để chẩn đoán."""
    best = (-1, -1)
    for c in citations:
        quote = normalize(c.excerpt)
        if c.id in cited and quote and quote in norm_body:
            rest = norm_body.replace(quote, " ", 1).split()
            allowed = set(re.findall(r"\d+", quote + " " + normalize(c.medicine_name)))
            new = sum(1 for w in rest for n in re.findall(r"\d+", w) if n not in allowed)
            best = (len(rest), new)
    return best


def _quote_with_gloss_ok(norm_body: str, citation: CitationResponse) -> bool:
    quote = normalize(citation.excerpt)
    if len(quote) < _MIN_QUOTE_CHARS or quote not in norm_body:
        return False
    rest = norm_body.replace(quote, " ", 1).split()
    if len(rest) > _MAX_GLOSS_WORDS:
        return False
    allowed_numbers = set(re.findall(r"\d+", quote + " " + normalize(citation.medicine_name)))
    return all(n in allowed_numbers for w in rest for n in re.findall(r"\d+", w))


def _is_supported_by(norm_body: str, citation: CitationResponse) -> bool:
    """Nội dung dòng có thực sự lấy từ đoạn trích dẫn không?

    So khớp theo **từ**: ≥ 60% từ (tối thiểu 2 từ) của dòng phải xuất hiện trong
    `excerpt` sau khi chuẩn hoá. Cụt mà không có căn cứ thì coi là bịa.
    """
    norm_src = normalize(citation.excerpt)
    if not norm_src:
        return False
    words = norm_body.split()
    if not words:
        return False
    hits = sum(1 for w in words if w in norm_src)
    return hits >= max(2, int(len(words) * 0.6))
