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
from app.textnorm import normalize

#: Ma tu choi — on dinh de client xu ly, KHONG so sanh `message`.
REFUSAL_NO_EVIDENCE = "no_evidence"
REFUSAL_UNSUPPORTED = "unsupported_claim"
REFUSAL_OUT_OF_SCOPE = "out_of_scope"
REFUSAL_NO_SOURCE = "no_verified_source"

#: Cau hoi vuot pham vi ung dung (chan doan, don thuoc, khan cap...).
_OUT_OF_SCOPE_PATTERNS = (
    r"ch[uẩ]n\s*[đoâ]n\s*o|ch[uẩ]n\s*[đoâ]n\s*đoán|diagnos",
    r"k[eê]\s*[đd]\s*g[oô]|c[ấu]p\s*c[ưa]u|c[uấ]n g[ấp]",
    r"li[eê]u\s*(d[uù]ng|d[uõ]|c[oô])|mg/kg|mcg/kg|iu/ng[àa]y",
    r"t[uương]ng\s*t[aá]c|drug interaction",
    r"k[eê]\s*thu[oô]c\s*n[àa]y\s*c[oô]|k[eê]\s*n[àa]y\s*[\u0111u\u01b0]",
    r"bao\s*thai|ng[ọo] cho con b[úi]|s[ữu]a m[ẹ]",
    r"thay\s*thu[oô]c\s*b[ạa]n|ng[ưng]\s*thu[oô]c|t[ắa]t\s*d[ư]",
    r"bao nhi[eê]u|t[ươ]ng ph[ảa]n|s[ự] d[ụng] c[ư]",
)

_OUT_OF_SCOPE_RE = re.compile("|".join(_OUT_OF_SCOPE_PATTERNS), re.IGNORECASE)

#: Cau da trich dan trong cau tra loi: `- noi dung [R1]`
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]{1,32})\]")

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
        found = _CITATION_RE.findall(line)
        if not found:
            # Dòng không có trích dẫn -> mệnh lệnh không có căn cứ -> từ chối.
            return False, []

        cited = [i for i in found if i in valid_ids]
        if len(cited) != len(found):
            # Trích dẫn mã bịa -> nghi vấn toàn bộ câu trả lời.
            return False, []

        # Lấy phần nội dung bỏ mã trích dẫn ở cuối để so khớp.
        body = _CITATION_RE.sub("", line).lstrip("- ").strip()
        if not body:
            return False, []
        norm_body = normalize(body)

        matched = False
        for cite_id in cited:
            source = next(c for c in citations if c.id == cite_id)
            if _is_supported_by(norm_body, source):
                matched = True
                break
        if not matched:
            # Có mã trích dẫn nhưng nội dung không bám nguồn -> bịa.
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
    return bool(_OUT_OF_SCOPE_RE.search(question))


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
