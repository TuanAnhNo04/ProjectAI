"""Truy hồi ngữ cảnh (retrieval) cho RAG — Day 20.

## Vì sao chấm điểm bằng từ khoá chứ không dùng embedding?

Máy dev **không có Docker** (xem `README.md`), nên không chạy được pgvector, và cấm
cài package mới (`docs/DECISIONS.md` D2). Ta dùng **BM25 đơn giản hoá** — thuần
Python, không phụ thuộc.

Đây là lựa chọn **tạm**, được ghi rõ trong `docs/DECISIONS.md`: khi có
Postgres/pgvector, chỉ cần viết thêm một hàm `retrieve()` cùng chữ ký, route không
đổi (giống cách `Repository` đã giữ ở Day 15).

## Nguyên tắc bất di bất dịch

⚠️ **Không đoạn truy hồi được nào mà thiếu `source`** — `AGENTS.md` mục 3 cấm hiển
thị thông tin không truy được nguồn. `Citation` nên `source` là bắt buộc.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.ingest import MIN_AUTHORITATIVE_SCORE
from app.schemas import CitationResponse, MedicineResponse
from app.textnorm import bigrams, content_tokens, tokenize

#: Trường nào của thuốc được phép trích dẫn, kèm nhãn tiếng Việt hiển thị.
#: ⚠️ Cố ý **không** có `manufacturer`: không phải thông tin an toàn, và đưa nó vào
#: prompt là mở rộng bề mặt dữ liệu gửi ra ngoài mà không cần thiết.
CITABLE_FIELDS: tuple[tuple[str, str], ...] = (
    ("composition", "Thành phần"),
    ("uses", "Công dụng"),
    ("side_effects", "Tác dụng phụ"),
)

#: Ngưỡng liên quan tối thiểu để coi một đoạn là **căn cứ**.
#:
#: Vì sao cần: trùng từ bắt gặp vẫn cho điểm > 0. Đo đoạn trong dữ liệu dev:
#: câu hỏi đúng ("Panadol 500mg dùng để làm gì") đạt ~0.40–0.47, còn câu hỏi
#: không liên quan ("thuốc xyzabc123 có tác dụng gì") chỉ đạt ~0.09. Ngưỡng 0.15
#: nằm giữa hai nhóm và **không** phải con số tự ý — nếu đổi bộ dữ liệu thì phải
#: đo lại (xem `tests/test_retrieval.py::test_nguong_loc_bien_tinh_bang_du_lieu`).
MIN_RELEVANCE_SCORE = 0.15

#: Tỉ lệ từ truy vấn tối thiểu phải xuất hiện trong đoạn để coi là **căn cứ**.
#:
#: Đo trên dữ liệu dev: "bạn bị gì?" khớp đúng một từ "ban" (trong "phát ban") và
#: đạt điểm 0.50 — cao hơn cả "tác dụng phụ của panadol" (0.30). Không có điều
#: kiện này thì hệ thống trả lời lệch hẳn câu hỏi mà vẫn tưởng là có căn cứ.
MIN_TERM_COVERAGE = 0.4

#: Số ký tự tối đa của một đoạn trích dẫn. Quá dài sẽ làm prompt phình và tăng
#: rủi ro mất ngữ cảnh giữa chừng.
MAX_EXCERPT_CHARS = 400


@dataclass(frozen=True)
class Retrieved:
    """Một đoạn truy hồi được + điểm liên quan."""

    citation: CitationResponse
    score: float


def _excerpt(value: str) -> str:
    """Cắt bớt độ dài, cắt tại ranh giới câu/từ để không bị cụt chữ."""
    text = " ".join(value.split())
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    return text[:MAX_EXCERPT_CHARS].rsplit(" ", 1)[0] + "..."


def _terms_of(text: str) -> list[str]:
    """Từ khoá + cụm dùng để chấm điểm một đoạn văn bản.

    ⚠️ **Giữ cả từ chức năng** ở đây (không lọc `STOPWORDS` như truy vấn): văn bản
    nguồn là nội dung để trích, lọc từ sẽ làm mất thông tin. Phía truy vấn mới là
    bên lọc.
    """
    tokens = tokenize(text)[:60]
    return tokens + bigrams(tokens)


def _score(query_terms: list[str], doc_terms: list[str]) -> float:
    """Điểm kiểu BM25 đơn giản, chuẩn hoá về 0..1.

    - `idf`: từ hiếm trong tập tài liệu thì đáng giá hơn (giống BM25).
    - `tf`: từ xuất hiện nhiều lần trong đoạn thì tăng, nhưng **bão hoà** (log) để
      một đoạn lặp từ không hạ gục điểm.
    - Ưu tiên cụm (bigram) hơn từ đơn: khớp "ha sot" mạnh hơn khớp lẻ "ha".
    """
    if not query_terms or not doc_terms:
        return 0.0

    doc_counts = Counter(doc_terms)
    doc_len = len(doc_terms)
    scores: dict[str, float] = {}
    for term in set(query_terms):
        if term not in doc_counts:
            continue
        tf = doc_counts[term]
        # bigram là cụm -> trọng số gấp đôi từ đơn
        weight = 1.0 + (1.0 if " " in term else 0.0)
        # bão hoà tf (log), chuẩn hoá theo độ dài đoạn
        scores[term] = weight * (1 + math.log(tf)) / (1 + 0.0005 * doc_len)

    if not scores:
        return 0.0

    raw = sum(scores.values())
    # Chia cho tổng "giá trị lý thuyết" để điểm luôn nằm trong 0..1
    max_possible = len(set(query_terms)) * 2.0
    return min(raw / max_possible, 1.0)


def _coverage(query_terms: list[str], doc_counts: Counter[str]) -> float:
    """Tỉ lệ từ truy vấn **có mặt** trong đoạn (không tính bigram).

    ⚠️ Vì sao cần: `_score` bão hoà nên một đoạn chỉ cần khớp **vài** từ là đạt
    điểm cao. Đo được trường hợp "bạn bị gì?" khớp mỗi "phát ban" là 0.50 — trả
    lời lệch hẳn câu hỏi. Đòi phủ ≥ 40% từ truy vấn đóng lỗ đó mà không cần đổi
    công thức điểm.
    """
    singles = {t for t in query_terms if " " not in t}
    if not singles:
        return 0.0
    return len(singles & set(doc_counts)) / len(singles)


def _named_medicines(
    medicines: list[MedicineResponse], terms: list[str]
) -> list[MedicineResponse]:
    """Thuốc được **gọi đích danh** trong câu hỏi.

    Một từ là "định danh" khi nằm trong tên thuốc A mà **không** xuất hiện trong văn
    bản của thuốc nào khác (vd "panadol"). Hoạt chất như "paracetamol" có mặt ở nhiều
    thuốc nên KHÔNG định danh — hỏi theo hoạt chất vẫn ra mọi thuốc chứa nó.

    ⚠️ Vì sao cần: đoạn trích có tiền tố nhãn ("Tác dụng phụ: ..."), nên câu "Panadol
    có tác dụng phụ gì" khớp nhãn của Decolgen/Hapacol/... dù người dùng chỉ hỏi
    Panadol. Từ có chữ số (hàm lượng "500mg") bị bỏ qua vì dùng chung nhiều thuốc.
    """
    candidates = {t for t in terms if " " not in t and t.isalpha()}
    if not candidates:
        return []

    def _vocab(med: MedicineResponse) -> set[str]:
        parts = [med.name] + [getattr(med, f, None) or "" for f, _ in CITABLE_FIELDS]
        return {tok for part in parts for tok in tokenize(part)}

    vocabs = [(med, _vocab(med), set(tokenize(med.name))) for med in medicines]
    named: list[MedicineResponse] = []
    for term in candidates:
        # Danh mục thật có nhiều sản phẩm cùng tên ("Panadol", "Panadol Extra"...):
        # từ vẫn là định danh nếu nó nằm trong TÊN của cả nhóm và không xuất hiện ở
        # văn bản của thuốc nào ngoài nhóm.
        owners = [med for med, _, names in vocabs if term in names]
        if not owners:
            continue
        if any(term in v for _, v, names in vocabs if term not in names):
            continue
        named.extend(m for m in owners if all(m is not n for n in named))
    return named


#: Cụm hỏi về TRƯỜNG thông tin (đã bỏ dấu) -> trường tương ứng. Các âm tiết của cụm
#: không được dùng để so khớp nội dung: bỏ dấu thì "tác" = "tặc" = "tắc", nên "tác
#: dụng phụ" khớp nhầm "ô tặc cốt", "hương phụ" trong thành phần thuốc đông y.
_FIELD_INTENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tac", "dung", "phu"), "side_effects"),
    (("tac", "dung"), "uses"),
    (("cong", "dung"), "uses"),
    (("thanh", "phan"), "composition"),
    (("hoat", "chat"), "composition"),
)


_VARIANT_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]+")
_STRENGTH_RE = re.compile(r"\d+(?:mg|mcg|g|ml|iu|ui|%)?")


def _field_intent(question: str, terms: list[str]) -> tuple[str | None, list[str]]:
    """(trường được hỏi, từ khoá còn lại sau khi bỏ âm tiết của cụm hỏi)."""
    tokens = tokenize(question)
    for phrase, field in _FIELD_INTENTS:
        n = len(phrase)
        if any(tuple(tokens[i : i + n]) == phrase for i in range(len(tokens) - n + 1)):
            rest = list(terms)
            for syllable in phrase:
                if syllable in rest:
                    rest.remove(syllable)
            return field, rest
    return None, terms


def build_citations(
    medicines: list[MedicineResponse],
    question: str,
    top_k: int = 5,
) -> list[Retrieved]:
    """Trích các đoạn liên quan từ danh mục thuốc, kèm mã trích dẫn.

    ⚠️ Mã trích dẫn (`R1`, `R2`...) do hàm này **sinh ra**, không lấy từ dữ liệu.
    Đó là điều làm cho việc một dòng CSV độc hại tự gắn mã trích dẫn giả trở nên
    bất khả thi (xem `app/prompt.py`).
    """
    terms = content_tokens(question)
    if not terms:
        return []
    wanted_field, terms = _field_intent(question, terms)
    if not terms:
        # Chỉ hỏi "tác dụng phụ là gì" mà không nêu thuốc nào -> không có căn cứ.
        return []
    query_terms = terms + bigrams(terms)
    # Mã biến thể chữ+số ("d3", "b12") phải có mặt: "Vitamin D3" không được trả lời
    # bằng thuốc chỉ chứa "vitamin". Hàm lượng ("500mg", "500") không tính.
    variants = {t for t in terms if _VARIANT_RE.fullmatch(t) and not _STRENGTH_RE.fullmatch(t)}

    medicines = _named_medicines(medicines, terms) or medicines

    hits: list[Retrieved] = []
    counter = 0
    for med in medicines:
        # Tên thuốc cũng là ngữ cảnh: hỏi "panadol 500mg" phải ra Panadol.
        for field, label in CITABLE_FIELDS:
            value = getattr(med, field, None)
            if not value:
                continue
            # Hoạt chất là một phần danh tính của thuốc: "tác dụng phụ của
            # paracetamol" phải khớp tác dụng phụ của mọi thuốc chứa paracetamol.
            identity = med.name
            if field != "composition" and med.composition:
                identity = f"{med.name} ({med.composition})"
            text = f"{identity} — {label}: {value}"
            doc_terms = _terms_of(text)
            score = _score(query_terms, doc_terms)
            if score < MIN_RELEVANCE_SCORE:
                # Trùng từ bắt gặp cho điểm > 0, nhưng câu hỏi không liên quan
                # thì điểm rất thấp -> không phải căn cứ.
                continue
            if variants and not variants <= set(doc_terms):
                continue
            if _coverage(query_terms, Counter(doc_terms)) < MIN_TERM_COVERAGE:
                # Điểm cao nhưng chỉ khớp vài từ bất hàm -> trả lời lệch câu hỏi.
                continue
            counter += 1
            hits.append(
                Retrieved(
                    citation=CitationResponse(
                        id=f"R{counter}",
                        medicine_name=med.name,
                        field=field,
                        excerpt=_excerpt(value),
                        # ⚠️ Không có nguồn -> không được trả về.
                        source=med.source,
                        is_authoritative=med.is_authoritative,
                        score=round(score, 4),
                        source_url=med.source_url or "",
                        retrieved_at=med.retrieved_at or "",
                        license=med.license or "",
                        reviewed_by=med.reviewed_by or "",
                    ),
                    score=score,
                )
            )

    # Hỏi đúng một trường (vd tác dụng phụ) -> chỉ trả đoạn của trường đó. Không có
    # thì là "không đủ căn cứ", không lấy trường khác cho đủ (trả lời lệch câu hỏi).
    if wanted_field:
        hits = [h for h in hits if h.citation.field == wanted_field]
    hits.sort(key=lambda r: r.score, reverse=True)
    hits = hits[:top_k]
    # Mã trích dẫn đánh lại theo thứ tự hiển thị: R1, R2... liên tục.
    return [
        Retrieved(citation=h.citation.model_copy(update={"id": f"R{i}"}), score=h.score)
        for i, h in enumerate(hits, start=1)
    ]


def has_authoritative_evidence(retrieved: list[Retrieved]) -> bool:
    """Có căn cứ **đủ tin cậy** để trả lời câu hỏi y khoa không?

    ⚠️ Ngưỡng lấy từ `app.ingest.MIN_AUTHORITATIVE_SCORE` — dùng dữ liệu dev
    (`is_authoritative=False`) thì hàm này **luôn trả `False`**, nên agent phải từ
    chối. Đây là hành vi đúng, không phải bug.
    """
    return any(
        r.citation.is_authoritative and r.score >= MIN_AUTHORITATIVE_SCORE
        for r in retrieved
    )
