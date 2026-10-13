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
    query_terms = terms + bigrams(terms)

    hits: list[Retrieved] = []
    counter = 0
    for med in medicines:
        # Tên thuốc cũng là ngữ cảnh: hỏi "panadol 500mg" phải ra Panadol.
        for field, label in CITABLE_FIELDS:
            value = getattr(med, field, None)
            if not value:
                continue
            text = f"{med.name} — {label}: {value}"
            doc_terms = _terms_of(text)
            score = _score(query_terms, doc_terms)
            if score < MIN_RELEVANCE_SCORE:
                # Trùng từ bắt gặp cho điểm > 0, nhưng câu hỏi không liên quan
                # thì điểm rất thấp -> không phải căn cứ.
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
                    ),
                    score=score,
                )
            )

    hits.sort(key=lambda r: r.score, reverse=True)
    return hits[:top_k]


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
