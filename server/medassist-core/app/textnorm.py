"""Chuẩn hoá và tách âm tiết tiếng Việt (Day 20, bước chuẩn bị cho RAG).

## Vì sao phải chuẩn hoá TRƯỚC khi tách từ?

Tiếng Việt viết **không dấu ở nguyên âm và thanh điệu** (`a`/`à`/`á`/`ả`/`ã`/`ạ`
đều là cùng một âm tiết "a"). Nếu tách từ trên chuỗi thô thì:

- `Hạ Sốt` -> `["Hạ", "Sốt"]`, còn `Ha Sot` -> `["Ha", "Sot"]` -> **không khớp**,
  dù là cùng một từ;
- người dùng gõ không dấu (rất phổ biến trên mobile) không tìm được gì.

Vì vậy thứ tự bắt buộc là: **bỏ dấu + hạ chữ thường -> tách âm tiết -> so khớp**.

## Vì sao tách theo âm tiết?

Tiếng Việt là tiếng **đơn âm tiết**: mỗi từ chỉ có một âm tiết. Tách theo khoảng
trắng cho "hạ sốt" hai từ, còn "hạch" là một từ — không có cách nào biết trước.
Nên ta tách thành âm tiết rồi so khớp theo **cặp âm tiết liền nhau (bigram)**, vừa
giải quyết vụ "hạ sốt"/"Ha Sot" vừa không cần bộ tách từ tiếng Việt (thư viện ngoài).

⚠️ Dùng lại `normalize()` của `sqlite_repository` chứ không viết lại: đó là nơi
duy nhất giữ bảng `_FOLD_EXTRA` xử lý chữ `Đ/đ` (mà `unicodedata` NFD **không** bỏ
được, vì `Đ` là chữ cái riêng chứ không phải chữ Latin có dấu).
"""

from __future__ import annotations

import re

from app.sqlite_repository import normalize

#: Sau khi `normalize()` (không dấu, chữ thường) chỉ còn chữ/số a-z và 0-9.
#: Tiếng Việt có âm tiết rất ngắn ("y", "cơ") nên **không** dùng ranh giới từ `\b`
#: theo hiểu thức tìm — chỉ gom ký tự chữ/số rồi cắt theo nhóm.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Từ chức năng tiếng Việt. Được loại khỏi *truy vấn* khi tìm kiếm, nhưng **giữ lại
#: trong văn bản nguồn** khi chấm điểm — nguồn là dữ liệu, không phải lệnh.
#:
#: ⚠️ Danh sách này **cố tình ngắn**. Bộ tiếng Việt đầy đủ dễ nuốt mất từ khoá thật
#: ("đau" là từ chức năng trong "đau bụng" nhưng lại là chủ đề của câu hỏi).
#:
#: ⚠️ Vì sao có "ban": hỏi "bạn bị gì?" thì "ban" là **tân từ** gọi đối tượng. Giữ
#: lại thì truy vấn chỉ còn đúng một từ "ban", mà nó lại khớp với triệu chứng "phát
#: ban" trong nguồn — hệ thống sẽ trả lời **lệch hẳn** câu hỏi. Đổ lại: muốn hỏi về
#: ban, người dùng gõ "phát ban"/"nổi ban", và "phát"/"noi" vẫn còn là từ khoá.
STOPWORDS = frozenset(
    """
    a an ac ai bao bay ben bi bo ca cap cho co cua coi con cua da de den
    deu di do duoc dung duy gai hay hoac hon khi khong la lam len luong
    ma moi mua nao nay nen neu nhi nhung noi nuoc o on or pai ra roi sao
    sang sua tai ten thuoc thi toi toan trong tu van va ve voi voi
    gi the nao bao nhieu day ban b khoe giup dac khac nua
    """.split()
)

#: Số âm tiết tối đa giữ lại từ một trường nguồn khi chấm điểm (tránh trường quá
#: dài làm nhiễu điểm của từ khoá thật).
MAX_TOKENS_PER_FIELD = 40


def tokenize(text: str) -> list[str]:
    """Tách thành âm tiết đã chuẩn hoá.

    >>> tokenize("Hạ sốt")
    ['ha', 'sot']
    >>> tokenize("Paracetamol 500mg")
    ['paracetamol', '500mg']
    """
    return _TOKEN_RE.findall(normalize(text))


def bigrams(tokens: list[str]) -> list[str]:
    """Cặp âm tiết liền nhau.

    `["ha", "sot"]` -> `["ha sot"]`. Đây là đơn vị so khớp chính: một cụm tiếng
    Việt nhiều từ sẽ khớp cả cụm thay vì khớp từng từ rời rạc.

    Với danh sách 1 phần tử, trả `[]` (không có cặp nào để so).
    """
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def content_tokens(text: str, keep_numbers: bool = True) -> list[str]:
    """Âm tiết đã bỏ từ chức năng, giới hạn độ dài.

    `keep_numbers=True` giữ lại "500", "500mg" — người dùng hỏi "panadol 500mg"
    thì số là một phần của tên thuốc, không phải từ chức năng.
    """
    kept = [
        t
        for t in tokenize(text)
        if keep_numbers or not t.isdigit()
    ]
    kept = [t for t in kept if t not in STOPWORDS]
    return kept[:MAX_TOKENS_PER_FIELD]


def query_terms(query: str) -> list[str]:
    """Từ khoá truy vấn: âm tiết đã bỏ từ chức năng + bigram của chúng.

    Trả về `["ha", "sot", "ha sot"]` cho `"Hạ sốt"` — tầng truy hồi dùng để chấm
    điểm, nên cả từ đơn lẫn cụm đều có giá trị.
    """
    tokens = content_tokens(query)
    return tokens + bigrams(tokens)


def normalize_query(query: str) -> str:
    """Chuẩn hoá chuỗi truy vấn để so khớp tên thuốc (dùng cho `search_medicines`)."""
    return normalize(query)
