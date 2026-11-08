"""Khử định danh văn bản trước khi gửi ra ngoài máy (điều kiện (b) của `AGENTS.md` mục 2).

Dùng cho adapter LLM đám mây (`app/llm_claude.py`) — **không** dùng cho Ollama cục bộ
(`app/llm_ollama.py`), vì dữ liệu đó không rời máy.

## Dựa trên nghiên cứu (28/10/2026)

Danh mục định danh theo HIPAA Safe Harbor (45 CFR §164.514(b)(2)(i), 18 nhóm):
https://www.law.cornell.edu/cfr/text/45/164.514 — regex chỉ bắt được các nhóm có cấu
trúc rõ (số điện thoại, email, CCCD/CMND, ngày tháng, địa chỉ web/IP). **Tên người tiếng
Việt không bắt được bằng regex thuần** (tên là từ thường) — tài liệu khuyến nghị hybrid
regex + NER; ở đây chỉ bắt tên có danh xưng dẫn trước ("bệnh nhân Nguyễn Văn A").

⚠️ Đây là **lưới an toàn giảm thiểu rủi ro, không phải đảm bảo tuyệt đối**. Luôn nói rõ
với người dùng giới hạn này trong màn hình đồng ý (xem `docs/PRIVACY.md`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Thứ tự quan trọng: số dài khớp trước số ngắn (CCCD 12 số trước CMND 9 số).
# Từ khoá (ngày, quận, ...) dùng cờ cục bộ `(?i:...)`; tên riêng PHẢI viết hoa để các
# câu hỏi thuốc thông thường ("2 viên Paracetamol 500mg") không bị che nhầm.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    (
        "PHONE",
        re.compile(r"(?<!\d)(?:\+?84|0)[\s.\-]?[35789](?:[\s.\-]?\d){8}(?!\d)"),
    ),
    # Điện thoại bàn: (028) 3822 1234 / 024 38221234 / 0292 3812345.
    (
        "PHONE",
        re.compile(r"(?<!\d)\(?0(?:2\d{1,2})\)?[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}(?!\d)"),
    ),
    ("ID_CARD", re.compile(r"(?<!\d)(?:\d{12}|\d{9})(?!\d)")),
    ("INSURANCE", re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d[A-Z]{2}\d{10}(?![A-Z0-9])")),
    # Mã BHYT dạng 2 chữ + 10-13 số (vd DN4797912345678).
    ("INSURANCE", re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{10,13}(?![A-Za-z0-9])")),
    (
        "MRN",
        re.compile(r"(?i)\b(?:MBN|BN|HS|MSBN|PID)[\s:\-]?\d{4,12}\b"),
    ),
    (
        "DATE",
        re.compile(
            r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:\d{2}|\d{4})(?!\d)"
        ),
    ),
    # Ngày viết chữ: "5 tháng 3 năm 1980", "ngày 5 tháng 3", "ngày 5/3".
    (
        "DATE",
        re.compile(
            r"(?i:(?<!\d)(?:ngày\s+)?(?:0?[1-9]|[12]\d|3[01])\s+tháng\s+(?:0?[1-9]|1[0-2])"
            r"(?:\s+năm\s+(?:19|20)\d{2})?(?!\d))"
        ),
    ),
    ("DATE", re.compile(r"(?i:\bngày)\s+(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])(?!\d)")),
    ("DATE", re.compile(r"(?<!\d)(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)")),
    # Địa chỉ đơn giản — chỉ bắt khi có mỏ neo ("số", "đường", phường/quận/huyện).
    (
        "ADDRESS",
        re.compile(
            r"\d+[A-Za-z]?(?:/\d+)?\s+(?:[A-ZĐÀ-Ỹ][a-zđà-ỹ]+\s+){1,3}"
            r"(?i:phường|quận|huyện|xã)\s+(?:\d{1,2}\b|[A-ZĐÀ-Ỹ][\wà-ỹ]*)"
            r"(?:\s*,?\s*(?i:phường|quận|huyện|xã|tp\.?|thành phố)\s+(?:\d{1,2}\b|[A-ZĐÀ-Ỹ][\wà-ỹ]*))*"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"(?i:\bsố\s+)\d+[A-Za-z]?(?:/\d+)?(?:\s+(?i:đường))?(?:\s+[A-ZĐÀ-Ỹ][a-zđà-ỹ]+){1,3}"
            r"(?:\s*,?\s*(?i:phường|quận|huyện|xã)\s+(?:\d{1,2}\b|[A-ZĐÀ-Ỹ][\wà-ỹ]*))*"
        ),
    ),
    ("ADDRESS", re.compile(r"(?i:\bđường)\s+(?:[A-ZĐÀ-Ỹ][a-zđà-ỹ]+\s*){1,3}")),
    (
        "ADDRESS",
        re.compile(
            r"(?i:\b(?:phường|quận|huyện|xã))\s+"
            r"(?:\d{1,2}\b|[A-ZĐÀ-Ỹ][\wà-ỹ]*(?:\s+[A-ZĐÀ-Ỹ][\wà-ỹ]*)?)"
        ),
    ),
    ("URL", re.compile(r"https?://\S+|www\.\S+")),
    ("IP", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    (
        # Tên có danh xưng dẫn trước — xem docstring: regex KHÔNG bắt được tên trần.
        # Token cuối có thể là chữ cái viết tắt ("Lê Thị B").
        "NAME",
        re.compile(
            r"(?i:\b(?:bệnh nhân|b\.?n|bn|ông|bà|anh|chị|em|cháu|bé|"
            r"tôi tên(?: là)?|tên tôi là|họ và tên|bs\.?|bác sĩ))\s+"
            r"((?:[A-ZĐÀ-Ỹ][a-zđà-ỹ]+\s+){1,3}(?:[A-ZĐÀ-Ỹ][a-zđà-ỹ]+|[A-ZĐ]\b\.?))"
        ),
    ),
]


@dataclass
class DeidentifyResult:
    text: str
    #: Số lượng mỗi loại định danh đã thay thế — CHỈ số đếm, không chứa giá trị gốc.
    redacted_counts: dict[str, int] = field(default_factory=dict)

    @property
    def had_any(self) -> bool:
        return sum(self.redacted_counts.values()) > 0


def deidentify(text: str) -> DeidentifyResult:
    """Thay các định danh tìm được bằng nhãn `<LOẠI>`. Không ném lỗi với mọi input."""
    # Chuẩn hoá NFC trước khi so khớp: tiếng Việt có cả NFC/NFD, lớp ký tự [à-ỹ] chỉ
    # khớp đúng khi tổ hợp dấu đã ghép (NFC).
    normalized = unicodedata.normalize("NFC", text or "")
    counts: dict[str, int] = {}
    result = normalized
    for label, pattern in _PATTERNS:
        result, n = pattern.subn(f"<{label}>", result)
        if n:
            counts[label] = n
    return DeidentifyResult(text=result, redacted_counts=counts)
