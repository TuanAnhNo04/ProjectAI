"""Hội thoại xã giao (small talk) theo luật — KHÔNG LLM, KHÔNG gửi gì ra ngoài.

Chỉ kích hoạt khi **toàn bộ** tin nhắn (<= 60 ký tự) là xã giao. Tin nhắn lẫn câu hỏi
thuốc/triệu chứng ("chào bạn, Panadol có tác dụng phụ gì") đi luồng thường. Dấu hiệu
khẩn cấp luôn thắng (kiểm tra ở đây và ở orchestrator). Không ghi log nội dung.
"""

from __future__ import annotations

import re
import zlib

from app.safety import detect_red_flags, detect_symptoms
from app.textnorm import normalize

MAX_LEN = 60

_SUF = r"(?: (?:ban|oi|nhe|nha|a|ah|ak|ha|nhi|vay|day|bot|ad|em|anh|chi|medassist|tro ly))*"
_GREET = r"(?:xin chao|chao|chao buoi (?:sang|trua|chieu|toi)|hello|hi|hey|helo|alo)"

_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "howareyou",
        rf"(?:{_GREET}{_SUF} )?(?:ban )?(?:co )?khoe (?:khong|ko|chu|hong)|"
        rf"(?:{_GREET}{_SUF} )?(?:hom nay )?(?:ban )?(?:the nao|ra sao|co on khong)|how are you",
    ),
    ("greeting", rf"{_GREET}{_SUF}"),
    (
        "thanks",
        rf"(?:cam on|thanks|thank you|thank|tks){_SUF}(?: nhieu| rat nhieu)?{_SUF}",
    ),
    ("bye", rf"(?:tam biet|bye bye|bye|goodbye|good bye|hen gap lai){_SUF}"),
    (
        "capabilities",
        r"(?:ban )?(?:co the )?lam duoc gi|(?:ban )?(?:co the )?giup (?:duoc )?(?:gi|toi gi)(?: duoc)?"
        r"(?: cho (?:toi|minh|em))?|giup gi duoc cho (?:toi|minh|em)|ban biet lam gi|what can you do",
    ),
    (
        "whoami",
        r"ban la (?:ai|gi)(?: vay)?|ban ten (?:la )?gi|ten ban la gi|gioi thieu (?:ve )?ban|who are you",
    ),
    (
        "ack",
        rf"(?:ok|oke|okay|okie|vang|da|da vang|u|uh|uhm|um|duoc roi|ok roi|hieu roi|ro roi|tot|tuyet){_SUF}",
    ),
    (
        "wellbeing",
        r"(?:hom nay )?(?:(?:minh|toi|em|tui|con) )?(?:hoi |dang |thay |cam thay )?"
        r"(?:buon|chan|stress|cang thang|khong khoe|khong vui|co don|lo lang|bat on)"
        r"(?: lam| qua| ghe| xiu| chut)?",
    ),
)
_COMPILED = tuple((name, re.compile(pat)) for name, pat in _PATTERNS)

_TAIL = "Mình là trợ lý thông tin, không phải bác sĩ và không chẩn đoán bệnh."
_ASK = (
    "Bạn có thể hỏi mình về tra cứu thuốc, tác dụng phụ, nhắc uống thuốc hoặc tình huống cần cấp cứu."
)

_REPLIES: dict[str, tuple[str, ...]] = {
    "greeting": (
        f"Xin chào bạn! {_TAIL} {_ASK}",
        f"Chào bạn, rất vui được gặp bạn. {_TAIL} Bạn muốn tra cứu thuốc hay tác dụng phụ nào hôm nay?",
        f"Chào bạn! Mình sẵn sàng hỗ trợ. {_ASK} {_TAIL}",
    ),
    "howareyou": (
        f"Mình ổn, cảm ơn bạn đã hỏi thăm! {_TAIL} Bạn cần mình hỗ trợ gì về thuốc không?",
        f"Cảm ơn bạn, mình vẫn hoạt động tốt. Còn bạn hôm nay thế nào? {_ASK}",
        f"Mình khỏe, cảm ơn bạn nhé. {_TAIL} Nếu bạn thấy không khỏe, hãy mô tả để mình hỗ trợ tra cứu.",
    ),
    "thanks": (
        "Không có gì ạ! Cần tra cứu thuốc hay nhắc uống thuốc, cứ hỏi mình nhé.",
        f"Rất vui được giúp bạn. {_ASK}",
        "Bạn đừng ngại nhé! Nếu còn băn khoăn về thuốc, hãy hỏi dược sĩ hoặc bác sĩ để chắc chắn hơn.",
    ),
    "bye": (
        "Tạm biệt bạn! Chúc bạn nhiều sức khỏe. Cần gì cứ quay lại hỏi mình nhé.",
        "Hẹn gặp lại bạn! Nhớ uống thuốc đúng chỉ dẫn của bác sĩ nhé.",
        "Chào bạn, chúc bạn một ngày thật tốt lành!",
    ),
    "whoami": (
        f"Mình là MedAssist, trợ lý tra cứu thông tin thuốc. {_TAIL} {_ASK}",
        f"Mình là trợ lý thông tin thuốc của ứng dụng MedAssist. {_TAIL} Mình chỉ trả lời dựa trên nguồn có trích dẫn.",
        f"Chào bạn, mình là MedAssist, giúp bạn tra cứu thuốc có nguồn trích dẫn. {_TAIL}",
    ),
    "capabilities": (
        "Mình có thể tra cứu thuốc, tác dụng phụ (có nguồn trích dẫn), đặt nhắc uống thuốc và chỉ dẫn khi gặp "
        "tình huống cấp cứu. Mình không chẩn đoán, không đưa liều dùng và luôn khuyên hỏi dược sĩ hoặc bác sĩ khi thiếu căn cứ.",
        "Mình giúp bạn: tra cứu thông tin thuốc và tác dụng phụ kèm nguồn, nhắc uống thuốc, nhận biết dấu hiệu "
        "cần cấp cứu. Giới hạn: mình không phải bác sĩ, không chẩn đoán và không tư vấn liều.",
        "Bạn có thể nhờ mình tra cứu thuốc, tác dụng phụ, đặt nhắc uống thuốc hoặc hỏi về tình huống cấp cứu. "
        "Mọi thông tin đều có trích dẫn; mình không chẩn đoán hay kê liều.",
    ),
    "ack": (
        "Vâng ạ. Bạn cần mình hỗ trợ thêm điều gì không?",
        f"Được rồi! {_ASK}",
        "Ok bạn nhé. Cần tra cứu gì thêm cứ hỏi mình.",
    ),
    "wellbeing": (
        "Mình rất tiếc khi nghe vậy. Bạn có thể mô tả rõ hơn bạn đang thấy thế nào không? "
        "Mình không phải bác sĩ nên không chẩn đoán, nhưng có thể giúp tra cứu thông tin.",
        "Cảm ơn bạn đã chia sẻ, mong bạn sớm thấy khá hơn. Bạn kể thêm triệu chứng cụ thể nhé? "
        "Nếu nặng hoặc kéo dài, hãy gặp bác sĩ.",
        "Mình hiểu rồi. Bạn mô tả cụ thể hơn được không (bắt đầu từ khi nào, có dùng thuốc gì)? "
        "Mình chỉ cung cấp thông tin tham khảo, không thay thế bác sĩ.",
    ),
}


def smalltalk_intent(message: str) -> str | None:
    """Tên intent xã giao nếu CẢ tin nhắn là xã giao; ngược lại None."""
    raw = (message or "").strip()
    if not raw or len(raw) > MAX_LEN:
        return None
    if detect_red_flags(raw):
        return None  # khẩn cấp luôn thắng
    if detect_symptoms(raw):
        return None  # triệu chứng -> luồng thường
    flat = re.sub(r"\s+", " ", re.sub(r"[^\w\s]|_", " ", normalize(raw))).strip()
    if not flat:
        return None
    for name, pat in _COMPILED:
        if pat.fullmatch(flat):
            return name
    return None


def smalltalk_reply(message: str, session_id: str = "") -> str | None:
    """Câu trả lời xã giao (xác định theo hash của session_id) hoặc None."""
    intent = smalltalk_intent(message)
    if intent is None:
        return None
    variants = _REPLIES[intent]
    return variants[zlib.crc32((session_id or "").encode("utf-8")) % len(variants)]
