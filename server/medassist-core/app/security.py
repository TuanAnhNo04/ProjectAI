"""Băm mật khẩu.

## Vì sao `bcrypt` chứ không phải `hashlib`?
`AGENTS.md` yêu cầu **argon2/bcrypt**. `hashlib.sha256` là **KHÔNG** phải mật
khẩu hóa: bẻ được siêu nhanh trên GPU. bcrypt cố ý chậm và có salt tích hợp.
Máy dev đã có sẵn `bcrypt` nên không cần cài thêm package nào.

## Token đã chuyển sang `app/jwt_tokens.py` (Day 16)
Hàm `issue_token`/`verify_token` của Day 15 đã **xoá** để không còn hai hệ thống
token cùng tồn tại — dễ nhầm lẫn và là nguồn lỗ hổng.
"""

from __future__ import annotations

import bcrypt

#: Số byte ngẫu nhiên cho định danh phiên (không dùng cho token đã băm).
TOKEN_BYTES = 32

#: bcrypt chỉ xử lý **tối đa 72 byte**. Mật khẩu dài hơn sẽ bị **cắt cụt** —
#: cắt âm thầm là lỗ hổng: `"A"*72 + "X"` và `"A"*72 + "Y"` sẽ cùng đăng nhập
#: được. Vì vậy ta **từ chối** thay vì cắt (quyết định N3).
MAX_PASSWORD_BYTES = 72

#: Số vòng lặp. 12 là mức tối thiểu chấp nhận được hiện nay (OWASP); bcrypt
#: mặc định cũng là 12 nhưng ghi rõ ở đây để không bị đổi âm thầm khi nâng
#: thư viện.
BCRYPT_ROUNDS = 12


class PasswordTooLongError(ValueError):
    """Mật khẩu vượt giới hạn byte của bcrypt — phải từ chối, không cắt."""


def _check_password_length(password: str) -> bytes:
    """Kiểm tra và trả về dạng bytes đã kiểm tra.

    Ném [PasswordTooLongError] nếu vượt 72 byte, [ValueError] nếu rỗng.
    """
    if not password:
        raise ValueError("Mật khẩu không được rỗng.")
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        # Nêu rõ vì sao, nhưng **không** echo lại mật khẩu.
        raise PasswordTooLongError(
            f"Mật khẩu dài {len(raw)} byte, vượt giới hạn {MAX_PASSWORD_BYTES} byte "
            f"của bcrypt."
        )
    return raw


def hash_password(password: str) -> str:
    """Băm mật khẩu bằng bcrypt với `rounds` >= 12. **Không** log kết quả này."""
    raw = _check_password_length(password)
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Kiểm tra mật khẩu. Trả `False` nếu hash sai dạng (không ném lỗi).

    Mật khẩu quá dài trả `False` thay vì ném — endpoint đăng nhập sẽ trả 401
    chứ không 500.
    """
    try:
        raw = _check_password_length(password)
    except (PasswordTooLongError, ValueError):
        return False

    try:
        return bcrypt.checkpw(raw, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Hash hỏng trong DB -> coi như sai mật khẩu, KHÔNG ném 500.
        return False


# NOTE: `issue_token` / `verify_token` của Day 15 đã bị **xoá** trong Day 16,
# thay bằng JWT thật trong `app/jwt_tokens.py`. Giữ cả hai hệ thống token sẽ
# gây nhầm lẫn và là nguồn lỗ hổng (token cũ vẫn còn hợp lệ).
