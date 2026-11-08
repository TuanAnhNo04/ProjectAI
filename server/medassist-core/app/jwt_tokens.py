"""JWT HS256 — dùng **`PyJWT`** (thư viện chuẩn), không tự cài thuật toán.

## Lịch sử (nợ N4 đã đóng 05/10/2026)

Bản đầu tiên **tự** cài HS256 bằng `hmac`/`base64` vì `AGENTS.md` mục 2 cấm tự thêm
package. Sau khi chủ dự án duyệt (`pip install "PyJWT>=2.8"`), toàn bộ phần ký/kiểm
chữ ký đã chuyển sang `PyJWT`. Lý do thay **không phải** "viết lại cho chuẩn" —
bản cũ vốn đã đúng thuật toán — mà là:

- `PyJWT` được kiểm thử với hàng nghìn vector tấn công JWT đã biết (rút gọn header,
  `alg` confusion, chữ ký cắt ngắn...). Tự viết thì không có lớp kiểm thử đó.
- Bản cũ phải **tự** nhớ kiểm tra `alg` đúng HS256, so sánh thời gian cố định, kiểm
  hạn. Bỏ sót một chỗ là lỗ hổng. `PyJWT` làm sẵn và có test riêng.

## ⚠️ `sub` là **int**, không phải chuỗi — và đây là chủ ý

RFC 7519 nói `sub` phải là StringOrURI, và `PyJWT` **từ chúng 2.10** ném
`InvalidSubjectError` nếu `sub` không phải chuỗi. Nhưng `app/deps.py` dùng `sub`
làm `user_id` truy vấn kho (`isinstance(sub, int)`).

Ta **giữ `sub` là int** và tắt kiểm tra đó (`options={"verify_sub": False}`), rồi
**tự kiểm tra lại bằng tay** trong `decode()`:

- đổi `sub` sang chuỗi sẽ phá hợp đồng `deps.py` và làm token cũ không dùng được;
- tắt kiểm tra rồi bỏ qua thì mất luôn cái kiểm tra đó — nên ta kiểm tra tường minh
  ngay trong `decode()`, ngay cả khi thư viện đã làm.

## Thuật toán: allowlist tường minh

`decode()` luôn truyền `algorithms=[ALGORITHM]`. ⚠️ Không bao giờ bỏ đối số này —
`PyJWT` sẽ **tin** `alg` trong header, và đó chính là đường cho tấn công
`alg: none` / đổi HS256 thành RS256 để bỏ qua kiểm tra chữ ký.

## Khoá ký: BẮT BUỘC từ biến môi trường, thiếu thì TỪ CHỐI

⚠️ Quyết định N2 (`docs/DECISIONS.md`): **không** fallback sang khoá ngẫu nhiên theo
tiến trình. Thiếu khoá mà vẫn chạy -> mọi token ký bằng khoá tự sinh -> restart là mất
phiên, hoặc tệ hơn là 2 instance với khoá khác nhau mà không ai biết.

- `MEDASSIST_SECRET_KEY`: bắt buộc, **≥ 32 byte**.
- Thiếu / quá ngắn -> ném [ConfigError] lúc **khởi động** (fail-fast).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError

#: Thuật toán ký. HS256 = HMAC-SHA256.
ALGORITHM = "HS256"

ACCESS_TTL_SECONDS = 15 * 60  # 15 phút (ROADMAP Day 16)
REFRESH_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 ngày

#: Tên biến môi trường chứa khoá ký.
SECRET_ENV_VAR = "MEDASSIST_SECRET_KEY"

#: Khoá tối thiểu. 256 bit là chuẩn cho HS256; khoá ngắn là yếu.
MIN_SECRET_BYTES = 32


class ConfigError(RuntimeError):
    """Cấu hình sai — server **không** được khởi động."""


def load_secret() -> bytes:
    """Đọc khoá ký từ biến môi trường, kiểm tra độ dài.

    Ném [ConfigError] nếu thiếu hoặc ngắn hơn [MIN_SECRET_BYTES] — **fail-fast có
    chủ đích** (quyết định N2). Thà báo lỗi lúc khởi động còn hơn chạy với khoá yếu.
    """
    raw = os.environ.get(SECRET_ENV_VAR)
    if not raw:
        raise ConfigError(
            f"Thiếu biến môi trường {SECRET_ENV_VAR}. Server không thể khởi động.\n"
            f"Tạo khoá ngẫu nhiên an toàn bằng:\n"
            f"  python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    key = raw.encode("utf-8")
    if len(key) < MIN_SECRET_BYTES:
        raise ConfigError(
            f"{SECRET_ENV_VAR} phải dài ít nhất {MIN_SECRET_BYTES} byte, "
            f"hiện chỉ {len(key)} byte."
        )
    return key


#: Khoá đã nạp một lần. `None` = chưa nạp.
_KEY: bytes | None = None


def get_key() -> bytes:
    """Khoá ký hiện hành (nạp một lần)."""
    global _KEY
    if _KEY is None:
        _KEY = load_secret()
    return _KEY


#: Giá trị mẫu đã xuất hiện trong repo (docker-compose, .env.example). Ở chế độ
#: production, dùng lại chúng = ai đọc repo cũng ký được token -> từ chối khởi động.
_KNOWN_PLACEHOLDER_MARKERS = ("doi-thanh", "thay-bang", "change-me", "changeme", "example", "test-only")


def is_production() -> bool:
    """`MEDASSIST_ENV=production` (đặt trên Railway)."""
    return os.environ.get("MEDASSIST_ENV", "").strip().lower() in {"production", "prod"}


def validate_config() -> None:
    """Gọi lúc **khởi động server** để fail-fast (xem `app/main.py`).

    Production còn chặn thêm khoá mẫu và khoá ít ký tự khác nhau (khoá yếu).
    """
    key = get_key()
    if is_production():
        text = key.decode("utf-8", errors="ignore").lower()
        if any(m in text for m in _KNOWN_PLACEHOLDER_MARKERS) or len(set(text)) < 10:
            raise ConfigError(
                f"{SECRET_ENV_VAR} là khoá mẫu/yếu — production từ chối khởi động. "
                "Tạo khoá ngẫu nhiên mới (secrets.token_urlsafe(48))."
            )


def reset_key_cache() -> None:
    """Xoá khoá đã cache — chỉ dùng cho test."""
    global _KEY
    _KEY = None


def encode(payload: dict[str, Any]) -> str:
    """Mã hoá payload thành JWT HS256."""
    return jwt.encode(payload, get_key(), algorithm=ALGORITHM)


def decode(token: str, *, verify_exp: bool = True) -> dict[str, Any] | None:
    """Giải mã và **kiểm tra chữ ký**. Trả `None` nếu sai/hết hạn.

    ⚠️ Mọi lỗi đều quy về `None` — vì caller (`app/deps.py`) coi `None` là "token
    không hợp lệ" và trả 401. Ném lỗi ra ngoài sẽ biến token rác thành 500.
    """
    try:
        payload = jwt.decode(
            token,
            get_key(),
            # ⚠️ Allowlist tường minh. Bỏ dòng này là mở đường cho `alg: none`.
            algorithms=[ALGORITHM],
            options={
                # `sub` là int theo thiết kế (xem docstring) — ta tự kiểm tra bên dưới.
                "verify_sub": False,
                # `verify_exp` do tham số hàm quyết định, không phải mặc định của thư viện.
                "verify_exp": verify_exp,
                "verify_signature": True,
                # Không nhận `nbf`/`iat` từ token: cả hai đều do ta phát ra, chấp
                # nhận chúng từ bên ngoài chỉ mở thêm bề mặt tấn công.
                "verify_nbf": False,
                "verify_iat": False,
            },
        )
    except InvalidTokenError:
        # Bao gồm: sai chữ ký, hết hạn, sai `alg`, token vỡ, `aud`/`iss` lạ...
        return None

    if not isinstance(payload, dict):
        return None

    # Tự kiểm tra những thứ `PyJWT` đã bỏ qua (vì ta tắt `verify_sub`).
    sub = payload.get("sub")
    if not isinstance(sub, int) or isinstance(sub, bool):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return None
    # Khi `verify_exp=False` thì thư viện **không** kiểm tra — ta kiểm tra thủ công.
    if verify_exp and exp <= int(time.time()):
        return None

    return payload


def issue_access_token(
    user_id: int, *, ttl_seconds: int = ACCESS_TTL_SECONDS, jti: str | None = None
) -> tuple[str, int]:
    """Trả `(token, expires_in)`. `jti` là id để thu hồi về sau (Day 26+)."""
    expires_in = ttl_seconds
    token = encode(
        {
            "sub": int(user_id),
            "type": "access",
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "jti": jti or secrets.token_hex(8),
        }
    )
    return token, expires_in


def issue_refresh_token(
    user_id: int, *, ttl_seconds: int = REFRESH_TTL_SECONDS
) -> tuple[str, int]:
    """Refresh token dài hạn — **chỉ** dùng để đổi lấy access token mới."""
    expires_in = ttl_seconds
    token = encode(
        {
            "sub": int(user_id),
            "type": "refresh",
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            # Refresh token phải **khác** access token kể cả khi payload khác.
            "jti": secrets.token_hex(16),
        }
    )
    return token, expires_in


def hash_token(token: str) -> str:
    """Băm refresh token trước khi lưu.

    Lưu token **thô** nghĩa là kẻ đọc được DB là có token đăng nhập được. Giống
    mật khẩu, chỉ giữ hash — sha256 là đủ ở đây vì token đã là chuỗi ngẫu nhiên
    dài (không cần bcrypt chậm).
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
