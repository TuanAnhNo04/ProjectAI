"""`/v1/auth/*` — đăng ký và đăng nhập.

## Bảo mật (xem `AGENTS.md` mục 3)
- Mật khẩu được **băm bcrypt** trước khi chạm kho; không bao giờ lưu plaintext.
- Response **không** chứa `password`/`password_hash` (chỉ trả `user` + `token`).
- Sai tài khoản **và** sai mật khẩu trả **cùng một** thông báo — không để lộ
  tài khoản nào tồn tại.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Request, Response

from app.deps import current_user_id, get_repository
from app.errors import (
    CODE_RATE_LIMITED,
    CODE_UNAUTHORIZED,
    CODE_VALIDATION_ERROR,
    APIError,
)
from app.jwt_tokens import (
    ACCESS_TTL_SECONDS,
    decode,
    hash_token,
    issue_access_token,
    issue_refresh_token,
)
from app.observability import audit, get_usage_tracker
from app.rate_limit import (
    RateLimitExceeded,
    get_limiter,
    login_rate_key,
)
from app.sessions import get_session_store
from app.schemas import (
    AuthResponse,
    DeleteAccountRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.security import (
    PasswordTooLongError,
    hash_password,
    verify_password,
)
from app.sqlite_repository import SQLiteRepository, _now_ts

router = APIRouter(prefix="/auth", tags=["auth"])

#: Thông báo **cố ý giống nhau** cho cả 2 trường hợp (tránh dò tài khoản).
_GENERIC_LOGIN_ERROR = "Sai tên đăng nhập hoặc mật khẩu."

#: IP client. Ưu tiên `X-Forwarded-For` (khi chạy sau reverse proxy) rồi mới tới
#: host của request — **không** tin `X-Forwarded-For` một cách mù quáng khi
#: chưa cấu hình proxy, vì client có thể tự giả header để né rate limit.
_TRUSTED_PROXY_HEADERS = {"x-forwarded-for", "x-real-ip"}


def _client_host(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    # TestClient không có IP thật -> dùng một khoá cố định để test ổn định.
    return host or "unknown"


def _username_for_user(repo: SQLiteRepository, user_id: int) -> str:
    """Tra `user_id` -> `username` để đối chiếu với claim `sub` trong token."""
    row = repo._conn.execute(  # noqa: SLF001  (kho adapter, chấp nhận)
        "SELECT username FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return row["username"] if row else ""


def _issue_session(
    repo: SQLiteRepository, user: UserResponse, family_id: str = ""
) -> AuthResponse:
    """Cấp cặp token (access ngắn hạn + refresh dài hạn) và lưu refresh (dạng hash).

    `family_id` quyết định token mới thuộc chuỗi nào (quyết định N1):
    - Rỗng -> tạo chuỗi **mới** (đăng nhập/đăng ký).
    - Có id -> nối vào chuỗi cũ (đổi token sau xoay).
    """
    if not family_id:
        family_id = secrets.token_urlsafe(16)

    access_token, expires_in = issue_access_token(user.id)
    refresh_token, refresh_ttl = issue_refresh_token(user.id)

    repo.save_refresh_token(
        hash_token(refresh_token), user.id, _now_ts() + refresh_ttl, family_id
    )

    return AuthResponse(
        user=user,
        token=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            refresh_expires_in=refresh_ttl,
        ),
    )


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=201,
    summary="Đăng ký tài khoản mới",
)
def register(
    body: RegisterRequest,
    request: Request,
    repo: SQLiteRepository = Depends(get_repository),
) -> AuthResponse:
    # Trùng tên -> `SQLiteRepository` ném `APIError(409, conflict)`.
    try:
        password_hash = hash_password(body.password)
    except PasswordTooLongError as exc:
        # Quyết định N3: **từ chối** mật khẩu quá 72 byte thay vì để bcrypt cắt
        # cụt (cắt cụt làm hai mật khẩu khác nhau cùng hợp lệ).
        raise APIError(
            status_code=422,
            code=CODE_VALIDATION_ERROR,
            message=str(exc),
        ) from exc

    user = repo.create_user(body.username, password_hash, body.email)
    return _issue_session(repo, user)


@router.post("/login", response_model=AuthResponse, summary="Đăng nhập")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    repo: SQLiteRepository = Depends(get_repository),
) -> AuthResponse:
    # Rate limit TRƯỚC khi kiểm tra mật khẩu — kể cả thành công cũng tiêu 1 lượt,
    # nên kẻ dò mật khẩu không đoán được mật khẩu đúng có khác hành vi không.
    key = login_rate_key(body.username, _client_host(request))
    limiter = get_limiter()
    try:
        limiter.check(key)
    except RateLimitExceeded as exc:
        # Header `Retry-After` để client biết chờ bao lâu.
        raise APIError(
            status_code=429,
            code=CODE_RATE_LIMITED,
            message="Thử đăng nhập quá nhiều lần. Vui lòng thử lại sau.",
            detail={"retry_after_seconds": exc.retry_after_seconds},
        ) from exc

    user = repo.get_user_by_username(body.username)
    stored_hash = repo.get_password_hash(body.username)

    if user is None or stored_hash is None or not verify_password(body.password, stored_hash):
        audit("login_failure", "denied")
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message=_GENERIC_LOGIN_ERROR,
        )

    # Thành công -> xoá bộ đếm (đừng phạt người dùng nhập đúng).
    limiter.reset(key)
    audit("login_success", "ok")
    response.headers["X-RateLimit-Remaining"] = str(limiter.attempts_left(key))
    return _issue_session(repo, user)


@router.post("/refresh", response_model=AuthResponse, summary="Đổi refresh token")
def refresh(
    body: RefreshRequest,
    repo: SQLiteRepository = Depends(get_repository),
) -> AuthResponse:
    """Đổi refresh token lấy access token mới.

    Refresh token **đã dùng sẽ bị thu hồi ngay** (rotation). Nếu token đã bị
    thu hồi lại được dùng lần hai => đó là dấu hiệu **bị đánh cắp**, và ta
    **thu hồi toàn bộ chuỗi token** của lần đăng nhập đó (quyết định N1).
    """
    token_hash = hash_token(body.refresh_token)
    row = repo.get_refresh_token(token_hash)

    # Kiểm tra tất cả điều kiện và trả **cùng** một lỗi -> không lộ thông tin.
    if row is None or row["expires_at"] <= _now_ts():
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Refresh token không hợp lệ hoặc đã hết hạn.",
        )

    if row["revoked"]:
        # ⚠️ Token hợp lệ nhưng **đã bị thu hồi** -> chỉ có thể là tái sử dụng
        # (nạn nhân đã xoay nó, hoặc kẻ tấn công đã dùng trước). Thu hồi cả
        # chuỗi để vô hiệu hoá **mọi** token còn sót của lần đăng nhập này.
        family_id = row["family_id"]
        if family_id:
            repo.revoke_token_family(family_id)
        audit("refresh_token_reuse", "denied")
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Refresh token không hợp lệ hoặc đã hết hạn.",
        )

    payload = decode(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Refresh token không hợp lệ hoặc đã hết hạn.",
        )

    user = repo.get_user_by_username(
        _username_for_user(repo, row["user_id"])
    )
    if user is None:
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Refresh token không hợp lệ hoặc đã hết hạn.",
        )

    # Rotation: thu hồi token cũ trước khi cấp cặp mới. Token mới **nối vào
    # cùng chuỗi** (family) để nếu sau này phát hiện tái sử dụng thì thu hồi
    # được cả chuỗi, kể cả token chưa tới lượt dùng.
    repo.revoke_refresh_token(token_hash)
    return _issue_session(repo, user, row["family_id"])


@router.post("/logout", status_code=204, summary="Đăng xuất (thu hồi phiên)")
def logout(
    body: RefreshRequest,
    repo: SQLiteRepository = Depends(get_repository),
) -> None:
    """Thu hồi 1 refresh token.

    Trả 204 kể cả khi token không tồn tại — client không cần phân biệt.
    """
    repo.revoke_refresh_token(hash_token(body.refresh_token))


@router.get("/me", response_model=UserResponse, summary="Thông tin tài khoản đang đăng nhập")
def me(
    user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> UserResponse:
    """Trả dữ liệu của **chính người gọi** (id lấy từ token, không nhận từ client).

    App gọi khi khôi phục phiên: hiện tên người dùng và xác nhận phiên còn hợp lệ. Token
    còn hạn nhưng tài khoản đã bị xoá thì trả 401 (không coi là còn phiên).
    """
    user = repo.get_user_by_id(user_id)
    if user is None:
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Token không hợp lệ hoặc đã hết hạn.",
        )
    return user


@router.post(
    "/delete-account",
    status_code=204,
    summary="Xoá tài khoản và toàn bộ dữ liệu (không thể hoàn tác)",
)
def delete_account(
    body: DeleteAccountRequest,
    user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> None:
    """Quyền được quên (quyết định D5): xoá user, lịch sử, phiên đăng nhập, hội thoại.

    ⚠️ Cần **cả** token **và** mật khẩu: access token sống 15 phút, nếu bị lấy trộm thì
    kẻ cắp không được phép xoá vĩnh viễn tài khoản chỉ bằng token. Thử sai mật khẩu bị
    giới hạn tốc độ như đăng nhập; sai thì trả cùng một thông báo chung.

    Hồ sơ sức khoẻ và lịch nhắc nằm **trong máy người dùng** (xem `docs/PRIVACY.md`),
    server không giữ nên không có gì để xoá ở đó.
    """
    # Khoá theo `user_id` (đã xác thực), không theo IP: chặn dò mật khẩu từ nhiều IP.
    key = f"delete-account:{user_id}"
    limiter = get_limiter()
    try:
        limiter.check(key)
    except RateLimitExceeded as exc:
        audit("account_delete_denied", "denied")
        raise APIError(
            status_code=429,
            code=CODE_RATE_LIMITED,
            message="Thử quá nhiều lần. Vui lòng thử lại sau.",
            detail={"retry_after_seconds": exc.retry_after_seconds},
        ) from exc

    username = _username_for_user(repo, user_id)
    stored_hash = repo.get_password_hash(username) if username else None
    if stored_hash is None or not verify_password(body.password, stored_hash):
        audit("account_delete_denied", "denied")
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Mật khẩu không đúng.",
        )

    repo.delete_user_and_data(user_id)
    get_session_store().clear_user(user_id)
    get_usage_tracker().forget(user_id)
    limiter.reset(key)
    audit("account_deleted", "ok")


@router.post(
    "/validate-password",
    summary="Kiểm tra độ mạnh mật khẩu (không cần token)",
)
def validate_password(body: LoginRequest) -> dict[str, bool]:
    """Kiểm tra độ mạnh mật khẩu phía server (Day 16 dùng khi siết luật).

    Endpoint công khai: không đụng kho dữ liệu, không ghi log.
    """
    password = body.password
    problems = {
        "long_enough": len(password) >= 8,
        "has_digit": any(c.isdigit() for c in password),
        "has_letter": any(c.isalpha() for c in password),
    }
    if not problems["long_enough"]:
        # Không trả lý do chi tiết cho từng quy tắc khác vì có thể dò được
        # chính sách; chỉ cần biết là chưa đủ mạnh.
        raise APIError(
            status_code=422,
            code=CODE_VALIDATION_ERROR,
            message="Mật khẩu chưa đủ mạnh.",
            detail=problems,
        )
    return {"ok": True}