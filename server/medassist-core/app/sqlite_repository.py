"""Adapter SQLite — dùng **chỉ module `sqlite3` của thư viện chuẩn**.

Không cài thêm package (quy tắc bảo mật của dự án). Chỉ đủ cho Day 15; khi có
Postgres sẽ thay bằng `PostgresRepository` cùng interface `Repository`.

Về bảo mật:
- **Mọi giá trị đều truyền qua tham số `?`** — không ghép chuỗi SQL, nên không
  có SQL injection.
- Mật khẩu lưu ở dạng **hash** (bcrypt), không bao giờ lưu plaintext.
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.errors import CODE_CONFLICT, APIError
from app.schemas import (
    HistoryItemResponse,
    MedicineResponse,
    SearchResponse,
    UserResponse,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    email         TEXT,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medicines (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    normalized_name  TEXT NOT NULL,
    composition      TEXT,
    uses             TEXT,
    side_effects     TEXT,
    manufacturer     TEXT,
    ingredients      TEXT,
    excellent_review REAL,
    average_review   REAL,
    poor_review      REAL,
    source           TEXT,
    is_authoritative INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_medicines_normalized ON medicines(normalized_name);
CREATE INDEX IF NOT EXISTS idx_medicines_ingredients ON medicines(ingredients);

CREATE TABLE IF NOT EXISTS history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id),
    filepath       TEXT NOT NULL,
    search_results TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id);

-- Refresh token: chỉ lưu **hash**, không lưu token thô (xem jwt_tokens.hash_token).
-- `family_id`: mọi token sinh ra từ **cùng một lần đăng nhập** dùng chung id.
-- Khi phát hiện tái sử dụng token đã xoay => thu hồi cả family (quyết định N1).
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    expires_at INTEGER NOT NULL,
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    family_id  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_family ON refresh_tokens(family_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> int:
    """Unix timestamp (giây) — dùng cho `expires_at`."""
    return int(datetime.now(timezone.utc).timestamp())


#: Ký tự Đông Nam Á **không tách được** bằng NFD (Đ là một code point riêng,
#: không phải "D" + dấu). Thiếu bảng này thì tìm "đau mỗi" sẽ KHÔNG ra
#: "Dau Moi" — đúng loại lỗi người dùng Việt gặp phải ngay.
_FOLD_EXTRA = str.maketrans(
    {
        "đ": "d",  # đ
        "Đ": "d",  # Đ
        "ø": "o",  # ø
        "Ø": "O",  # Ø
        "ł": "l",  # ł
        "Ł": "L",  # Ł
        "æ": "ae",  # æ
        "Æ": "AE",  # Æ
        "œ": "oe",  # œ
        "Œ": "OE",  # Œ
        "ß": "ss",  # ß
        "þ": "th",  # þ
        "Þ": "TH",  # Þ
        "ð": "d",  # ð
        "Ð": "D",  # Ð
    }
)


def normalize(text: str) -> str:
    """Bỏ dấu + hạ chữ thường + gộp khoảng trắng.

    Nhờ vậy gõ "Panadol" / "panadol" / "PANADOL" đều ra kết quả như nhau — và
    người dùng Việt gõ không dấu vẫn tìm được ("dau moi" -> "đau mỗi").

    ⚠️ Bắt buộc phải có bảng `_FOLD_EXTRA`: `unicodedata.normalize("NFD", "Đ")`
    **không** cho ra "D" vì `Đ` là một chữ cái riêng, không phải chữ cái Latin
    có dấu. Bỏ bảng này thì mọi từ bắt đầu bằng "Đ/đ" đều không tìm được.
    """
    folded = text.translate(_FOLD_EXTRA)
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", folded) if not unicodedata.combining(c)
    )
    return " ".join(stripped.lower().split())


def _to_user(row: sqlite3.Row) -> UserResponse:
    return UserResponse(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _to_medicine(row: sqlite3.Row) -> MedicineResponse:
    return MedicineResponse(
        name=row["name"],
        composition=row["composition"],
        uses=row["uses"],
        side_effects=row["side_effects"],
        manufacturer=row["manufacturer"],
        # ⚠️ Lưu dạng **JSON array**, không nối bằng dấu cách: hoạt chất có
        # thể chứa khoảng trắng ("paracetamol 500mg", "vitamin c") — nối rồi
        # tách lại sẽ vỡ vụn.
        ingredients=json.loads(row["ingredients"]) if row["ingredients"] else [],
        excellent_review_percent=row["excellent_review"],
        average_review_percent=row["average_review"],
        poor_review_percent=row["poor_review"],
        source=row["source"],
        is_authoritative=bool(row["is_authoritative"]),
        updated_at=(
            datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        ),
    )


def _to_history(row: sqlite3.Row) -> HistoryItemResponse:
    return HistoryItemResponse(
        id=row["id"],
        filepath=row["filepath"],
        search_results=[
            MedicineResponse(**item) for item in json.loads(row["search_results"])
        ],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class SQLiteRepository:
    """Cài đặt interface `Repository` bằng SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # `foreign_keys` mặc định **TẮT** trong SQLite -> phải bật thủ công.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Thêm cột còn thiếu cho DB đã tạo từ phiên bản cũ.

        ⚠️ `CREATE TABLE IF NOT EXISTS` **không** thêm cột cho bảng đã tồn tại,
        nên file SQLite cũ sẽ thiếu `family_id` và mọi truy vấn có cột đó sẽ
        ném lỗi. Ở đây dùng cách idempotent: kiểm tra `PRAGMA table_info` rồi
        `ALTER TABLE ADD COLUMN` nếu thiếu.

        (Đây là migration tối giản cho SQLite. Khi chuyển sang Postgres sẽ
        thay bằng file migration có version.)
        """
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(refresh_tokens)")
        }
        if "family_id" not in columns:
            self._conn.execute(
                "ALTER TABLE refresh_tokens ADD COLUMN family_id TEXT NOT NULL DEFAULT ''"
            )

    def close(self) -> None:
        self._conn.close()

    # -- Người dùng --

    def create_user(self, username: str, password_hash: str, email: str) -> UserResponse:
        try:
            cur = self._conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at)"
                " VALUES (?, ?, ?, ?)",
                (username, email, password_hash, _now()),
            )
        except sqlite3.IntegrityError as exc:
            # UNIQUE(username) bị vi phạm -> trùng tên, KHÔNG phải lỗi 500.
            raise APIError(
                status_code=409,
                code=CODE_CONFLICT,
                message="Tên đăng nhập đã tồn tại.",
            ) from exc
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM users WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return _to_user(row)

    def get_user_by_username(self, username: str) -> UserResponse | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return _to_user(row) if row else None

    def get_password_hash(self, username: str) -> str | None:
        """Lấy hash để xác thực. Route dùng, **không** trả ra ngoài."""
        row = self._conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row["password_hash"] if row else None

    # -- Thuốc --

    def _search(self, query: str | None, limit: int, offset: int) -> SearchResponse:
        if query:
            # Tìm theo **tên hoặc hoạt chất** (Day 18). `LIKE` + tham số `?` ->
            # an toàn với `%`, `_`, `'` của người dùng vì chúng **không** được
            # nội suy vào SQL.
            pattern = f"%{normalize(query)}%"
            where = (
                "WHERE normalized_name LIKE ? OR IFNULL(ingredients, '') LIKE ?"
            )
            args = (pattern, pattern)
            rows = self._conn.execute(
                f"SELECT * FROM medicines {where} ORDER BY name LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            total = self._conn.execute(
                f"SELECT COUNT(*) AS c FROM medicines {where}", args
            ).fetchone()["c"]
        else:
            rows = self._conn.execute(
                "SELECT * FROM medicines ORDER BY name LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = self._conn.execute("SELECT COUNT(*) AS c FROM medicines").fetchone()["c"]
        return SearchResponse(
            results=[_to_medicine(r) for r in rows], query=query, total=total
        )

    def search_medicines(self, query: str, limit: int, offset: int) -> SearchResponse:
        return self._search(query, limit, offset)

    def list_medicines(self, limit: int, offset: int) -> SearchResponse:
        return self._search(None, limit, offset)

    def add_medicines(self, items: list[dict[str, Any]]) -> int:
        """Nạp danh mục thuốc (Day 18 dùng; Day 15 có sẵn để seed).

        Chấp nhận dict **thiếu trường** — khoá thiếu sẽ điền `None`. Nếu bắt
        buộc đủ 11 khoá thì mỗi nguồn dữ liệu phải viết lại tên cột không
        đồng nhất, dễ sinh lỗi "thiếu tham số" khó hiểu.
        """
        fields = (
            "composition",
            "uses",
            "side_effects",
            "manufacturer",
            "source",
        )
        self._conn.executemany(
            "INSERT INTO medicines (name, normalized_name, composition, uses,"
            " side_effects, manufacturer, ingredients, excellent_review,"
            " average_review, poor_review, source, is_authoritative, updated_at)"
            " VALUES (:name, :normalized_name, :composition, :uses,"
            " :side_effects, :manufacturer, :ingredients, :excellent_review,"
            " :average_review, :poor_review, :source, :is_authoritative,"
            " :updated_at)",
            [
                {
                    **item,
                    # Điền khoá còn thiếu để câu INSERT không thiếu tham số.
                    **{key: item.get(key) for key in fields},
                    "ingredients": item.get("ingredients", ""),
                    "excellent_review": item.get("excellent_review"),
                    "average_review": item.get("average_review"),
                    "poor_review": item.get("poor_review"),
                    # Dữ liệu dev mặc định KHÔNG phải nguồn lâm sàng.
                    "is_authoritative": int(bool(item.get("is_authoritative", False))),
                    "normalized_name": normalize(str(item.get("name", ""))),
                    "updated_at": item.get("updated_at") or _now(),
                }
                for item in items
            ],
        )
        self._conn.commit()
        return len(items)

    # -- Lịch sử (luôn lọc theo user_id) --

    def add_history(
        self, user_id: int, filepath: str, medicines: list[MedicineResponse]
    ) -> HistoryItemResponse:
        cur = self._conn.execute(
            "INSERT INTO history (user_id, filepath, search_results, created_at)"
            " VALUES (?, ?, ?, ?)",
            (
                user_id,
                filepath,
                json.dumps([m.model_dump(mode="json") for m in medicines]),
                _now(),
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM history WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return _to_history(row)

    def list_history(
        self, user_id: int, limit: int, offset: int
    ) -> list[HistoryItemResponse]:
        rows = self._conn.execute(
            "SELECT * FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
        return [_to_history(r) for r in rows]

    def count_history(self, user_id: int) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) AS c FROM history WHERE user_id = ?", (user_id,)
        ).fetchone()["c"]

    def delete_history(self, user_id: int, item_id: int) -> bool:
        # ⚠️ `user_id` nằm **trong** điều kiện -> không thể xoá nhầm của người khác.
        cur = self._conn.execute(
            "DELETE FROM history WHERE id = ? AND user_id = ?", (item_id, user_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # -- Refresh token (chỉ lưu hash) --

    def save_refresh_token(
        self,
        token_hash: str,
        user_id: int,
        expires_at: int,
        family_id: str = "",
    ) -> None:
        """Lưu hash refresh token kèm `family_id` (chuỗi đặt token)."""
        self._conn.execute(
            "INSERT INTO refresh_tokens (token_hash, user_id, expires_at, revoked,"
            " created_at, family_id) VALUES (?, ?, ?, 0, ?, ?)",
            (token_hash, user_id, expires_at, _now(), family_id),
        )
        self._conn.commit()

    def get_refresh_token(self, token_hash: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM refresh_tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()

    def revoke_refresh_token(self, token_hash: str) -> bool:
        """Thu hồi 1 refresh token (dùng khi logout hoặc đổi token)."""
        cur = self._conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ? AND revoked = 0",
            (token_hash,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def revoke_token_family(self, family_id: str) -> int:
        """Thu hồi **toàn bộ** token cùng chuỗi; trả số token vừa thu hồi.

        ⚠️ Gọi khi phát hiện **tái sử dụng** refresh token đã xoay (quyết định N1).
        Mục đích: nếu kẻ tấn công đã chôm được token, chỉ thu hồi token vừa dùng
        là **không đủ** — token chôm có thể là token *chưa* tới lượt dùng và vẫn
        còn hiệu lực. Thu hồi cả chuỗi buộc cả hai bên phải đăng nhập lại, nên
        chiến lại "dùng trước, hợp pháp hoá sau" không còn tác dụng.

        Lưu ý: cố ý **không** xoá hàng — giữ dấu vết để điều tra.
        """
        if not family_id:
            return 0
        cur = self._conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE family_id = ? AND revoked = 0",
            (family_id,),
        )
        self._conn.commit()
        return cur.rowcount

    def count_active_family_tokens(self, family_id: str) -> int:
        """Số token còn hiệu lực trong chuỗi — dùng cho test/điều tra."""
        if not family_id:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM refresh_tokens"
            " WHERE family_id = ? AND revoked = 0 AND expires_at > ?",
            (family_id, _now_ts()),
        ).fetchone()
        return int(row["n"])

    def revoke_all_refresh_tokens(self, user_id: int) -> int:
        """Thu hồi **mọi** phiên của user (đăng xuất khỏi mọi thiết bị)."""
        cur = self._conn.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ? AND revoked = 0",
            (user_id,),
        )
        self._conn.commit()
        return cur.rowcount

    def purge_expired_refresh_tokens(self) -> int:
        """Dọn token đã hết hạn. Chạy định kỳ, không bắt buộc đúng đắn."""
        cur = self._conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?", (int(_now_ts()),)
        )
        self._conn.commit()
        return cur.rowcount