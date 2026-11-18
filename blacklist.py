"""Bot-scoped blacklist records and atomic JSON persistence."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any

RecordKey = tuple[str, str, str, str, str]
SCOPES = frozenset({"global", "group", "session"})


def _timestamp(value: Any, field: str) -> float:
    """Validate an ordinary JSON timestamp.

    Args:
        value: Numeric timestamp to validate.
        field: Field name used in errors.

    Returns:
        A finite, nonnegative timestamp.

    Raises:
        ValueError: If the value is not a valid timestamp.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    try:
        timestamp = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite and nonnegative") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return timestamp


@dataclass(frozen=True)
class BanContext:
    """Identity of a bot and the current conversation."""

    platform_id: str
    bot_id: str
    group_id: str
    session_id: str


@dataclass(frozen=True)
class BanRecord:
    """A temporary or permanent restriction for one user and scope."""

    user_id: str
    platform_id: str
    bot_id: str
    scope: str
    scope_id: str
    reason: str
    created_at: float
    expires_at: float | None
    operator: str
    source: str

    def __post_init__(self) -> None:
        """Reject invalid records before they can enter persistent storage.

        Raises:
            ValueError: If a record field is invalid.
        """
        for field in (
            "user_id",
            "platform_id",
            "bot_id",
            "scope",
            "scope_id",
            "reason",
            "operator",
            "source",
        ):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"{field} must be a string")
        if not self.user_id.strip():
            raise ValueError("user_id must not be empty")
        if self.scope not in SCOPES:
            raise ValueError(f"Unknown blacklist scope: {self.scope}")
        if self.scope == "global" and self.scope_id:
            raise ValueError("Global records must have an empty scope_id")
        if self.scope != "global" and not self.scope_id:
            raise ValueError("Scoped records must have a scope_id")
        _timestamp(self.created_at, "created_at")
        if self.expires_at is not None:
            _timestamp(self.expires_at, "expires_at")

    @property
    def key(self) -> RecordKey:
        """Return the collision-free identity used for record updates.

        Returns:
            The platform, bot, scope, scope identifier and user identifier.
        """
        return (
            self.platform_id,
            self.bot_id,
            self.scope,
            self.scope_id,
            self.user_id,
        )

    def matches(self, context: BanContext, *, applicable_only: bool = True) -> bool:
        """Check whether a record belongs to a bot or its current conversation.

        Args:
            context: Bot and conversation identity to match.
            applicable_only: Also require the current group or session to match.

        Returns:
            Whether the record matches the requested context.
        """
        # Empty namespaces belong to legacy records whose bot cannot be inferred.
        if self.platform_id and self.platform_id != context.platform_id:
            return False
        if self.bot_id and self.bot_id != context.bot_id:
            return False
        if not applicable_only or self.scope == "global":
            return True
        if self.scope == "group":
            return bool(context.group_id) and self.scope_id == context.group_id
        return self.scope_id == context.session_id

    def is_active(self, now: float) -> bool:
        """Check expiry without giving permanent records an artificial deadline.

        Args:
            now: Current Unix timestamp.

        Returns:
            Whether the restriction is still active.
        """
        return self.expires_at is None or self.expires_at > now


def scope_target(context: BanContext, scope: str) -> tuple[str, str]:
    """Resolve a configured scope to a persistent scope identifier.

    Args:
        context: Identity of the current bot and conversation.
        scope: One of global, group or session.

    Returns:
        The scope and its identifier; group scope becomes session in private chat.

    Raises:
        ValueError: If the scope is unknown or lacks a required identity.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown blacklist scope: {scope}")
    if scope == "global":
        return "global", ""
    if scope == "group" and context.group_id:
        return "group", context.group_id
    if not context.session_id:
        raise ValueError("Session scope requires a session identifier")
    return "session", context.session_id


class BlacklistStore:
    """Thread-safe in-memory records whose mutations commit to disk first."""

    def __init__(self, path: Path, legacy_path: Path | None = None) -> None:
        """Load records or import the original plugin's database once.

        Args:
            path: Destination of the versioned blacklist database.
            legacy_path: Original plugin database, read only if path is absent.

        Raises:
            ValueError: If an existing database cannot be safely decoded.
            OSError: If reading or saving a database fails.
        """
        self.path = Path(path)
        self._lock = RLock()
        self.legacy_imported = False
        self._records: dict[RecordKey, BanRecord] = {}
        now = time.time()
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or type(payload.get("schema_version")) is not int
                or not isinstance(payload.get("records"), list)
            ):
                raise ValueError("Unsupported blacklist database format")
            for item in payload["records"]:
                if not isinstance(item, dict):
                    raise ValueError("Blacklist database contains a non-object record")
                try:
                    record = BanRecord(**item)
                except TypeError as exc:
                    raise ValueError(
                        "Blacklist database contains invalid fields"
                    ) from exc
                self._merge_loaded_record(record, now)
        elif legacy_path is not None and Path(legacy_path).is_file():
            payload = json.loads(Path(legacy_path).read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("Legacy blacklist database must be an object")
            for key, item in payload.items():
                if isinstance(item, dict):
                    if "permanent" in item and not isinstance(item["permanent"], bool):
                        raise ValueError("Legacy permanent flag must be a boolean")
                    if item.get("permanent") is True:
                        expires_at = None
                    elif "unblock_time" in item:
                        expires_at = item["unblock_time"]
                    elif "expires_at" in item:
                        expires_at = item["expires_at"]
                    else:
                        raise ValueError("Legacy blacklist record lacks an expiry")
                    scope = str(item.get("scope", "global"))
                    scope = "session" if scope == "private" else scope
                    scope_id = (
                        "" if scope == "global" else str(item.get("scope_id") or "")
                    )
                    fallback_user = str(key)
                    if fallback_user.startswith(
                        ("global:", "group:", "session:", "private:")
                    ):
                        fallback_user = fallback_user.rsplit(":", 1)[-1]
                    raw_user_id = item.get("user_id", fallback_user)
                    if isinstance(raw_user_id, bool) or not isinstance(
                        raw_user_id, (str, int)
                    ):
                        raise ValueError("Legacy user_id must be a string or integer")
                    user_id = str(raw_user_id).strip()
                    reason = str(item.get("reason", "旧版数据迁移"))
                    operator = str(item.get("operator", "legacy"))
                    created_at = item.get("created_at", now)
                else:
                    user_id = str(key)
                    scope, scope_id = "global", ""
                    reason, operator, created_at = "旧版数据迁移", "legacy", now
                    expires_at = item
                    if expires_at is None:
                        raise ValueError(
                            "Legacy scalar records must have a numeric expiry"
                        )
                for field, value in (
                    ("created_at", created_at),
                    ("expires_at", expires_at),
                ):
                    if value is None and field == "expires_at":
                        continue
                    if isinstance(value, bool):
                        raise ValueError(f"Legacy {field} must be numeric")
                    try:
                        converted = _timestamp(float(value), field)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            f"Legacy {field} must be a finite timestamp"
                        ) from exc
                    if field == "created_at":
                        created_at = converted
                    else:
                        expires_at = converted
                record = BanRecord(
                    user_id=user_id,
                    platform_id="",
                    bot_id="",
                    scope=scope,
                    scope_id=scope_id,
                    reason=reason,
                    created_at=created_at,
                    expires_at=expires_at,
                    operator=operator,
                    source="legacy",
                )
                self._merge_loaded_record(record, now)
            self._write_records(self._records)
            self.legacy_imported = True

    def _merge_loaded_record(self, record: BanRecord, now: float) -> None:
        """Keep the strongest active record if a database repeats its identity.

        Args:
            record: Validated record being loaded.
            now: Timestamp used to ignore expired records.
        """
        if not record.is_active(now):
            return
        previous = self._records.get(record.key)
        if previous is not None and (
            previous.expires_at is None
            or (
                record.expires_at is not None
                and previous.expires_at >= record.expires_at
            )
        ):
            return
        self._records[record.key] = record

    def _write_records(self, records: dict[RecordKey, BanRecord]) -> None:
        """Atomically replace the database without truncating the previous file.

        Args:
            records: Complete candidate state to persist.

        Raises:
            OSError: If writing or replacing the database fails.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(
                    {
                        "schema_version": 1,
                        "records": [asdict(records[key]) for key in sorted(records)],
                    },
                    output,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    def get_active(
        self, user_id: str, context: BanContext, now: float | None = None
    ) -> BanRecord | None:
        """Find the strongest active restriction across all applicable scopes.

        Args:
            user_id: User to look up.
            context: Identity of the current bot and conversation.
            now: Optional Unix timestamp, primarily for deterministic callers.

        Returns:
            A permanent record or the temporary record with the latest expiry.
        """
        now = time.time() if now is None else _timestamp(now, "now")
        with self._lock:
            matches = [
                record
                for record in self._records.values()
                if record.user_id == user_id
                and record.matches(context)
                and record.is_active(now)
            ]
        return max(
            matches,
            key=lambda record: (record.expires_at is None, record.expires_at or 0),
            default=None,
        )

    def list_records(
        self,
        context: BanContext,
        now: float | None = None,
        applicable_only: bool = False,
    ) -> list[BanRecord]:
        """List active records belonging to the current bot.

        Args:
            context: Bot and conversation identity to match.
            now: Optional Unix timestamp.
            applicable_only: Limit results to the current group or session.

        Returns:
            An independent list, with permanent records ordered first.
        """
        now = time.time() if now is None else _timestamp(now, "now")
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.matches(context, applicable_only=applicable_only)
                and record.is_active(now)
            ]
        return sorted(
            records,
            key=lambda record: (
                record.expires_at is not None,
                record.expires_at or 0,
                record.scope,
                record.scope_id,
                record.user_id,
            ),
        )

    def ban(self, record: BanRecord) -> BanRecord:
        """Persist a restriction without shortening or downgrading an active ban.

        Args:
            record: Validated restriction to apply.

        Returns:
            The actual record that was committed, including any retained expiry.

        Raises:
            ValueError: If a temporary restriction has already expired.
            OSError: If persistence fails; the previous in-memory state is kept.
        """
        now = time.time()
        if not record.is_active(now):
            raise ValueError("A temporary ban must expire in the future")
        with self._lock:
            updated = {
                key: existing
                for key, existing in self._records.items()
                if existing.is_active(now)
            }
            previous = updated.get(record.key)
            if previous is not None:
                expires_at = record.expires_at
                if previous.expires_at is None:
                    expires_at = None
                elif expires_at is not None:
                    expires_at = max(previous.expires_at, expires_at)
                record = replace(
                    record, created_at=previous.created_at, expires_at=expires_at
                )
            updated[record.key] = record
            self._write_records(updated)
            self._records = updated
            return record

    def unban(self, user_id: str, context: BanContext) -> int:
        """Remove every restriction affecting a user in the current conversation.

        Args:
            user_id: User whose restrictions should be removed.
            context: Current bot and conversation identity.

        Returns:
            The number of active records removed; other groups and bots stay intact.

        Raises:
            OSError: If persistence fails; the previous in-memory state is kept.
        """
        now = time.time()
        with self._lock:
            active = {
                key: record
                for key, record in self._records.items()
                if record.is_active(now)
            }
            updated = {
                key: record
                for key, record in active.items()
                if not (record.user_id == user_id and record.matches(context))
            }
            removed = len(active) - len(updated)
            if updated != self._records:
                self._write_records(updated)
                self._records = updated
            return removed
