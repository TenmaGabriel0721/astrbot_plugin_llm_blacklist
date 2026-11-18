"""Regression coverage for blacklist lifetime, isolation and durable writes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from astrbot_plugin_llm_blacklist import blacklist as store_module
from astrbot_plugin_llm_blacklist.blacklist import (
    BanContext,
    BanRecord,
    BlacklistStore,
    scope_target,
)


@pytest.fixture
def context() -> BanContext:
    return BanContext(
        "platform-a", "bot-a", "group-a", "platform-a:GroupMessage:group-a"
    )


@pytest.fixture
def clock(monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(store_module.time, "time", lambda: now[0])
    return now


def record(
    context: BanContext,
    *,
    user_id: str = "user-a",
    scope: str = "global",
    expiry: float | None = 2_000.0,
) -> BanRecord:
    scope, scope_id = scope_target(context, scope)
    return BanRecord(
        user_id=user_id,
        platform_id=context.platform_id,
        bot_id=context.bot_id,
        scope=scope,
        scope_id=scope_id,
        reason="reason",
        created_at=1_000.0,
        expires_at=expiry,
        operator="admin-a",
        source="admin",
    )


def test_permanent_survives_restart_and_requires_explicit_unban(
    tmp_path, context, clock
):
    path = tmp_path / "blacklist.json"
    store = BlacklistStore(path)
    store.ban(record(context, expiry=None))
    clock[0] += 50 * 365 * 86400

    restarted = BlacklistStore(path)
    assert restarted.get_active("user-a", context).expires_at is None
    assert json.loads(path.read_text())["records"][0]["expires_at"] is None
    assert restarted.unban("user-a", context) == 1
    assert BlacklistStore(path).get_active("user-a", context) is None


def test_repeated_ban_never_shortens_or_downgrades(tmp_path, context, clock):
    store = BlacklistStore(tmp_path / "blacklist.json")
    original = store.ban(record(context, expiry=4_000.0))
    clock[0] += 10
    updated = store.ban(
        replace(record(context, expiry=1_800.0), created_at=clock[0], reason="updated")
    )
    assert updated.expires_at == original.expires_at
    assert updated.created_at == original.created_at
    assert updated.reason == "updated"

    store.ban(record(context, expiry=None))
    downgraded = store.ban(record(context, expiry=1_900.0))
    assert downgraded.expires_at is None
    assert downgraded.created_at == original.created_at


def test_scopes_remain_active_when_settings_change_and_unban_is_local(
    tmp_path, context, clock
):
    store = BlacklistStore(tmp_path / "blacklist.json")
    other_group = replace(
        context, group_id="group-b", session_id="platform-a:GroupMessage:group-b"
    )
    store.ban(record(context, scope="global", expiry=5_000.0))
    store.ban(record(context, scope="group", expiry=None))
    store.ban(record(context, scope="session", expiry=4_000.0))
    store.ban(record(other_group, scope="group", expiry=None))

    assert store.get_active("user-a", context).expires_at is None
    assert len(store.list_records(context)) == 4
    assert len(store.list_records(context, applicable_only=True)) == 3
    assert store.unban("user-a", context) == 3
    assert store.get_active("user-a", context) is None
    assert store.get_active("user-a", other_group).expires_at is None


@pytest.mark.parametrize("change", [{"platform_id": "platform-b"}, {"bot_id": "bot-b"}])
def test_same_user_and_group_are_isolated_between_bots(
    tmp_path, context, clock, change
):
    store = BlacklistStore(tmp_path / "blacklist.json")
    store.ban(record(context, scope="group", expiry=None))
    other_bot = replace(context, **change)
    assert store.get_active("user-a", other_bot) is None
    assert store.list_records(other_bot) == []
    assert store.unban("user-a", other_bot) == 0
    assert store.get_active("user-a", context) is not None


def test_group_scope_in_private_chat_uses_session(context):
    private = replace(
        context, group_id="", session_id="platform-a:FriendMessage:user-a"
    )
    assert scope_target(private, "group") == ("session", private.session_id)
    with pytest.raises(ValueError, match="Session scope"):
        scope_target(replace(private, session_id=""), "session")


def test_expiry_boundary_and_reads_do_not_write(tmp_path, context, clock, monkeypatch):
    store = BlacklistStore(tmp_path / "blacklist.json")
    store.ban(record(context, expiry=1_060.0))
    clock[0] = 1_060.0

    def unexpected_write(*_):
        raise AssertionError("Read-only lookup wrote the database")

    monkeypatch.setattr(store, "_write_records", unexpected_write)
    assert store.get_active("user-a", context) is None
    assert store.list_records(context) == []


def test_legacy_migration_keeps_scopes_permanence_and_original_file(
    tmp_path, context, clock
):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "scalar-user": "2000",
                "global:global:global-user": {
                    "user_id": "global-user",
                    "scope": "global",
                    "scope_id": "global",
                    "unblock_time": 3_000,
                },
                "group:group-a:group-user": {
                    "user_id": "group-user",
                    "scope": "group",
                    "scope_id": "group-a",
                    "unblock_time": 4_000,
                },
                "private-user": {
                    "user_id": "private-user",
                    "scope": "private",
                    "scope_id": context.session_id,
                    "unblock_time": None,
                },
                "permanent-user": {"user_id": "permanent-user", "permanent": True},
                "expired-user": 999,
            }
        )
    )
    original = legacy.read_bytes()
    path = tmp_path / "new.json"
    store = BlacklistStore(path, legacy)

    assert store.legacy_imported is True
    assert len(store.list_records(context)) == 5
    assert store.get_active("scalar-user", context) is not None
    assert store.get_active("group-user", context) is not None
    assert store.get_active("group-user", replace(context, group_id="group-b")) is None
    assert store.get_active("private-user", context).scope == "session"
    assert store.get_active("private-user", context).expires_at is None
    assert store.get_active("permanent-user", context).expires_at is None
    assert store.get_active("expired-user", context) is None
    assert legacy.read_bytes() == original


def test_empty_destination_prevents_reimport_after_unban(tmp_path, context, clock):
    path, legacy = tmp_path / "new.json", tmp_path / "legacy.json"
    legacy.write_text('{"user-a": 3000}')
    store = BlacklistStore(path, legacy)
    assert store.unban("user-a", context) == 1
    assert BlacklistStore(path, legacy).legacy_imported is False
    assert BlacklistStore(path, legacy).get_active("user-a", context) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"user-a": None},
        {"user-a": {}},
        {"user-a": {"permanent": "true"}},
        {"user-a": float("inf")},
        {"user-a": {"user_id": None, "unblock_time": None}},
    ],
)
def test_malformed_legacy_data_is_not_interpreted_as_permanent(tmp_path, payload):
    path, legacy = tmp_path / "new.json", tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload))
    original = legacy.read_bytes()
    with pytest.raises(ValueError):
        BlacklistStore(path, legacy)
    assert not path.exists()
    assert legacy.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"schema_version": 2, "records": []}',
        '{"schema_version": true, "records": []}',
        '{"schema_version": 1.0, "records": []}',
        '{"schema_version": 1, "records": [{}]}',
    ],
)
def test_malformed_modern_data_is_preserved(tmp_path, payload):
    path = tmp_path / "blacklist.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        BlacklistStore(path)
    assert path.read_text() == payload


@pytest.mark.parametrize(
    "expiry", [float("inf"), float("nan"), -1, True, "2000", 10**400]
)
def test_invalid_modern_expiry_is_rejected(context, expiry):
    with pytest.raises(ValueError):
        record(context, expiry=expiry)


@pytest.mark.parametrize("operation", ["ban", "unban"])
def test_replace_failure_preserves_disk_and_memory(
    tmp_path, context, clock, monkeypatch, operation
):
    path = tmp_path / "blacklist.json"
    store = BlacklistStore(path)
    store.ban(record(context, expiry=None))
    original = path.read_bytes()

    def fail_replace(*_):
        raise OSError("replace failed")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        if operation == "ban":
            store.ban(record(context, user_id="user-b"))
        else:
            store.unban("user-a", context)

    assert path.read_bytes() == original
    assert store.get_active("user-a", context).expires_at is None
    assert store.get_active("user-b", context) is None
    assert list(tmp_path.glob("*.tmp")) == []


def test_partial_write_failure_preserves_previous_state(
    tmp_path, context, clock, monkeypatch
):
    path = tmp_path / "blacklist.json"
    store = BlacklistStore(path)
    store.ban(record(context, expiry=None))
    original = path.read_bytes()

    def fail_dump(_payload, output, **_kwargs):
        output.write("{partial")
        raise OSError("write failed")

    monkeypatch.setattr(store_module.json, "dump", fail_dump)
    with pytest.raises(OSError, match="write failed"):
        store.ban(record(context, user_id="user-b"))

    assert path.read_bytes() == original
    assert len(store.list_records(context)) == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_parallel_mutations_do_not_lose_records(tmp_path, context, clock):
    path = tmp_path / "blacklist.json"
    store = BlacklistStore(path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                store.ban,
                [record(context, user_id=f"user-{i}", expiry=None) for i in range(12)],
            )
        )
    assert len(results) == 12
    assert len(store.list_records(context)) == 12
    assert len(BlacklistStore(path).list_records(context)) == 12
