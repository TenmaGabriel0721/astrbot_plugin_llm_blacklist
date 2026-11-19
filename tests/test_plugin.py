import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from astrbot.api.event import MessageEventResult
from astrbot.api.message_components import At, AtAll, Plain
from astrbot.core.pipeline.process_stage.method.star_request import StarRequestSubStage
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.provider.register import llm_tools
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.filter.permission import PermissionTypeFilter
from astrbot.core.star.star import StarMetadata, star_map
from astrbot.core.star.star_handler import (
    EventType,
    StarHandlerMetadata,
    StarHandlerRegistry,
    star_handlers_registry,
)
from astrbot_plugin_llm_blacklist import main as plugin_module
from astrbot_plugin_llm_blacklist.blacklist import BanContext, BanRecord
from astrbot_plugin_llm_blacklist.main import LLMBlacklistPlugin

CLOCK = 1_000_000.0


class FakeEvent:
    """Represent the public event API without connecting to a platform."""

    def __init__(
        self,
        user_id="actor",
        group_id="group-A",
        platform_id="platform-A",
        bot_id="bot-A",
        session_id=None,
        admin=False,
        message="黑名单",
        messages=None,
        nickname="",
        platform_name="test",
    ):
        self.message_obj = SimpleNamespace(
            sender=SimpleNamespace(user_id=user_id, nickname=nickname),
            self_id=bot_id,
            group_id=group_id,
            message=list(messages) if messages is not None else [],
        )
        self.unified_msg_origin = session_id or (
            f"{platform_id}:GroupMessage:{group_id}"
            if group_id
            else f"{platform_id}:FriendMessage:{user_id}"
        )
        self._group_id = group_id
        self._platform_id = platform_id
        self._platform_name = platform_name
        self._bot_id = bot_id
        self._admin = admin
        self.message_str = message
        self._extras = {}
        self._result = None
        self.call_llm = False
        self.stopped = False
        self.is_at_or_wake_command = True
        self.trace = SimpleNamespace(record=lambda *args, **kwargs: None)

    def get_sender_id(self):
        return self.message_obj.sender.user_id

    def get_sender_name(self):
        return self.message_obj.sender.nickname

    def get_group_id(self):
        return self._group_id

    def get_platform_id(self):
        return self._platform_id

    def get_platform_name(self):
        return self._platform_name

    def get_self_id(self):
        return self._bot_id

    def get_message_str(self):
        return self.message_str

    def get_messages(self):
        return self.message_obj.message

    def is_admin(self):
        return self._admin

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def should_call_llm(self, prohibited):
        self.call_llm = prohibited

    def stop_event(self):
        self.stopped = True
        self._result = MessageEventResult().stop_event()

    def is_stopped(self):
        return self.stopped

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def clear_result(self):
        self._result = None

    def plain_result(self, text):
        return MessageEventResult().message(text)


def identity(event):
    return BanContext(
        platform_id=event.get_platform_id(),
        bot_id=event.get_self_id(),
        group_id=event.get_group_id(),
        session_id=event.unified_msg_origin,
    )


async def invoke_blacklist_command(plugin, event, metadata):
    command_filter = next(
        event_filter
        for event_filter in metadata.event_filters
        if isinstance(event_filter, CommandFilter)
    )
    assert command_filter.filter(event, {})
    return [
        result
        async for result in plugin.blacklist_command(
            event, **event.get_extra("parsed_params")
        )
    ]


@pytest.fixture
def command_metadata():
    metadata = star_handlers_registry.get_handler_by_full_name(
        f"{plugin_module.__name__}_blacklist_command"
    )
    assert metadata is not None
    return metadata


@pytest.fixture
def plugin_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin_module.time, "time", lambda: CLOCK)

    def data_directory(name):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(plugin_module.StarTools, "get_data_dir", data_directory)

    async def create(**overrides):
        config = {
            "default_blacklist_duration": 10,
            "max_blacklist_duration": 1440,
            "strict_block_mode": True,
            "allow_llm_permanent_ban": True,
            "protected_user_ids": [],
        }
        config.update(overrides)
        context = SimpleNamespace(get_config=lambda umo=None: {"admins_id": ["owner"]})
        plugin = LLMBlacklistPlugin(context, config)
        await plugin.initialize()
        return plugin

    return create


@pytest_asyncio.fixture
async def plugin(plugin_factory):
    return await plugin_factory()


@pytest.mark.asyncio
async def test_permanent_ban_survives_restart_and_elapsed_time(
    plugin, plugin_factory, monkeypatch
):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(event, "openid-target", permanent=True)

    record = plugin.store.get_active("openid-target", identity(event))
    assert record is not None
    assert record.expires_at is None

    monkeypatch.setattr(plugin_module.time, "time", lambda: CLOCK + 50 * 365 * 86400)
    restarted = await plugin_factory()
    assert (
        restarted.store.get_active("openid-target", identity(event)).expires_at is None
    )


@pytest.mark.asyncio
async def test_default_duration_and_maximum_are_enforced(plugin):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(event, "default-target")
    await plugin.llm_blacklist_manage(event, "capped-target", duration_minutes=100_000)

    assert plugin.store.get_active("default-target", identity(event)).expires_at == (
        CLOCK + 10 * 60
    )
    assert plugin.store.get_active("capped-target", identity(event)).expires_at == (
        CLOCK + 1440 * 60
    )


@pytest.mark.asyncio
async def test_zero_duration_explicitly_removes_permanent_ban(plugin):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(event, "target", permanent=True)

    await plugin.llm_blacklist_manage(event, "target", duration_minutes=0)

    assert plugin.store.get_active("target", identity(event)) is None


@pytest.mark.asyncio
async def test_shorter_repeat_keeps_existing_temporary_expiry(plugin):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(event, "target", duration_minutes=60)

    await plugin.llm_blacklist_manage(event, "target", duration_minutes=5)

    assert plugin.store.get_active("target", identity(event)).expires_at == CLOCK + 3600


@pytest.mark.asyncio
async def test_temporary_repeat_cannot_downgrade_permanent_ban(plugin):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(event, "target", permanent=True)

    await plugin.llm_blacklist_manage(event, "target", duration_minutes=5)

    assert plugin.store.get_active("target", identity(event)).expires_at is None


@pytest.mark.asyncio
async def test_permanent_and_unban_parameters_conflict(plugin):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(
        event, "target", duration_minutes=0, permanent=True
    )

    assert plugin.store.get_active("target", identity(event)) is None


@pytest.mark.asyncio
async def test_llm_permanent_ban_can_be_disabled(plugin_factory):
    plugin = await plugin_factory(allow_llm_permanent_ban=False)
    event = FakeEvent()

    await plugin.llm_blacklist_manage(event, "target", permanent=True)

    assert plugin.store.get_active("target", identity(event)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["bot-A", "owner", "protected"])
async def test_bot_admin_and_configured_users_are_protected(plugin_factory, target):
    plugin = await plugin_factory(protected_user_ids=["protected"])
    event = FakeEvent()

    await plugin.llm_blacklist_manage(event, target, permanent=True)

    assert plugin.store.get_active(target, identity(event)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duration", [True, 1.5, float("nan"), float("inf"), "oops", None]
)
async def test_invalid_duration_does_not_mutate_blacklist(plugin, duration):
    event = FakeEvent()

    result = await plugin.llm_blacklist_manage(
        event, "target", duration_minutes=duration
    )

    assert isinstance(result, str)
    assert plugin.store.get_active("target", identity(event)) is None


@pytest.mark.asyncio
async def test_global_scope_is_separate_for_each_bot_and_platform(plugin):
    actor = FakeEvent()
    await plugin.llm_blacklist_manage(actor, "target", permanent=True)
    matching = FakeEvent(user_id="target", group_id="another-group")
    another_bot = FakeEvent(user_id="target", bot_id="bot-B")
    another_platform = FakeEvent(user_id="target", platform_id="platform-B")

    await plugin.block_blacklisted_messages(matching)
    await plugin.block_blacklisted_messages(another_bot)
    await plugin.block_blacklisted_messages(another_platform)

    assert matching.stopped
    assert not another_bot.stopped
    assert not another_platform.stopped


@pytest.mark.asyncio
async def test_existing_group_ban_can_be_read_and_unbanned_after_reload(
    plugin_factory, command_metadata
):
    plugin = await plugin_factory()
    plugin.store.ban(
        BanRecord(
            user_id="target",
            platform_id="platform-A",
            bot_id="bot-A",
            scope="group",
            scope_id="group-A",
            reason="Historical group record",
            created_at=CLOCK,
            expires_at=None,
            operator="owner",
            source="admin",
        )
    )
    plugin = await plugin_factory()
    matching = FakeEvent(user_id="target")
    unrelated = FakeEvent(user_id="target", group_id="group-B")

    await plugin.block_blacklisted_messages(matching)
    await plugin.block_blacklisted_messages(unrelated)

    assert matching.stopped
    assert not unrelated.stopped

    query = FakeEvent(message="黑名单 查询 target")
    results = await invoke_blacklist_command(plugin, query, command_metadata)
    assert "用户：target" in results[0].get_plain_text()
    assert plugin.store.get_active("target", identity(query)).scope == "group"
    unban = FakeEvent(user_id="owner", admin=True, message="解除拉黑 target")
    results = await invoke_blacklist_command(plugin, unban, command_metadata)
    assert "已解除用户 target" in results[0].get_plain_text()
    assert plugin.store.get_active("target", identity(matching)) is None


@pytest.mark.asyncio
async def test_historical_session_scope_distinguishes_branches_of_same_group(
    plugin_factory,
):
    plugin = await plugin_factory()
    plugin.store.ban(
        BanRecord(
            user_id="target",
            platform_id="platform-A",
            bot_id="bot-A",
            scope="session",
            scope_id="platform-A:GroupMessage:group-A",
            reason="Historical session record",
            created_at=CLOCK,
            expires_at=None,
            operator="owner",
            source="admin",
        )
    )
    matching = FakeEvent(user_id="target")
    unrelated = FakeEvent(
        user_id="target", session_id="platform-A:GroupMessage:group-A-branch"
    )

    await plugin.block_blacklisted_messages(matching)
    await plugin.block_blacklisted_messages(unrelated)

    assert matching.stopped
    assert not unrelated.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("obsolete_scope", ["group", "session", "invalid"])
@pytest.mark.parametrize("source", ["admin", "llm"])
async def test_new_bans_are_global_even_with_obsolete_scope_setting(
    plugin_factory, command_metadata, obsolete_scope, source
):
    plugin = await plugin_factory(ban_scope=obsolete_scope)
    actor = FakeEvent(user_id="owner", admin=True, message="永久拉黑 target")
    if source == "admin":
        await invoke_blacklist_command(plugin, actor, command_metadata)
    else:
        await plugin.llm_blacklist_manage(actor, "target", permanent=True)

    record = plugin.store.get_active("target", identity(actor))
    assert record.scope == "global"
    assert record.scope_id == ""
    elsewhere = FakeEvent(
        user_id="target",
        group_id="another-group",
        session_id="platform-A:GroupMessage:another-group-branch",
    )
    await plugin.block_blacklisted_messages(elsewhere)
    assert elsewhere.stopped


@pytest.mark.asyncio
async def test_expired_temporary_ban_does_not_block_message(plugin, monkeypatch):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", duration_minutes=1)
    monkeypatch.setattr(plugin_module.time, "time", lambda: CLOCK + 60)
    event = FakeEvent(user_id="target")

    await plugin.block_blacklisted_messages(event)

    assert not event.stopped
    assert not event.call_llm


@pytest.mark.asyncio
async def test_nonstrict_mode_blocks_commands_and_llm_but_keeps_listeners(
    plugin_factory,
):
    plugin = await plugin_factory(strict_block_mode=False)
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    event = FakeEvent(user_id="target")
    command = SimpleNamespace(event_filters=[CommandFilter("测试")])
    command_group = SimpleNamespace(event_filters=[CommandGroupFilter("管理")])
    listener = SimpleNamespace(event_filters=[object()])
    handlers = [command, listener, command_group]
    event.set_extra("activated_handlers", handlers)

    await plugin.block_blacklisted_messages(event)

    assert handlers == [listener]
    assert event.call_llm
    assert not event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["block_waiting_llm", "block_llm_request"])
async def test_llm_hooks_stop_blacklisted_request(plugin, hook):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    event = FakeEvent(user_id="target")

    arguments = (event, SimpleNamespace()) if hook == "block_llm_request" else (event,)
    await getattr(plugin, hook)(*arguments)

    assert event.stopped


@pytest.mark.asyncio
async def test_administrator_remains_able_to_recover_after_becoming_protected(plugin):
    event = FakeEvent(user_id="owner", admin=True)
    plugin.store.ban(
        BanRecord(
            user_id="owner",
            platform_id="platform-A",
            bot_id="bot-A",
            scope="global",
            scope_id="",
            reason="Legacy entry",
            created_at=CLOCK,
            expires_at=None,
            operator="legacy",
            source="legacy",
        )
    )

    await plugin.block_blacklisted_messages(event)
    await plugin.block_waiting_llm(event)
    await plugin.block_llm_request(event, SimpleNamespace())

    assert not event.stopped
    assert not event.call_llm


@pytest.mark.asyncio
async def test_strict_mode_stops_later_handlers_in_actual_sdk_pipeline(plugin):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    event = FakeEvent(user_id="target")
    calls = []

    async def blocker(current_event):
        calls.append("blocker")
        await plugin.block_blacklisted_messages(current_event)

    async def listener(current_event):
        calls.append("listener")

    module_name = "test_blacklist_pipeline"
    star_map[module_name] = StarMetadata(name="blacklist-pipeline-test")
    event.set_extra(
        "activated_handlers",
        [
            StarHandlerMetadata(
                event_type=EventType.AdapterMessageEvent,
                handler_full_name=f"{module_name}_{handler.__name__}",
                handler_name=handler.__name__,
                handler_module_path=module_name,
                handler=handler,
                event_filters=[],
            )
            for handler in [blocker, listener]
        ],
    )
    event.set_extra("handlers_parsed_params", {})

    try:
        async for _ in StarRequestSubStage().process(event):
            pass
    finally:
        star_map.pop(module_name, None)

    assert calls == ["blocker"]
    assert event.call_llm
    assert event.stopped


@pytest.mark.asyncio
async def test_initialize_migrates_old_data_without_modifying_source(
    plugin_factory, tmp_path
):
    legacy_path = tmp_path / "astrbot_plugin_llmtempban" / "blacklist.json"
    legacy_path.parent.mkdir(parents=True)
    original = json.dumps({"legacy-target": CLOCK + 600})
    legacy_path.write_text(original, encoding="utf-8")

    plugin = await plugin_factory()

    assert plugin.store.get_active("legacy-target", identity(FakeEvent())) is not None
    assert legacy_path.read_text(encoding="utf-8") == original


def test_real_sdk_tool_schema_exposes_permanent_boolean():
    tool = llm_tools.get_func("llm_blacklist_manage")

    assert tool is not None
    assert tool.parameters["properties"]["permanent"]["type"] == "boolean"
    assert "duration_minutes" in tool.parameters["properties"]


def test_shared_command_aliases_keep_priority_and_allow_public_filtering(
    command_metadata,
):
    command_filters = [
        event_filter
        for event_filter in command_metadata.event_filters
        if isinstance(event_filter, CommandFilter)
    ]
    assert len(command_filters) == 1
    assert command_filters[0].command_name == "黑名单"
    assert command_filters[0].alias >= {
        "llm黑名单",
        "拉黑",
        "永久拉黑",
        "解除拉黑",
        "llm拉黑",
        "llm永久拉黑",
        "llm解除拉黑",
    }
    assert command_metadata.extras_configs["priority"] == 100
    assert not any(
        isinstance(event_filter, PermissionTypeFilter)
        for event_filter in command_metadata.event_filters
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "minutes", "reason"),
    [
        ("拉黑 openid-target", 10, "管理员操作"),
        ("拉黑\topenid-target 25  repeated\n harassment", 25, "repeated harassment"),
        ("llm拉黑 openid-target", 10, "管理员操作"),
        (
            "llm拉黑 openid-target 100000 repeated harassment",
            1440,
            "repeated harassment",
        ),
    ],
)
async def test_direct_temporary_commands_parse_target_duration_and_reason(
    plugin, command_metadata, message, minutes, reason
):
    event = FakeEvent(user_id="owner", admin=True, message=message)

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    # The native SDK puts the target in action; the handler must recover the
    # intended operation from the command token before consuming arguments.
    assert event.get_extra("parsed_params") == {"action": "openid-target"}
    record = plugin.store.get_active("openid-target", identity(event))
    assert record is not None
    assert record.expires_at == CLOCK + minutes * 60
    assert record.reason == reason
    assert record.source == "admin"
    assert record.operator == "owner"
    assert "已保存拉黑记录" in results[0].get_plain_text()
    assert event.call_llm
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["黑名单", "llm黑名单"])
async def test_nested_temporary_commands_keep_their_argument_offsets(
    plugin, command_metadata, command
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} 拉黑 target 25 repeated harassment",
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("target", identity(event))
    assert record is not None
    assert record.expires_at == CLOCK + 25 * 60
    assert record.reason == "repeated harassment"
    assert results
    assert not event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "拉黑 new-target 25 unauthorized action",
        "永久拉黑 new-target unauthorized action",
        "解除拉黑 target",
        "llm拉黑 new-target 25 unauthorized action",
        "llm永久拉黑 new-target unauthorized action",
        "llm解除拉黑 target",
        "黑名单 永久 new-target unauthorized action",
    ],
)
async def test_nonadmin_cannot_manage_blacklist_even_if_handler_called_directly(
    plugin, command_metadata, message
):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    before = plugin.store.path.read_bytes()
    event = FakeEvent(message=message)
    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "仅 AstrBot 管理员" in results[0].get_plain_text()
    assert plugin.store.get_active("new-target", identity(event)) is None
    assert plugin.store.get_active("target", identity(event)).expires_at is None
    assert plugin.store.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["黑名单 永久", "llm黑名单 永久", "永久拉黑", "llm永久拉黑"],
)
async def test_admin_permanent_command_is_independent_of_llm_permanent_gate(
    plugin_factory, command_metadata, command
):
    plugin = await plugin_factory(allow_llm_permanent_ban=False)
    event = FakeEvent(
        user_id="owner", admin=True, message=f"{command} target repeated harassment"
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("target", identity(event))
    assert results
    assert record is not None
    assert record.expires_at is None
    assert record.reason == "repeated harassment"
    assert record.source == "admin"
    assert event.call_llm
    if command in {"永久拉黑", "llm永久拉黑"}:
        assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["黑名单 解封", "llm黑名单 解封", "解除拉黑", "llm解除拉黑"],
)
async def test_admin_can_explicitly_unban_permanent_record(
    plugin, command_metadata, command
):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    event = FakeEvent(user_id="owner", admin=True, message=f"{command} target")

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "已解除用户 target" in results[0].get_plain_text()
    assert plugin.store.get_active("target", identity(event)) is None
    assert event.call_llm
    if command in {"解除拉黑", "llm解除拉黑"}:
        assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "黑名单 拉黑 target nope",
        "拉黑",
        "永久拉黑",
        "解除拉黑",
        "拉黑 target nope",
        "拉黑 target 0",
        "拉黑 target -1",
        "拉黑 target 1.5",
        "解除拉黑 target extra",
        "llm拉黑 target nope",
        "llm解除拉黑 target extra",
    ],
)
async def test_invalid_admin_command_does_not_mutate_blacklist(
    plugin, command_metadata, message
):
    await plugin.llm_blacklist_manage(FakeEvent(), "existing-target", permanent=True)
    before = plugin.store.path.read_bytes()
    event = FakeEvent(user_id="owner", admin=True, message=message)

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "操作失败" in results[0].get_plain_text()
    assert plugin.store.get_active("target", identity(event)) is None
    assert plugin.store.path.read_bytes() == before
    assert event.call_llm
    if not message.startswith("黑名单 "):
        assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "target"),
    [
        ("拉黑 bot-A 25", "bot-A"),
        ("永久拉黑 owner repeated harassment", "owner"),
        ("llm永久拉黑 protected repeated harassment", "protected"),
    ],
)
async def test_direct_commands_preserve_protected_user_policy(
    plugin_factory, command_metadata, message, target
):
    plugin = await plugin_factory(protected_user_ids=["protected"])
    event = FakeEvent(user_id="owner", admin=True, message=message)

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "不能拉黑" in results[0].get_plain_text()
    assert plugin.store.get_active(target, identity(event)) is None
    assert event.call_llm
    assert event.stopped


@pytest.mark.asyncio
async def test_uninitialized_direct_command_still_stops_propagation(
    plugin, command_metadata
):
    plugin.store = None
    event = FakeEvent(user_id="owner", admin=True, message="拉黑 target 25")

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "尚未完成初始化" in results[0].get_plain_text()
    assert event.call_llm
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("duration", "with_mention"),
    [
        pytest.param("25", False, id="explicit-id-valid"),
        pytest.param("invalid", False, id="explicit-id-invalid"),
        pytest.param("25", True, id="mention-valid"),
        pytest.param("invalid", True, id="mention-invalid"),
    ],
)
async def test_direct_command_priority_and_stop_prevent_competing_sdk_handler(
    plugin, command_metadata, monkeypatch, duration, with_mention
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"拉黑 {'@Target Name(target)' if with_mention else 'target'} "
        f"{duration} repeated harassment",
        messages=(
            [
                Plain("/拉黑 "),
                At(qq="target", name="Target Name"),
                Plain(f" {duration} repeated harassment"),
            ]
            if with_mention
            else None
        ),
    )
    calls = []

    async def competing_command(current_event):
        calls.append("competing")

    module_name = "test_blacklist_command_collision"
    competing = StarHandlerMetadata(
        event_type=EventType.AdapterMessageEvent,
        handler_full_name=f"{module_name}_competing_command",
        handler_name="competing_command",
        handler_module_path=module_name,
        handler=competing_command,
        event_filters=[],
    )
    competing.event_filters.append(CommandFilter("拉黑", handler_md=competing))
    command = replace(command_metadata, handler=plugin.blacklist_command)
    registry = StarHandlerRegistry()
    # Register the competing command first so order depends on SDK priority.
    registry.append(competing)
    registry.append(command)
    monkeypatch.setitem(star_map, module_name, StarMetadata(name="competing-command"))
    monkeypatch.setitem(
        star_map, command.handler_module_path, StarMetadata(name="blacklist-command")
    )
    activated = []
    parsed = {}
    for handler in registry.get_handlers_by_event_type(EventType.AdapterMessageEvent):
        if all(
            event_filter.filter(event, {}) for event_filter in handler.event_filters
        ):
            activated.append(handler)
            parsed[handler.handler_full_name] = dict(event.get_extra("parsed_params"))
    assert activated == [command, competing]
    event.set_extra("activated_handlers", activated)
    event.set_extra("handlers_parsed_params", parsed)

    responses = []

    async def handler_stage(current_event):
        async for result in StarRequestSubStage().process(current_event):
            yield result

    async def send_spy(current_event):
        if not current_event.is_stopped() and current_event.get_result() is not None:
            responses.append(current_event.get_result().get_plain_text())

    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stages = [
        SimpleNamespace(process=handler_stage),
        SimpleNamespace(process=send_spy),
    ]
    await scheduler._process_stages(event)

    assert calls == []
    assert len(responses) == 1
    record = plugin.store.get_active("target", identity(event))
    if duration == "25":
        assert "已保存拉黑记录" in responses[0]
        assert record is not None
        assert record.expires_at == CLOCK + 25 * 60
    else:
        assert "操作失败" in responses[0]
        assert record is None
    assert event.call_llm
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "rendered_mention", "target"),
    [
        ("拉黑", "@Target Name With Spaces(123456)", 123456),
        ("llm拉黑", "", "openid-target"),
        ("黑名单 拉黑", "@Target Name With Spaces", "openid-target"),
        ("llm黑名单 拉黑", "<@openid-target>", "openid-target"),
    ],
)
async def test_temporary_mention_uses_component_id_and_plain_arguments(
    plugin, command_metadata, command, rendered_mention, target
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} {rendered_mention} 25 repeated harassment",
        messages=[
            Plain(f"/{command} "),
            At(qq=target, name="Target Name With Spaces"),
            Plain(" 25 repeated harassment"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active(str(target), identity(event))
    assert record is not None
    assert record.expires_at == CLOCK + 25 * 60
    assert record.reason == "repeated harassment"
    assert record.source == "admin"
    assert record.nickname == "Target Name With Spaces"
    assert "已保存拉黑记录" in results[0].get_plain_text()
    assert event.call_llm
    assert event.stopped == (command in {"拉黑", "llm拉黑"})


@pytest.mark.asyncio
async def test_temporary_mention_without_arguments_uses_default_duration(
    plugin, command_metadata
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message="拉黑",
        messages=[Plain("/拉黑 "), At(qq="openid-target", name="")],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("openid-target", identity(event))
    assert record is not None
    assert record.expires_at == CLOCK + 10 * 60
    assert record.reason == "管理员操作"
    assert "已保存拉黑记录" in results[0].get_plain_text()
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command", ["永久拉黑", "llm永久拉黑", "黑名单 永久", "llm黑名单 永久"]
)
async def test_permanent_mention_uses_component_id_and_multiword_reason(
    plugin, command_metadata, command
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} @Target Name(openid-target) repeated harassment",
        messages=[
            Plain(f"/{command} "),
            At(qq="openid-target", name="Target Name"),
            Plain(" repeated harassment"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("openid-target", identity(event))
    assert record is not None
    assert record.expires_at is None
    assert record.reason == "repeated harassment"
    assert record.nickname == "Target Name"
    assert "已保存拉黑记录" in results[0].get_plain_text()
    assert event.call_llm
    assert event.stopped == (command in {"永久拉黑", "llm永久拉黑"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "unban"),
    [
        ("解除拉黑", True),
        ("llm解除拉黑", True),
        ("黑名单 解封", True),
        ("llm黑名单 解封", True),
        ("黑名单 查询", False),
        ("llm黑名单 查询", False),
    ],
)
async def test_mention_query_and_unban_use_actual_id(
    plugin, command_metadata, command, unban
):
    await plugin.llm_blacklist_manage(FakeEvent(), "openid-target", permanent=True)
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} @Target Name(openid-target)",
        messages=[
            Plain(f"/{command} "),
            At(qq="openid-target", name="Target Name"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    text = results[0].get_plain_text()
    record = plugin.store.get_active("openid-target", identity(event))
    if unban:
        assert "已解除用户 openid-target" in text
        assert record is None
    else:
        assert "用户：openid-target" in text
        assert "永久" in text
        assert record is not None
    assert event.call_llm
    assert event.stopped == (command in {"解除拉黑", "llm解除拉黑"})


@pytest.mark.asyncio
@pytest.mark.parametrize("wake_prefix", ["/", "!"])
async def test_leading_bot_wake_mention_is_skipped_before_target(
    plugin, command_metadata, wake_prefix
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message="拉黑 @Target Name(openid-target) 25 repeated harassment",
        messages=[
            At(qq="bot-A", name="Bot Name"),
            Plain(f"{wake_prefix}拉黑 "),
            At(qq="openid-target", name="Target Name"),
            Plain(" 25 repeated harassment"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("openid-target", identity(event))
    assert record is not None
    assert record.expires_at == CLOCK + 25 * 60
    assert record.reason == "repeated harassment"
    assert record.nickname == "Target Name"
    assert plugin.store.get_active("bot-A", identity(event)) is None
    assert "已保存拉黑记录" in results[0].get_plain_text()
    assert event.stopped


@pytest.mark.asyncio
async def test_adjacent_plain_components_are_joined_around_mention(
    plugin, command_metadata
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message="拉黑 @Target(target) 25 repeated harassment",
        messages=[
            Plain("/拉"),
            Plain("黑 "),
            At(qq="target", name="Target"),
            Plain(" 2"),
            Plain("5 repeated "),
            Plain("harassment"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("target", identity(event))
    assert record is not None
    assert record.expires_at == CLOCK + 25 * 60
    assert record.reason == "repeated harassment"
    assert "已保存拉黑记录" in results[0].get_plain_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["拉黑", "永久拉黑"])
async def test_reason_mention_does_not_override_explicit_target_id(
    plugin, command_metadata, command
):
    duration = " 25" if command == "拉黑" else ""
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} explicit-target{duration} repeated harassment of "
        "@Other Name(mentioned-target)",
        messages=[
            Plain(f"/{command} explicit-target{duration} repeated harassment of "),
            At(qq="mentioned-target", name="Other Name"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("explicit-target", identity(event))
    assert record is not None
    assert record.reason == "repeated harassment of @Other Name(mentioned-target)"
    assert record.nickname == ""
    assert record.expires_at == (CLOCK + 25 * 60 if command == "拉黑" else None)
    assert plugin.store.get_active("mentioned-target", identity(event)) is None
    assert "已保存拉黑记录" in results[0].get_plain_text()
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "reason_prefix", "name", "expected_reason"),
    [
        ("拉黑", " 25 ", "Other Name", "@Other Name(mentioned-target)"),
        (
            "永久拉黑",
            " repeated harassment of ",
            "",
            "repeated harassment of @mentioned-target",
        ),
    ],
)
async def test_reason_mention_is_allowed_after_duration_or_reason_text(
    plugin, command_metadata, command, reason_prefix, name, expected_reason
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} @Target(target){reason_prefix}@Other Name(mentioned-target)",
        messages=[
            Plain(f"/{command} "),
            At(qq="target", name="Target"),
            Plain(reason_prefix),
            At(qq="mentioned-target", name=name),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("target", identity(event))
    assert record is not None
    assert record.reason == expected_reason
    assert record.nickname == "Target"
    assert plugin.store.get_active("mentioned-target", identity(event)) is None
    assert "已保存拉黑记录" in results[0].get_plain_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [At(qq="all", name="全体成员"), AtAll()])
async def test_mention_all_is_rejected_without_mutation(
    plugin, command_metadata, target
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message="永久拉黑",
        messages=[Plain("/永久拉黑 "), target],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "操作失败" in results[0].get_plain_text()
    assert plugin.store.list_records(identity(event)) == []
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["拉黑", "永久拉黑", "解除拉黑"])
async def test_adjacent_target_mentions_are_rejected_without_mutation(
    plugin, command_metadata, command
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} @First(first-target) @Second(second-target)",
        messages=[
            Plain(f"/{command} "),
            At(qq="first-target", name="First"),
            Plain(" "),
            At(qq="second-target", name="Second"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "操作失败" in results[0].get_plain_text()
    assert plugin.store.list_records(identity(event)) == []
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["黑名单 查询", "黑名单 解封"])
async def test_query_and_unban_mention_require_single_target(
    plugin, command_metadata, command
):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    before = plugin.store.path.read_bytes()
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"{command} @Target(target) extra text @Other(other-target)",
        messages=[
            Plain(f"/{command} "),
            At(qq="target", name="Target"),
            Plain(" extra text "),
            At(qq="other-target", name="Other"),
        ],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "操作失败" in results[0].get_plain_text()
    assert plugin.store.get_active("target", identity(event)).expires_at is None
    assert plugin.store.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["拉黑", "永久拉黑", "解除拉黑"])
async def test_nonadmin_mention_cannot_ban_or_unban(plugin, command_metadata, command):
    await plugin.llm_blacklist_manage(FakeEvent(), "existing-target", permanent=True)
    before = plugin.store.path.read_bytes()
    target = "existing-target" if command == "解除拉黑" else "new-target"
    event = FakeEvent(
        message=f"{command} @Target({target})",
        messages=[Plain(f"/{command} "), At(qq=target, name="Target")],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "仅 AstrBot 管理员" in results[0].get_plain_text()
    assert plugin.store.get_active("new-target", identity(event)) is None
    assert (
        plugin.store.get_active("existing-target", identity(event)).expires_at is None
    )
    assert plugin.store.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["bot-A", "owner", "protected"])
async def test_postcommand_mention_preserves_protected_user_policy(
    plugin_factory, command_metadata, target
):
    plugin = await plugin_factory(protected_user_ids=["protected"])
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=f"永久拉黑 @Target({target})",
        messages=[Plain("/永久拉黑 "), At(qq=target, name="Target")],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    assert "不能拉黑" in results[0].get_plain_text()
    assert plugin.store.list_records(identity(event)) == []
    assert event.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target", ["@nickname", "@nickname(123456)", "[At:123456]", "<@123456>"]
)
async def test_plain_mention_text_requires_real_component_or_user_id(
    plugin, command_metadata, target
):
    message = f"永久拉黑 {target} repeated harassment"
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message=message,
        messages=[Plain(f"/{message}")],
    )

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    text = results[0].get_plain_text()
    assert "操作失败" in text
    assert "ID" in text
    assert "艾特" in text or "@" in text
    assert plugin.store.list_records(identity(event)) == []
    assert event.stopped


@pytest.mark.asyncio
async def test_nickname_survives_reload_and_readonly_commands_use_saved_snapshot(
    plugin, plugin_factory, command_metadata
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        message="永久拉黑 @Saved Name(target)",
        messages=[Plain("/永久拉黑 "), At(qq="target", name="  下头 张三\n ")],
    )
    results = await invoke_blacklist_command(plugin, event, command_metadata)
    assert "用户：target" in results[0].get_plain_text()
    assert "昵称：下头 张三" in results[0].get_plain_text()
    payload = json.loads(plugin.store.path.read_text())
    assert payload["records"][0]["nickname"] == "下头 张三"

    restarted = await plugin_factory()
    before = restarted.store.path.read_bytes()
    for message, components in [
        ("黑名单", None),
        (
            "黑名单 查询 @New Name(target)",
            [Plain("/黑名单 查询 "), At(qq="target", name="后来改名")],
        ),
    ]:
        reader = FakeEvent(message=message, messages=components)
        results = await invoke_blacklist_command(restarted, reader, command_metadata)
        assert "用户：target" in results[0].get_plain_text()
        assert "昵称：下头 张三" in results[0].get_plain_text()
        assert "后来改名" not in results[0].get_plain_text()
    assert restarted.store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_llm_nickname_uses_matching_mention_not_wake_bot_or_other_users(plugin):
    event = FakeEvent(
        nickname="Operator Name",
        messages=[
            At(qq="bot-A", name="Wake Bot"),
            At(qq="unrelated", name="Other User"),
            At(qq="target", name=" Target\n Name "),
        ],
    )

    result = await plugin.llm_blacklist_manage(event, "target", permanent=True)

    assert plugin.store.get_active("target", identity(event)).nickname == "Target Name"
    assert "昵称：Target Name" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["actor", "other-target"])
async def test_sender_nickname_is_only_used_for_the_sender_target(plugin, target):
    event = FakeEvent(nickname=" Sender\t Name ")

    result = await plugin.llm_blacklist_manage(event, target, permanent=True)

    expected = "Sender Name" if target == "actor" else ""
    record = plugin.store.get_active(target, identity(event))
    assert record.nickname == expected
    assert f"昵称：{expected or '未记录'}" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "expected_name", "expected_actions"),
    [
        (
            [{"card": " Group Card ", "nickname": "Account Name"}],
            "Group Card",
            ["get_group_member_info"],
        ),
        (
            [{"card": "", "nickname": " Group Nickname "}],
            "Group Nickname",
            ["get_group_member_info"],
        ),
        (
            [{"card": "", "nickname": ""}, {"nick": " Stranger Name "}],
            "Stranger Name",
            ["get_group_member_info", "get_stranger_info"],
        ),
        (
            [RuntimeError("Member unavailable"), {"nickname": "Fallback Name"}],
            "Fallback Name",
            ["get_group_member_info", "get_stranger_info"],
        ),
    ],
)
async def test_direct_id_nickname_lookup_uses_bot_routing_and_group_card_priority(
    plugin, command_metadata, responses, expected_name, expected_actions
):
    event = FakeEvent(
        user_id="owner",
        admin=True,
        group_id="654321",
        bot_id="999999",
        platform_name="aiocqhttp",
        message="永久拉黑 123456",
    )
    event.bot = SimpleNamespace(call_action=AsyncMock(side_effect=responses))

    results = await invoke_blacklist_command(plugin, event, command_metadata)

    record = plugin.store.get_active("123456", identity(event))
    assert record.nickname == expected_name
    assert f"昵称：{expected_name}" in results[0].get_plain_text()
    calls = event.bot.call_action.await_args_list
    assert [item.kwargs["action"] for item in calls] == expected_actions
    for item in calls:
        assert item.kwargs["self_id"] == "999999"
        assert item.kwargs["user_id"] == 123456
        assert item.kwargs["no_cache"] is False
        if item.kwargs["action"] == "get_group_member_info":
            assert item.kwargs["group_id"] == 654321


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, [], "invalid", {"nickname": None}])
async def test_unavailable_nickname_lookup_does_not_prevent_ban(plugin, response):
    event = FakeEvent(group_id="", platform_name="aiocqhttp")
    event.bot = SimpleNamespace(call_action=AsyncMock(return_value=response))

    result = await plugin.llm_blacklist_manage(event, "123456", permanent=True)

    record = plugin.store.get_active("123456", identity(event))
    assert record is not None
    assert record.nickname == ""
    assert "已保存拉黑记录" in result
    assert "昵称：未记录" in result


@pytest.mark.asyncio
async def test_nickname_lookup_timeout_does_not_prevent_ban(plugin):
    event = FakeEvent(group_id="", platform_name="aiocqhttp")
    event.bot = SimpleNamespace(call_action=AsyncMock(side_effect=TimeoutError))

    result = await plugin.llm_blacklist_manage(event, "123456", permanent=True)

    assert plugin.store.get_active("123456", identity(event)).nickname == ""
    assert "已保存拉黑记录" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("known_mention", [True, False])
async def test_nickname_lookup_skips_network_for_known_mentions_or_openids(
    plugin, known_mention
):
    target = "123456" if known_mention else "openid-target"
    components = [At(qq=target, name="Known Name")] if known_mention else []
    event = FakeEvent(platform_name="aiocqhttp", messages=components)
    event.bot = SimpleNamespace(call_action=AsyncMock())

    await plugin.llm_blacklist_manage(event, target, permanent=True)

    event.bot.call_action.assert_not_awaited()
    assert plugin.store.get_active(target, identity(event)).nickname == (
        "Known Name" if known_mention else ""
    )


@pytest.mark.asyncio
async def test_repeated_ban_keeps_known_nickname_and_updates_a_new_snapshot(plugin):
    event = FakeEvent(messages=[At(qq="target", name="Original Name")])
    await plugin.llm_blacklist_manage(event, "target", permanent=True)
    await plugin.llm_blacklist_manage(FakeEvent(), "target", duration_minutes=1)
    assert (
        plugin.store.get_active("target", identity(event)).nickname == "Original Name"
    )

    renamed = FakeEvent(messages=[At(qq="target", name="Updated Name")])
    await plugin.llm_blacklist_manage(renamed, "target", duration_minutes=1)
    record = plugin.store.get_active("target", identity(event))
    assert record.nickname == "Updated Name"
    assert record.expires_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("黑名单", "黑名单（"),
        ("llm黑名单", "黑名单（"),
        ("黑名单 列表", "黑名单（"),
        ("llm黑名单 列表", "黑名单（"),
        ("黑名单 查询 target", "用户：target"),
        ("llm黑名单 查询 target", "用户：target"),
        ("黑名单 帮助", "黑名单"),
    ],
)
async def test_public_blacklist_readonly_commands_are_available_to_nonadmins(
    plugin, command_metadata, message, expected
):
    await plugin.llm_blacklist_manage(FakeEvent(), "target", permanent=True)
    before = plugin.store.path.read_bytes()
    reader = FakeEvent(user_id="ordinary-user", message=message)

    results = await invoke_blacklist_command(plugin, reader, command_metadata)

    assert len(results) == 1
    assert expected in results[0].get_plain_text()
    assert "仅 AstrBot 管理员" not in results[0].get_plain_text()
    assert plugin.store.path.read_bytes() == before
    assert reader.call_llm
    assert not reader.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "message", ["黑名单", "llm黑名单 列表", "黑名单 查询 target", "黑名单 帮助"]
)
async def test_blacklisted_public_readonly_command_replies_once_and_stops_sdk_pipeline(
    plugin_factory, command_metadata, monkeypatch, strict, message
):
    plugin = await plugin_factory(strict_block_mode=strict)
    operator = FakeEvent(user_id="operator")
    await plugin.llm_blacklist_manage(operator, "target", permanent=True)
    await plugin.llm_blacklist_manage(operator, "blocked-reader", permanent=True)
    before = plugin.store.path.read_bytes()
    event = FakeEvent(user_id="blocked-reader", message=message)
    command_filter = next(
        item
        for item in command_metadata.event_filters
        if isinstance(item, CommandFilter)
    )
    assert command_filter.filter(event, {})
    parsed_params = dict(event.get_extra("parsed_params"))
    calls = []
    responses = []
    llm_calls = []

    async def prefix(current_event):
        calls.append("prefix")

    async def blocker(current_event):
        calls.append("block")
        await plugin.block_blacklisted_messages(current_event)

    async def competing(current_event):
        calls.append("competing")

    module_name = "test_public_blacklist_pipeline"
    prefix_md = StarHandlerMetadata(
        event_type=EventType.AdapterMessageEvent,
        handler_full_name=f"{module_name}_prefix",
        handler_name="prefix",
        handler_module_path=module_name,
        handler=prefix,
        event_filters=[],
    )
    competing_md = replace(
        prefix_md, handler_full_name=f"{module_name}_competing", handler=competing
    )
    blocker_md = star_handlers_registry.get_handler_by_full_name(
        f"{plugin_module.__name__}_block_blacklisted_messages"
    )
    assert blocker_md is not None
    blocker_md = replace(blocker_md, handler=blocker)
    command_md = replace(command_metadata, handler=plugin.blacklist_command)
    activated = [prefix_md, blocker_md, command_md, competing_md]
    event.set_extra("activated_handlers", activated)
    event.set_extra(
        "handlers_parsed_params", {command_md.handler_full_name: parsed_params}
    )
    monkeypatch.setitem(star_map, module_name, StarMetadata(name="public-reader-spy"))
    monkeypatch.setitem(
        star_map, command_md.handler_module_path, StarMetadata(name="blacklist-command")
    )

    async def handler_stage(current_event):
        async for result in StarRequestSubStage().process(current_event):
            yield result

    async def send_spy(current_event):
        if not current_event.is_stopped() and current_event.get_result() is not None:
            responses.append(current_event.get_result().get_plain_text())
            if not current_event.call_llm:
                llm_calls.append("llm")

    scheduler = PipelineScheduler.__new__(PipelineScheduler)
    scheduler.stages = [
        SimpleNamespace(process=handler_stage),
        SimpleNamespace(process=send_spy),
    ]
    await scheduler._process_stages(event)

    assert calls == ["prefix", "block"]
    assert len(responses) == 1
    assert "黑名单" in responses[0] or "用户：target" in responses[0]
    assert llm_calls == []
    assert event.get_extra("activated_handlers") is activated
    assert activated == [prefix_md, blocker_md, command_md]
    assert event.get_extra("llm_blacklist_public_read") is True
    assert event.stopped
    assert event.call_llm
    assert plugin.store.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "聊天 黑名单",
        "黑名单 永久 target",
        "llm黑名单 解封 blocked-reader",
        "拉黑 target",
        "黑名单 未知 target",
    ],
)
async def test_blacklisted_management_or_unrelated_text_does_not_get_public_exemption(
    plugin, command_metadata, message
):
    await plugin.llm_blacklist_manage(FakeEvent(), "blocked-reader", permanent=True)
    before = plugin.store.path.read_bytes()
    event = FakeEvent(user_id="blocked-reader", message=message)
    event.set_extra("activated_handlers", [command_metadata])

    await plugin.block_blacklisted_messages(event)

    assert event.stopped
    assert event.call_llm
    assert not event.get_extra("llm_blacklist_public_read", False)
    assert plugin.store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_unrelated_same_named_handler_cannot_get_public_blacklist_exemption(
    plugin, command_metadata
):
    await plugin.llm_blacklist_manage(FakeEvent(), "blocked-reader", permanent=True)
    event = FakeEvent(user_id="blocked-reader", message="黑名单 列表")
    spoofed = replace(
        command_metadata,
        handler_full_name="unrelated_plugin_blacklist_command",
        handler_module_path="unrelated_plugin",
    )
    event.set_extra("activated_handlers", [spoofed])

    await plugin.block_blacklisted_messages(event)

    assert event.stopped
    assert not event.get_extra("llm_blacklist_public_read", False)


@pytest.mark.asyncio
async def test_storage_failure_does_not_report_success_or_mutate_state(
    plugin, monkeypatch
):
    event = FakeEvent()

    def fail_to_save(record):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(plugin.store, "ban", fail_to_save)

    result = await plugin.llm_blacklist_manage(event, "target", permanent=True)

    assert isinstance(result, str)
    assert "失败" in result
    assert plugin.store.get_active("target", identity(event)) is None


def test_far_future_legacy_expiry_can_be_displayed():
    record = BanRecord(
        user_id="legacy-target",
        platform_id="",
        bot_id="",
        scope="global",
        scope_id="",
        reason="Legacy far-future expiry",
        created_at=CLOCK,
        expires_at=1e30,
        operator="legacy",
        source="legacy",
    )

    result = LLMBlacklistPlugin._describe_record(record)

    assert "legacy-target" in result
    assert "远期" in result


@pytest.mark.asyncio
async def test_blacklisted_requester_cannot_unban_itself_in_followup_tool_call(plugin):
    event = FakeEvent()
    await plugin.llm_blacklist_manage(event, "actor", permanent=True)

    await plugin.llm_blacklist_manage(event, "actor", duration_minutes=0)

    assert plugin.store.get_active("actor", identity(event)).expires_at is None


@pytest.mark.asyncio
async def test_string_false_cannot_accidentally_enable_permanent_ban(plugin):
    event = FakeEvent()

    await plugin.llm_blacklist_manage(event, "target", permanent="false")

    assert plugin.store.get_active("target", identity(event)) is None
