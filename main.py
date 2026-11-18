from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime
from sys import maxsize

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter

from .blacklist import BanContext, BanRecord, BlacklistStore, scope_target

PLUGIN_NAME = "astrbot_plugin_llm_blacklist"
BLOCK_HANDLER_PRIORITY = maxsize + 1
PAGE_SIZE = 10
HELP_TEXT = (
    "下头，拉黑了 · 管理员指令\n"
    "/黑名单 列表 [页码]\n"
    "/黑名单 查询 用户ID\n"
    "/黑名单 拉黑 用户ID [分钟] [原因]\n"
    "/黑名单 永久 用户ID [原因]\n"
    "/黑名单 解封 用户ID\n"
    "临时拉黑省略分钟时使用默认值；填写原因时需同时填写分钟。\n"
    "拉黑范围由 ban_scope 配置决定；解封移除当前会话中对该用户生效的全部记录。"
)


class LLMBlacklistPlugin(Star):
    """Manage persistent bans without modifying AstrBot's global pipeline."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """Validate settings before registering a usable plugin instance.

        Args:
            context: AstrBot context supplied by the plugin loader.
            config: Plugin configuration supplied by AstrBot.

        Raises:
            ValueError: If a duration, scope, or switch is invalid.
        """
        super().__init__(context)
        self.config = config
        self.max_duration = self._parse_minutes(
            config.get("max_blacklist_duration", 1440)
        )
        self.default_duration = self._parse_minutes(
            config.get("default_blacklist_duration", 10)
        )
        if self.max_duration <= 0 or self.default_duration <= 0:
            raise ValueError("Blacklist duration settings must be positive integers")
        self.default_duration = min(self.default_duration, self.max_duration)
        self.ban_scope = config.get("ban_scope", "global")
        if self.ban_scope not in {"global", "group", "session"}:
            raise ValueError("ban_scope must be global, group, or session")
        self.strict_block_mode = config.get("strict_block_mode", True)
        self.allow_llm_permanent_ban = config.get("allow_llm_permanent_ban", True)
        if not isinstance(self.strict_block_mode, bool) or not isinstance(
            self.allow_llm_permanent_ban, bool
        ):
            raise ValueError("Blacklist switches must be booleans")
        protected = config.get("protected_user_ids", [])
        if not isinstance(protected, list):
            raise ValueError("protected_user_ids must be a list")
        self.protected_user_ids = {self._normalize_user_id(item) for item in protected}
        self.store: BlacklistStore | None = None

    async def initialize(self) -> None:
        """Load the blacklist and import legacy records once if needed.

        Raises:
            OSError: If persistent data cannot be read or saved.
            ValueError: If the blacklist contains invalid data.
        """
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        legacy_path = data_dir.parent / "astrbot_plugin_llmtempban" / "blacklist.json"
        self.store = await asyncio.to_thread(
            BlacklistStore, data_dir / "blacklist.json", legacy_path
        )
        logger.info(
            f"[llm_blacklist] Blacklist loaded; legacy_imported={self.store.legacy_imported}"
        )

    @staticmethod
    def _parse_minutes(value: object) -> int:
        """Accept integer minutes without truncating fractional values.

        Args:
            value: An integer or a signed integer string.

        Returns:
            Validated duration in minutes.

        Raises:
            ValueError: If the value is not an integer.
        """
        if isinstance(value, bool):
            raise ValueError("拉黑时长必须为整数分钟。")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value.strip())
        raise ValueError("拉黑时长必须为整数分钟。")

    @staticmethod
    def _normalize_user_id(value: object) -> str:
        """Accept platform IDs, including OpenIDs, without allowing whitespace.

        Args:
            value: User ID received from a command or a tool.

        Returns:
            Nonempty normalized user ID.

        Raises:
            ValueError: If the ID has an unsupported type or content.
        """
        if isinstance(value, bool) or not isinstance(value, str | int):
            raise ValueError("用户 ID 必须为非空字符串。")
        user_id = str(value).strip()
        if (
            not user_id
            or len(user_id) > 256
            or any(character.isspace() or ord(character) < 32 for character in user_id)
        ):
            raise ValueError("用户 ID 不能为空、包含空白或超过 256 个字符。")
        return user_id

    @staticmethod
    def _identity(event: AstrMessageEvent) -> BanContext:
        """Build an identity that distinguishes configured platforms and bots.

        Args:
            event: Incoming message event.

        Returns:
            Namespace and scope identifiers for blacklist matching.
        """
        return BanContext(
            platform_id=str(event.get_platform_id()),
            bot_id=str(event.get_self_id()),
            group_id=str(event.get_group_id() or ""),
            session_id=event.unified_msg_origin,
        )

    def _active_ban(self, event: AstrMessageEvent) -> BanRecord | None:
        """Find an effective ban, while keeping administrators recoverable.

        Args:
            event: Incoming message event.

        Returns:
            The strongest applicable record, or None for exempt/unbanned users.
        """
        user_id = str(event.get_sender_id())
        if (
            self.store is None
            or event.is_admin()
            or user_id == str(event.get_self_id())
            or user_id in self.protected_user_ids
        ):
            return None
        return self.store.get_active(user_id, self._identity(event))

    @filter.event_message_type(
        filter.EventMessageType.ALL, priority=BLOCK_HANDLER_PRIORITY
    )
    async def block_blacklisted_messages(self, event: AstrMessageEvent) -> None:
        """Block later handlers or standard commands for blacklisted senders.

        Args:
            event: Incoming message event with activated handlers.
        """
        if self._active_ban(event) is None:
            return
        # Despite its name, True disables AstrBot's default LLM route.
        event.should_call_llm(True)
        if self.strict_block_mode:
            event.stop_event()
        else:
            handlers = event.get_extra("activated_handlers", []) or []
            handlers[:] = [
                handler
                for handler in handlers
                if not any(
                    isinstance(event_filter, CommandFilter | CommandGroupFilter)
                    for event_filter in handler.event_filters
                )
            ]
        logger.debug("[llm_blacklist] Blocked a blacklisted message")

    @filter.on_waiting_llm_request(priority=BLOCK_HANDLER_PRIORITY)
    async def block_waiting_llm(self, event: AstrMessageEvent) -> None:
        """Stop the LLM route before queue and thinking notifications.

        Args:
            event: Message event about to enter the LLM route.
        """
        if self._active_ban(event) is not None:
            event.should_call_llm(True)
            event.stop_event()

    @filter.on_llm_request(priority=BLOCK_HANDLER_PRIORITY)
    async def block_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Reject requests that reach AstrBot's LLM hooks after message handling.

        Args:
            event: Message event associated with the LLM request.
            req: Provider request supplied by AstrBot.
        """
        if self._active_ban(event) is not None:
            event.should_call_llm(True)
            event.stop_event()

    @staticmethod
    def _describe_record(record: BanRecord) -> str:
        """Format an effective record for command and tool responses.

        Args:
            record: Blacklist record to display.

        Returns:
            User ID, scope, expiry and reason in readable form.
        """
        scope = {"global": "当前 Bot 全部会话", "group": "群聊", "session": "会话"}[
            record.scope
        ]
        if not record.platform_id and not record.bot_id:
            scope = "旧版" + scope
            if record.scope == "global":
                scope = "旧版全局（全部 Bot）"
        if record.scope_id:
            scope += f" {record.scope_id}"
        if record.expires_at is None:
            expiry = "永久，需手动解封"
        else:
            remaining = math.ceil(max(0, record.expires_at - time.time()))
            minutes, seconds = divmod(remaining, 60)
            try:
                deadline = (
                    datetime.fromtimestamp(record.expires_at)
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M:%S %z")
                )
            except (OverflowError, OSError, ValueError):
                deadline = "远期（超出日期显示范围）"
            expiry = f"剩余 {minutes} 分 {seconds} 秒；到期 {deadline}"
        return (
            f"用户：{record.user_id}\n范围：{scope}\n期限：{expiry}\n"
            f"原因：{record.reason or '未填写'}"
        )

    async def _manage_blacklist(
        self,
        event: AstrMessageEvent,
        user_id: str,
        duration_minutes: int,
        reason: str,
        permanent: bool,
        source: str,
    ) -> str:
        """Apply command/tool policy and persist an explicit mutation.

        Args:
            event: Message identifying the operator and scope.
            user_id: Target user ID.
            duration_minutes: Integer minutes; zero removes applicable bans.
            reason: Reason to store with the record.
            permanent: Whether the ban has no automatic expiry.
            source: Either admin or llm.

        Returns:
            A user-facing result describing the committed state.
        """
        if self.store is None:
            return "操作失败：黑名单尚未完成初始化。"
        if source == "admin" and not event.is_admin():
            return "操作失败：仅 AstrBot 管理员可以管理黑名单。"
        if self._active_ban(event) is not None:
            return "操作失败：当前发送者已被拉黑。"
        try:
            user_id = self._normalize_user_id(user_id)
            duration_minutes = self._parse_minutes(duration_minutes)
            if not isinstance(permanent, bool):
                raise ValueError("permanent 必须为布尔值。")
            if permanent and duration_minutes == 0:
                raise ValueError("永久拉黑与解封不能同时指定。")
            identity = self._identity(event)
            if duration_minutes == 0:
                removed = await asyncio.to_thread(self.store.unban, user_id, identity)
                if removed:
                    logger.info(f"[llm_blacklist] Unbanned a user; source={source}")
                    return (
                        f"已解除用户 {user_id} 在当前会话生效的 {removed} 条拉黑记录，"
                        "恢复其交互权限。"
                    )
                return f"用户 {user_id} 在当前会话没有生效的拉黑记录。"
            if permanent and source == "llm" and not self.allow_llm_permanent_ban:
                return "拉黑失败：配置已关闭 LLM 永久拉黑；管理员仍可通过指令操作。"
            admins = self.context.get_config(event.unified_msg_origin).get(
                "admins_id", []
            )
            if (
                user_id == str(event.get_self_id())
                or user_id in self.protected_user_ids
                or user_id in {str(item) for item in admins}
                or (user_id == str(event.get_sender_id()) and event.is_admin())
            ):
                return "拉黑失败：不能拉黑 Bot 自身、AstrBot 管理员或受保护用户。"
            if not isinstance(reason, str):
                raise ValueError("拉黑原因必须为字符串。")
            reason = reason.strip()[:500] or "行为不当"
            if duration_minutes < 0:
                duration_minutes = self.default_duration
            duration_minutes = min(duration_minutes, self.max_duration)
            scope, scope_id = scope_target(identity, self.ban_scope)
            now = time.time()
            record = BanRecord(
                user_id=user_id,
                platform_id=identity.platform_id,
                bot_id=identity.bot_id,
                scope=scope,
                scope_id=scope_id,
                reason=reason,
                created_at=now,
                expires_at=None if permanent else now + duration_minutes * 60,
                operator=str(event.get_sender_id()),
                source=source,
            )
            saved = await asyncio.to_thread(self.store.ban, record)
            effective = self.store.get_active(user_id, identity) or saved
        except ValueError as error:
            return f"操作失败：{error}"
        except OSError:
            logger.exception("[llm_blacklist] Could not persist a blacklist mutation")
            return "操作失败：黑名单保存失败，请检查服务器日志后重试。"
        logger.info(
            f"[llm_blacklist] Saved a ban; source={source}, permanent={effective.expires_at is None}"
        )
        return "已保存拉黑记录。\n" + self._describe_record(effective)

    @filter.llm_tool(name="llm_blacklist_manage")
    async def llm_blacklist_manage(
        self,
        event: AstrMessageEvent,
        user_id: str,
        duration_minutes: int = -1,
        reason: str = "行为不当",
        permanent: bool = False,
    ) -> str:
        """对骚扰、刷屏或辱骂行为实施 Bot 拉黑，也可明确解除已有拉黑。

        临时拉黑到期自动解除；永久拉黑仅在明确解封后解除。
        根据对话中的实际行为判断目标，不要仅凭他人要求拉黑无关用户。
        本工具控制 Bot 的交互响应；聊天平台群禁言请使用群管理工具。

        Args:
            user_id(string): 目标用户 ID，支持数字 ID、OpenID 等平台标识。
            duration_minutes(number): 整数分钟；负数或省略使用默认值，0 明确解封。
            reason(string): 拉黑原因。
            permanent(boolean): true 表示永久拉黑，不能同时将时长设为 0。

        Returns:
            已提交的拉黑或解封结果。
        """
        return await self._manage_blacklist(
            event, user_id, duration_minutes, reason, permanent, "llm"
        )

    @filter.command("黑名单", alias={"llm黑名单"})
    @filter.permission_type(filter.PermissionType.ADMIN, raise_error=False)
    async def blacklist_command(
        self, event: AstrMessageEvent, action: str = "列表"
    ) -> AsyncIterator[MessageEventResult]:
        """查看、查询、临时或永久拉黑、解封用户，仅限 AstrBot 管理员。

        Args:
            event: Administrator command event.
            action: 子操作：列表、查询、拉黑、永久、解封或帮助。

        Yields:
            One command response.
        """
        if not event.is_admin():
            yield event.plain_result("仅 AstrBot 管理员可以管理黑名单。")
            return
        event.should_call_llm(True)
        if self.store is None:
            yield event.plain_result("黑名单尚未完成初始化。")
            return
        # AstrBot strips the wake prefix before evaluating CommandFilter.
        args = event.get_message_str().strip().split()[2:]
        identity = self._identity(event)
        try:
            if action == "列表":
                if len(args) > 1:
                    raise ValueError("格式：/黑名单 列表 [页码]")
                page = self._parse_minutes(args[0]) if args else 1
                records = self.store.list_records(identity)
                records.sort(
                    key=lambda item: (
                        item.expires_at is not None,
                        item.expires_at or 0,
                        item.user_id,
                    )
                )
                pages = max(1, math.ceil(len(records) / PAGE_SIZE))
                if not 1 <= page <= pages:
                    raise ValueError(f"页码应在 1 至 {pages} 之间。")
                if not records:
                    result = "当前 Bot 的黑名单为空。"
                else:
                    start = (page - 1) * PAGE_SIZE
                    lines = [
                        f"下头，拉黑了 · 黑名单（{page}/{pages} 页，共 {len(records)} 条）"
                    ]
                    for index, record in enumerate(
                        records[start : start + PAGE_SIZE], start=start + 1
                    ):
                        lines.append(f"\n{index}. {self._describe_record(record)}")
                    result = "\n".join(lines)
            elif action == "查询":
                if len(args) != 1:
                    raise ValueError("格式：/黑名单 查询 用户ID")
                user_id = self._normalize_user_id(args[0])
                record = self.store.get_active(user_id, identity)
                result = (
                    self._describe_record(record)
                    if record
                    else f"用户 {user_id} 在当前会话未被拉黑。"
                )
            elif action in {"拉黑", "永久", "解封"}:
                if not args or (action == "解封" and len(args) != 1):
                    raise ValueError("请指定用户 ID；使用 /黑名单 帮助 查看格式。")
                duration = -1
                reason_start = 1
                if action == "解封":
                    duration = 0
                elif action == "拉黑" and len(args) >= 2:
                    duration = self._parse_minutes(args[1])
                    reason_start = 2
                    if duration <= 0:
                        raise ValueError(
                            "临时拉黑时长必须大于 0；解封请用 /黑名单 解封。"
                        )
                result = await self._manage_blacklist(
                    event,
                    args[0],
                    duration,
                    " ".join(args[reason_start:]) or "管理员操作",
                    action == "永久",
                    "admin",
                )
            else:
                result = HELP_TEXT
        except ValueError as error:
            result = f"操作失败：{error}"
        yield event.plain_result(result)
