"""LLM 重试对进度面板可见的回归测试。

线上事故（002624，2026-10-30）：Bull Researcher 的 LLM 请求超时后，应用层
默默重试 3 次、每次等满 150s——用户在界面上看到的是"卡住不动"约 10 分钟后
报 Request timed out.。重试 warning 原本只写日志文件，本组测试锁住
"warning → 进度面板提示" 这条转发链，以及提示的保留/清理规则。
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from web.progress import ProgressTracker
from web.runner import _RETRY_LOGGER_NAME, _RetryNoticeHandler


@pytest.mark.unit
def test_add_notice_formats_and_dedups():
    tracker = ProgressTracker()
    tracker.add_notice("LLM 连接失败（APITimeoutError），5.0 秒后重试（1/3）")
    tracker.add_notice("LLM 连接失败（APITimeoutError），5.0 秒后重试（1/3）")

    assert len(tracker.notices) == 1
    assert tracker.notices[0].endswith("（1/3）")
    assert tracker.notices[0].startswith("[")


@pytest.mark.unit
def test_add_notice_keeps_only_recent_five():
    tracker = ProgressTracker()
    for i in range(8):
        tracker.add_notice(f"第 {i} 次重试")

    assert len(tracker.notices) == 5
    assert "第 7 次重试" in tracker.notices[-1]
    # 保留的是最后 5 条（第 3~7 次），第 2 次已被挤掉
    assert "第 2 次重试" not in " ".join(tracker.notices)


@pytest.mark.unit
def test_notice_ignored_after_stop_requested():
    tracker = ProgressTracker()
    tracker.is_running = True
    assert tracker.request_stop() is True
    tracker.add_notice("停了就别再冒提示")
    assert tracker.notices == []


@pytest.mark.unit
def test_notices_cleared_on_stop():
    tracker = ProgressTracker()
    tracker.add_notice("某次重试")
    tracker.is_running = True
    tracker.request_stop()
    assert tracker.notices == []


@pytest.mark.unit
def test_handler_forwards_retry_warning_to_tracker():
    tracker = ProgressTracker()
    handler = _RetryNoticeHandler(tracker)
    logger = logging.getLogger(_RETRY_LOGGER_NAME)

    logger.addHandler(handler)
    try:
        logger.warning(
            "LLM 连接失败（%s），%s 秒后重试（%d/%d）",
            "APITimeoutError", "5.0", 1, 3,
        )
    finally:
        logger.removeHandler(handler)

    assert len(tracker.notices) == 1
    assert "APITimeoutError" in tracker.notices[0]
    # info 级别不该进面板：handler 只挂 WARNING 起
    logger.info("普通日志")
    assert len(tracker.notices) == 1


@pytest.mark.unit
def test_handler_detached_after_run(monkeypatch):
    """线程结束后钩子必须摘掉，否则 tracker 泄漏给下一次无关运行。"""
    from web import runner

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph",
        MagicMock(side_effect=AssertionError("不应真的建图")),
    )
    monkeypatch.setattr(runner, "record_incomplete_task", MagicMock())

    logger = logging.getLogger(_RETRY_LOGGER_NAME)
    before = set(logger.handlers)

    tracker = ProgressTracker()
    thread = runner.run_analysis_in_thread(
        "300750", "2026-10-30", {}, tracker, selected_analysts=["market"]
    )
    thread.join(timeout=10)

    assert set(logger.handlers) == before
    assert not any(isinstance(h, _RetryNoticeHandler) for h in logger.handlers)
