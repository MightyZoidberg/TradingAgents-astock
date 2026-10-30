"""Background thread runner for TradingAgentsGraph pipeline."""

from __future__ import annotations

import logging
import re
import threading
import traceback
from typing import Any

from web.history import clear_incomplete_task, record_incomplete_task
from web.progress import ALL_ANALYST_IDS, PIPELINE_STAGES, ProgressTracker, visible_stages
from web.stock_display import normalize_report_state_mentions, normalize_stock_mentions

# LLM 应用层重试的 warning 从这个 logger 发出（openai_client.invoke）。
_RETRY_LOGGER_NAME = "tradingagents.llm_clients.openai_client"


class _RetryNoticeHandler(logging.Handler):
    """把 LLM 重试 warning 转发到进度面板。

    重试原本只写日志文件：每次重试要等满 llm_timeout（默认 150s），3 次重试
    就是约 10 分钟，界面上却只是"卡住不动"——用户无从分辨是在干活还是在等死。
    """

    def __init__(self, tracker: ProgressTracker):
        super().__init__(level=logging.WARNING)
        self._tracker = tracker

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._tracker.add_notice(record.getMessage())
        except Exception:  # noqa: BLE001 — 日志钩子绝不反噬主流程
            pass


_REPORT_KEY_TO_STAGE = {s["report_key"]: s["id"] for s in PIPELINE_STAGES}

_ANALYST_REPORT_KEYS = [
    "market_report", "sentiment_report", "news_report",
    "fundamentals_report", "policy_report", "hot_money_report", "lockup_report",
]


def _discard_stopped_run(
    ticker: str,
    trade_date: str,
    config: dict,
    tracker: ProgressTracker,
) -> None:
    """Clear resumable artifacts for a user-stopped run."""
    from tradingagents.graph.checkpointer import clear_checkpoint

    clear_incomplete_task(ticker, trade_date)
    clear_checkpoint(config["data_cache_dir"], ticker, trade_date)
    tracker.mark_stopped()


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _detect_completed_stages(
    chunk: dict[str, Any],
    tracker: ProgressTracker,
) -> None:
    """Check the streamed chunk for newly completed stages."""
    for report_key in _ANALYST_REPORT_KEYS:
        stage_id = _REPORT_KEY_TO_STAGE[report_key]
        content = chunk.get(report_key, "")
        if content and tracker.stage_status(stage_id) != "done":
            report = normalize_stock_mentions(str(content), tracker.ticker, chunk)
            tracker.mark_stage_done(stage_id, _strip_think_tags(report))

    dqs = chunk.get("data_quality_summary", "")
    if dqs and tracker.stage_status("quality_gate") != "done":
        tracker.mark_stage_done("quality_gate", normalize_stock_mentions(str(dqs), tracker.ticker, chunk))

    debate = chunk.get("investment_debate_state")
    if debate and isinstance(debate, dict):
        judge = debate.get("judge_decision", "")
        if judge and tracker.stage_status("debate") != "done":
            tracker.mark_stage_done("debate", normalize_stock_mentions(str(judge), tracker.ticker, chunk))

    trader_plan = chunk.get("trader_investment_plan", "")
    if trader_plan and tracker.stage_status("trader") != "done":
        report = normalize_stock_mentions(str(trader_plan), tracker.ticker, chunk)
        tracker.mark_stage_done("trader", _strip_think_tags(report))

    risk = chunk.get("risk_debate_state")
    if risk and isinstance(risk, dict):
        risk_judge = risk.get("judge_decision", "")
        if risk_judge and tracker.stage_status("risk") != "done":
            tracker.mark_stage_done("risk", normalize_stock_mentions(str(risk_judge), tracker.ticker, chunk))

    final = chunk.get("final_trade_decision", "")
    if final and tracker.stage_status("pm") != "done":
        report = normalize_stock_mentions(str(final), tracker.ticker, chunk)
        tracker.mark_stage_done("pm", _strip_think_tags(report))


def _infer_active_stage(tracker: ProgressTracker) -> None:
    """Set the current_stage to the first non-completed stage."""
    for stage in visible_stages(tracker.selected_analysts):
        if tracker.stage_status(stage["id"]) == "pending":
            tracker.mark_stage_active(stage["id"])
            return


def _run(ticker: str, trade_date: str, config: dict, tracker: ProgressTracker) -> None:
    """Execute the full pipeline in the current thread."""
    from cli.stats_handler import StatsCallbackHandler
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    stats = StatsCallbackHandler()

    graph = TradingAgentsGraph(
        selected_analysts=tracker.selected_analysts,
        debug=True,
        config=config,
        callbacks=[stats],
    )

    init_state, args, _ = graph.prepare_graph_run(
        ticker,
        trade_date,
        callbacks=[stats],
    )

    last_chunk: dict[str, Any] = {}

    try:
        def _close_and_discard() -> None:
            graph.close_graph_run()
            _discard_stopped_run(ticker, trade_date, config, tracker)

        if tracker.stop_requested:
            _close_and_discard()
            return

        stream = graph.graph.stream(init_state, **args)
        while True:
            tracker.wait_if_paused()
            if tracker.stop_requested:
                _close_and_discard()
                return
            try:
                chunk = next(stream)
            except StopIteration:
                break

            if tracker.stop_requested:
                _close_and_discard()
                return

            last_chunk = chunk
            _detect_completed_stages(chunk, tracker)
            _infer_active_stage(tracker)
            record_incomplete_task(
                ticker,
                trade_date,
                status="paused" if tracker.is_paused else "running",
                completed_stages=tracker.completed_stages,
                selected_analysts=tracker.selected_analysts,
            )

            s = stats.get_stats()
            tracker.update_stats(s["llm_calls"], s["tool_calls"], s["tokens_in"], s["tokens_out"])

        if tracker.stop_requested:
            _close_and_discard()
            return

        if not last_chunk:
            raise RuntimeError("分析没有返回任何结果，请清理断点后重试。")

        # #55: 报告标的统一显示为「代码+名称」，须在 finalize 落盘前归一化 last_chunk
        normalize_report_state_mentions(last_chunk, ticker)

        signal = graph.finalize_graph_run(ticker, trade_date, last_chunk)
        if tracker.stop_requested:
            _close_and_discard()
            return

        tracker.mark_complete(last_chunk, signal)
        clear_incomplete_task(ticker, trade_date)
    finally:
        graph.close_graph_run()


def run_analysis_in_thread(
    ticker: str,
    trade_date: str,
    config: dict,
    tracker: ProgressTracker,
    selected_analysts: list[str] | None = None,
) -> threading.Thread:
    """Launch the pipeline in a daemon thread. Returns the thread handle.

    selected_analysts: 本次勾选要跑的分析师（PIPELINE_STAGES 前 7 项的 id 子集）。
    None = 全部 7 个（原行为）。至少要有一个——空列表会被 TradingAgentsGraph
    在建图时拒绝，这里提前归一成默认，避免把明显非法的输入送进后台线程。
    """
    analysts = [a for a in (selected_analysts or ALL_ANALYST_IDS) if a in ALL_ANALYST_IDS]
    if not analysts:
        analysts = list(ALL_ANALYST_IDS)

    tracker.ticker = ticker
    tracker.trade_date = trade_date
    tracker.selected_analysts = analysts
    tracker.is_running = True
    tracker.mark_stage_active(visible_stages(analysts)[0]["id"])
    record_incomplete_task(
        ticker,
        trade_date,
        status="running",
        completed_stages=tracker.completed_stages,
        selected_analysts=analysts,
    )

    def _target() -> None:
        notice_handler = _RetryNoticeHandler(tracker)
        retry_logger = logging.getLogger(_RETRY_LOGGER_NAME)
        retry_logger.addHandler(notice_handler)
        try:
            _run(ticker, trade_date, config, tracker)
        except Exception as exc:
            if tracker.stop_requested:
                try:
                    _discard_stopped_run(ticker, trade_date, config, tracker)
                except Exception:
                    traceback.print_exc()
                return
            traceback.print_exc()
            record_incomplete_task(
                ticker,
                trade_date,
                status="error",
                error=str(exc),
                completed_stages=tracker.completed_stages,
                selected_analysts=tracker.selected_analysts,
            )
            tracker.mark_error(str(exc))
        finally:
            retry_logger.removeHandler(notice_handler)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t
