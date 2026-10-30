"""Web 侧「勾选分析项，仅处理勾选项」的回归测试。

核心图 TradingAgentsGraph(selected_analysts=...) 一直支持只构建选中的分析师，
但 Web 界面从未暴露——全选 7 项的完整流水线动辄几十分钟，一次超时就前功尽弃。
这里锁死贯通链路上的关键行为：

1. 质量门控只审核勾选的分析师（否则没勾的被打成「F·报告为空」，fail_count
   被无关空报告推过阈值、误跳过 LLM 复审）；
2. runner 把勾选项原样传给 TradingAgentsGraph 建图；
3. 进度面板/活跃阶段推断只认勾选的阶段；
4. 未完成任务记录带上勾选项——断点按节点名寻址，续跑必须用同一组勾选重建图。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.quality_gate import create_quality_gate
from web.progress import ALL_ANALYST_IDS, ProgressTracker, visible_stages


def _gate_state(reports: dict[str, str]) -> dict:
    """质量门控节点看到的最小 state：未给的报告字段视为空串。"""
    return {
        "trade_date": "2026-10-30",
        "company_of_interest": "300750",
        **reports,
    }


def _long_report() -> str:
    # 带汇总表格的长报告：硬检查应给 A，确保测试只关注「勾选与否」这一个变量。
    return "|指标|数值|\n|---|---|\n|收盘|100|\n" * 5 + "x" * 200


@pytest.mark.unit
def test_unselected_analysts_are_not_graded_f():
    """没勾的分析师必须标「本次未选择运行」，而不是 F 级差评。"""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="审核通过")

    node = create_quality_gate(llm, selected_analysts=["market", "news"])
    out = node(_gate_state({
        "market_report": _long_report(),
        "news_report": _long_report(),
    }))

    summary = out["data_quality_summary"]
    assert "[—] 本次未选择运行" in summary
    assert "报告为空" not in summary
    # 只勾 2 位、都合格 ⇒ 失败数不过半 ⇒ LLM 复审照常执行
    assert "审核通过" in summary


@pytest.mark.unit
def test_no_selection_means_all_seven():
    """不传 selected_analysts（老调用方）= 原行为：7 位都进硬检查。"""
    llm = MagicMock()

    node = create_quality_gate(llm)
    out = node(_gate_state({}))

    assert out["data_quality_summary"].count("报告为空") == 7
    # 7 份空报告 ⇒ 失败 7，7*2 > 7 ⇒ 跳过 LLM 复审（与原 fail_count<4 阈值一致）
    llm.invoke.assert_not_called()


@pytest.mark.unit
def test_review_threshold_is_proportional():
    """勾 1 位且它失败 ⇒ 过半失败 ⇒ 跳过 LLM 复审（原 <4 阈值罩不到的新场景）。"""
    llm = MagicMock()
    node = create_quality_gate(llm, selected_analysts=["market"])
    out = node(_gate_state({"market_report": ""}))

    llm.invoke.assert_not_called()
    assert "跳过 — 多数报告未通过硬检查" in out["data_quality_summary"]


@pytest.mark.unit
def test_review_prompt_only_lists_selected_analysts():
    """LLM 复审的提示词只包含勾选的分析师，表格行也不给没勾的留位。"""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="ok")

    node = create_quality_gate(llm, selected_analysts=["market", "hot_money"])
    node(_gate_state({
        "market_report": _long_report(),
        "hot_money_report": _long_report(),
    }))

    prompt = llm.invoke.call_args[0][0]
    assert "2 位分析师" in prompt
    assert "技术分析师" in prompt
    assert "游资追踪师" in prompt
    for absent in ("情绪分析师", "新闻分析师", "基本面分析师", "政策分析师", "解禁监控师"):
        assert absent not in prompt, f"未勾选的 {absent} 不应出现在复审提示词里"


@pytest.mark.unit
def test_runner_builds_graph_with_selected_analysts(monkeypatch):
    """runner 必须把勾选项传给 TradingAgentsGraph——少这一步，界面勾了也白勾。"""
    from web import runner

    captured: dict = {}

    class _FakeGraph:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            # stream 立即结束 ⇒ _run 走「没有返回任何结果」的报错路径，
            # 被 _target 捕获后只调用 mock 掉的 record_incomplete_task，无真实副作用。
            self.graph = SimpleNamespace(stream=MagicMock(return_value=iter([])))
            self.prepare_graph_run = MagicMock(return_value=({}, {}, None))
            self.close_graph_run = MagicMock()

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph", _FakeGraph
    )
    record = MagicMock()
    monkeypatch.setattr(runner, "record_incomplete_task", record)

    tracker = ProgressTracker(ticker="300750", trade_date="2026-10-30")
    thread = runner.run_analysis_in_thread(
        "300750", "2026-10-30", {"data_cache_dir": "x"}, tracker,
        selected_analysts=["news", "market"],
    )
    thread.join(timeout=10)

    assert captured.get("selected_analysts") == ["news", "market"]
    assert tracker.selected_analysts == ["news", "market"]


@pytest.mark.unit
def test_run_analysis_in_thread_defaults_to_all_analysts(monkeypatch):
    """不传勾选（老调用方）= 全部 7 项，行为与升级前一致。"""
    from web import runner

    def _boom(**kwargs):
        raise AssertionError("不应真的建图")

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph", _boom
    )
    monkeypatch.setattr(runner, "record_incomplete_task", MagicMock())

    tracker = ProgressTracker()
    thread = runner.run_analysis_in_thread(
        "300750", "2026-10-30", {}, tracker, selected_analysts=None
    )
    thread.join(timeout=10)

    assert tracker.selected_analysts == list(ALL_ANALYST_IDS)


@pytest.mark.unit
def test_empty_selection_falls_back_to_all_analysts(monkeypatch):
    """空列表（非法输入）在入口处归一成全选，不送进后台线程才被建图拒绝。"""
    from web import runner

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph",
        MagicMock(side_effect=AssertionError("不应真的建图")),
    )
    monkeypatch.setattr(runner, "record_incomplete_task", MagicMock())

    tracker = ProgressTracker()
    thread = runner.run_analysis_in_thread(
        "300750", "2026-10-30", {}, tracker, selected_analysts=[]
    )
    thread.join(timeout=10)

    assert tracker.selected_analysts == list(ALL_ANALYST_IDS)


@pytest.mark.unit
def test_visible_stages_respect_selection_and_keep_pipeline_order():
    stages = visible_stages(["news", "market"])
    ids = [s["id"] for s in stages]
    # 勾选的分析师按 PIPELINE_STAGES 原顺序出现（用户勾选顺序 news,market 不得倒置），
    # 其后紧跟固定的质量门控 → 辩论 → 交易 → 风控 → 最终决策。
    assert ids[:2] == ["market", "news"]
    assert ids[2:] == ["quality_gate", "debate", "trader", "risk", "pm"]
    # None = 全部 12 个阶段（原行为）
    assert len(visible_stages(None)) == 12


@pytest.mark.unit
def test_infer_active_stage_skips_unselected_analysts():
    """活跃阶段推断不得停在没勾选的分析师上（它永远不会完成）。"""
    from web import runner

    tracker = ProgressTracker()
    tracker.selected_analysts = ["fundamentals"]
    runner._infer_active_stage(tracker)
    assert tracker.current_stage == "fundamentals"


@pytest.mark.unit
def test_record_incomplete_task_persists_selection(tmp_path, monkeypatch):
    """未完成任务记录必须存勾选项：断点续跑要用同一组勾选重建图。"""
    from web import history

    monkeypatch.setattr(
        history, "_INCOMPLETE_TASKS_FILE", tmp_path / "incomplete.json"
    )
    # 隔离真实的完成记录与断点查询（读的是用户主目录下的真实文件）
    monkeypatch.setattr(history, "_results_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(history, "_checkpoint_step", lambda *a: None)

    history.record_incomplete_task(
        "300750", "2026-10-30",
        status="running",
        completed_stages=["market"],
        selected_analysts=["market", "news"],
    )
    match = [e for e in history.get_incomplete_history() if e["ticker"] == "300750"]
    assert match and match[0]["selected_analysts"] == ["market", "news"]

    # 不带勾选的旧式调用不写该键（老任务记录保持原样，消费端回落全选）
    history.record_incomplete_task(
        "600519", "2026-10-29", status="error", error="boom"
    )
    old = [e for e in history.get_incomplete_history() if e["ticker"] == "600519"]
    assert old and "selected_analysts" not in old[0]


@pytest.mark.unit
def test_sidebar_validates_empty_selection():
    from web.components import sidebar

    assert sidebar.validate_analyst_selection([]) is not None
    assert sidebar.validate_analyst_selection(None) is not None
    assert sidebar.validate_analyst_selection(["market"]) is None


@pytest.mark.unit
def test_llm_config_persistence_round_trips_selection(tmp_path, monkeypatch):
    """勾选项随 llm_config.json 持久化：关掉浏览器再开，勾选不丢。"""
    from web.components import sidebar

    monkeypatch.setattr(sidebar, "_LLM_CONFIG_PATH", tmp_path / "llm_config.json")
    monkeypatch.setattr(
        sidebar.st, "session_state", {"selected_analysts": ["market", "policy"]}
    )

    sidebar._save_llm_config()
    saved = json.loads((tmp_path / "llm_config.json").read_text(encoding="utf-8"))
    assert saved["selected_analysts"] == ["market", "policy"]

    # 恢复：空 session_state 读回勾选；白名单过滤挡掉配置文件里的未知 id
    monkeypatch.setattr(sidebar.st, "session_state", {})
    (tmp_path / "llm_config.json").write_text(
        json.dumps({"selected_analysts": ["market", "bogus"]}),
        encoding="utf-8",
    )
    sidebar._load_saved_llm_config()
    assert sidebar.st.session_state["selected_analysts"] == ["market"]
