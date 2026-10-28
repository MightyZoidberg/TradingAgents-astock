"""最小可运行示例 —— 就是 README「快速开始 · 3. 运行分析」里的那段代码。

`config` 是传给 TradingAgentsGraph 的**覆盖项**字典，不是配置文件：
只写要改的键，其余取 tradingagents/default_config.py 的默认值（#101）。
API key 走 .env（见 README 第 2 步）。运行：uv run python main.py
"""
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()

# ── 智谱 GLM 示例 ───────────────────────────────────
config = {
    "llm_provider": "glm",
    "deep_think_llm": "glm-5.3",
    "quick_think_llm": "glm-5.3-flash",
    "output_language": "Chinese",
}

# ── DeepSeek 示例 ───────────────────────────────────
# config = {
#     "llm_provider": "deepseek",
#     "deep_think_llm": "deepseek-chat",
#     "quick_think_llm": "deepseek-chat",
#     "output_language": "Chinese",
# }

if __name__ == "__main__":
    ta = TradingAgentsGraph(debug=True, config=config)
    final_state, decision = ta.propagate("688017", "2026-05-12")
    print(decision)
