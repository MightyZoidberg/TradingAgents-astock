#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A 股历史数据批量入库：新浪直连 → 本地 SQLite 库（单文件，可拷贝迁移）。

用法（在项目根目录，用 .venv 的 Python）::

    # 全市场：股票列表 + 全历史日线 + 后复权因子（首次/日常增量都跑这个）
    python scripts/ingest_localdb.py

    # 只入指定的几只
    python scripts/ingest_localdb.py --stocks 600519,000001

    # 加上财报三表（每只 3 次请求，全市场约 3~4 小时，建议夜里跑）
    python scripts/ingest_localdb.py --fundamentals

    # 库直接放移动硬盘
    python scripts/ingest_localdb.py --db E:/astock/astock.db

    # 试试水：只入前 5 只
    python scripts/ingest_localdb.py --limit 5

说明：
- 增量是默认行为：库里已有的股票只补最新缺口（缺口天数折算 K 线根数），没有的
  股票抓全历史；重复运行安全（按 主键 upsert），中断后重跑即可续上。
- 引擎为新浪直连（东财行情接口和通达信 TCP 在部分网络被拦，新浪最通用）。
  AKShare 等第三方库以后可加为可选引擎，表结构不变。
- 迁移：入库进程退出后库目录只剩一个 .db 文件，整文件拷走即可（拷前确认没有
  进程在跑本脚本/工具）。另一台机器设 TRADINGAGENTS_LOCAL_DB 指向它即可使用。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from tradingagents.dataflows import localdb
from tradingagents.dataflows.a_stock import _fetch_financial_report_sina_raw

# Windows 控制台默认 GBK，中文日志/股票名会炸；统一走 UTF-8（不可重配置则忽略）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

_KLINE_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_LIST_URL = (
    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
_HFQ_URL = "https://finance.sina.com.cn/realstock/company/{symbol}/hfq.js"

_MARKET_TZ = timezone(timedelta(hours=8))

session = requests.Session()
session.headers.update(UA)


# ---------------------------------------------------------------------------
# 新浪抓取
# ---------------------------------------------------------------------------


def _get(url: str, params: dict | None = None, tries: int = 3):
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, params=params or {}, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - 统一重试
            last_err = e
            if attempt < tries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"GET {url} 失败（重试 {tries} 次）: {last_err}")


def fetch_stock_list() -> list[tuple[str, str, str]]:
    """全市场股票列表（沪深京），返回 [(code, name, exchange)]。"""
    out, page = [], 1
    seen = set()
    while True:
        r = _get(_LIST_URL, {
            "page": str(page), "num": "100", "sort": "symbol", "asc": "1",
            "node": "hs_a", "symbol": "", "_s_r_a": "page",
        })
        rows = r.json() or []
        if not rows:
            break
        for row in rows:
            code = str(row.get("code", "")).strip()
            name = str(row.get("name", "")).strip()
            symbol = str(row.get("symbol", "")).strip()
            if not re.fullmatch(r"\d{6}", code) or not name:
                continue
            if code in seen:
                continue
            seen.add(code)
            prefix = symbol[:2].lower()
            exchange = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix)
            if exchange is None:
                continue
            out.append((code, name, exchange))
        if len(rows) < 100:
            break
        page += 1
        time.sleep(0.2)
    return out


def fetch_daily_bars(code: str, datalen: int) -> pd.DataFrame:
    """新浪日线（不复权，成交量单位：股），返回 Date/OHLC/Volume。"""
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(_exchange_of(code), "sz")
    r = _get(_KLINE_URL, {
        "symbol": f"{prefix}{code}", "scale": "240", "ma": "no",
        "datalen": str(datalen),
    })
    data = r.json() or []
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            "Date": item["day"],
            "Open": float(item["open"]), "High": float(item["high"]),
            "Low": float(item["low"]), "Close": float(item["close"]),
            "Volume": int(item["volume"]),
        }
        for item in data
    ])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


_HFQ_PATTERN = re.compile(r'"d":"(\d{4}-\d{2}-\d{2})",\s*"f":"([0-9.]+)"')


def fetch_hfq_factors(code: str) -> list[tuple[str, float]]:
    """新浪后复权因子（事件日稀疏序列；读方按日 forward-fill）。"""
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(_exchange_of(code), "sz")
    r = _get(_HFQ_URL.format(symbol=f"{prefix}{code}"))
    return [
        (d, float(f)) for d, f in _HFQ_PATTERN.findall(r.text or "")
    ]


def _exchange_of(code: str) -> str:
    """6 位代码 → 交易所（与列表接口拿到的 exchange 无关的兜底判断）。"""
    if code.startswith("6"):
        return "SH"
    if code.startswith(("4", "8", "9")):
        return "BJ"
    return "SZ"


# ---------------------------------------------------------------------------
# 入库流程
# ---------------------------------------------------------------------------


def _incremental_datalen(conn, code: str, max_bars: int) -> int:
    """库里最后一根的日期 → 折算要补的根数；没有数据就抓全历史。"""
    last = localdb.last_bar_date(conn, code)
    if not last:
        return max_bars
    today = datetime.now(_MARKET_TZ).date()
    gap_days = (today - datetime.strptime(last, "%Y-%m-%d").date()).days
    # 交易日 ≈ 自然日 × 5/7，再加 15 根缓冲；多抓的行 upsert 幂等覆盖
    return max(20, min(max_bars, gap_days * 5 // 7 + 15))


def ingest(conn, codes: list[str], *, daily=True, factors=True, fundamentals=False,
           max_bars=9000, fund_periods=120, sleep=0.3, full=False) -> list[str]:
    failures = []
    bars_total = factors_total = fund_total = 0

    try:
        from tqdm import tqdm
        progress = tqdm(codes, desc="ingest", unit="stock", ascii=True)
    except ImportError:
        progress = codes

    for code in progress:
        try:
            if daily:
                df = fetch_daily_bars(code, max_bars if full else _incremental_datalen(conn, code, max_bars))
                if df.empty:
                    raise RuntimeError("新浪日线返回空（代码错误或已退市？）")
                bars_total += localdb.upsert_daily_bars(conn, code, df)
                conn.commit()

            if factors:
                events = fetch_hfq_factors(code)
                if events:
                    factors_total += localdb.upsert_hfq_factors(conn, code, events)
                    conn.commit()

            if fundamentals:
                for report_type in localdb.REPORT_TYPES:
                    raw = _fetch_financial_report_sina_raw(code, report_type, fund_periods)
                    if not raw.empty:
                        fund_total += localdb.upsert_financial(conn, code, report_type, raw)
                    conn.commit()
                    time.sleep(sleep)
        except KeyboardInterrupt:
            print(f"\n[中断] 已完成到 {code} 之前，直接重跑本脚本即可续上")
            conn.commit()
            raise
        except Exception as e:  # noqa: BLE001 - 单只失败不拖垮全量
            failures.append((code, str(e)))
            try:
                conn.rollback()
            except Exception:
                pass
        time.sleep(sleep)

    print(f"日线入库 {bars_total} 行，复权因子 {factors_total} 条，财报科目 {fund_total} 行")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="A 股历史数据批量入库（新浪直连 → 本地 SQLite）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default=None,
                        help="库文件路径（默认 TRADINGAGENTS_LOCAL_DB / ~/.tradingagents/astock.db）")
    parser.add_argument("--stocks", default=None,
                        help="逗号分隔的 6 位代码列表；不给 = 全市场")
    parser.add_argument("--stocks-file", default=None,
                        help="从文件读代码列表（每行一个，# 开头为注释）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 只（试水用）")
    parser.add_argument("--no-daily", action="store_true", help="跳过日线")
    parser.add_argument("--no-factors", action="store_true", help="跳过复权因子")
    parser.add_argument("--no-stock-list", action="store_true",
                        help="跳过股票列表刷新")
    parser.add_argument("--fundamentals", action="store_true",
                        help="入库财报三表（每只 3 次请求，较慢）")
    parser.add_argument("--fund-periods", type=int, default=120,
                        help="财报抓取期数（120 期 ≈ 30 年）")
    parser.add_argument("--full", action="store_true",
                        help="日线忽略库中已有深度，整段重抓（默认增量）")
    parser.add_argument("--max-bars", type=int, default=9000,
                        help="全历史抓取的 K 线根数上限（1990 年至今约 8700）")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="每只股票之间的间隔秒数")
    args = parser.parse_args()

    db_path = args.db or localdb.get_local_db_path()
    print(f"本地库: {db_path}")
    conn = localdb.connect(db_path)

    failures: list[tuple[str, str]] = []
    try:
        # 1) 股票列表（全市场跑表时刷新；指定股票清单时跳过）
        if not args.no_stock_list and not (args.stocks or args.stocks_file):
            print("刷新股票列表 ...")
            rows = fetch_stock_list()
            if rows:
                localdb.upsert_stock_basic(conn, rows)
                conn.commit()
                print(f"股票列表: {len(rows)} 只（沪深京）")
            else:
                failures.append(("_stock_list", "新浪列表接口返回空"))

        # 2) 确定要处理的股票
        if args.stocks:
            codes = [c.strip() for c in args.stocks.split(",") if c.strip()]
        elif args.stocks_file:
            codes = [
                line.split("#", 1)[0].strip()
                for line in Path(args.stocks_file).read_text(encoding="utf-8").splitlines()
                if line.split("#", 1)[0].strip()
            ]
        else:
            codes = [row[0] for row in conn.execute(
                "SELECT code FROM stock_basic ORDER BY code"
            ).fetchall()]
        if args.limit:
            codes = codes[: args.limit]
        print(f"待处理: {len(codes)} 只（日线={'开' if not args.no_daily else '关'}，"
              f"因子={'开' if not args.no_factors else '关'}，"
              f"财报={'开' if args.fundamentals else '关'}）")

        # 3) 逐只入库
        if codes:
            failures += ingest(
                conn, codes,
                daily=not args.no_daily,
                factors=not args.no_factors,
                fundamentals=args.fundamentals,
                max_bars=args.max_bars,
                fund_periods=args.fund_periods,
                sleep=args.sleep,
                full=args.full,
            )
    finally:
        # 收尾 checkpoint：把 WAL 并回主文件，之后库目录只剩一个 .db，可整拷迁移
        localdb.checkpoint_and_close(conn)

    if failures:
        print(f"\n失败 {len(failures)} 项（重跑本脚本即可只补增量）:")
        for code, err in failures[:20]:
            print(f"  {code}: {err}")
        if len(failures) > 20:
            print(f"  ... 其余 {len(failures) - 20} 项略")
        sys.exit(1)
    print("完成。")


if __name__ == "__main__":
    main()
