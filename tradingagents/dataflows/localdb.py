# -*- coding: utf-8 -*-
"""本地 A 股数据库（SQLite 单文件）。

目标：把 A 股历史数据沉淀成一个**单文件 SQLite 库**，本工具和其它工具都能读，
并且可以整文件拷贝（移动硬盘/U盘）在电脑间迁移。

约定（对二次开发和外部消费方都很重要，改动前先想清楚）：
- 单文件：WAL 模式只用于写入加速，写入进程正常退出后只剩 ``astock.db`` 一个文件；
  拷贝迁移前请确认没有进程正持有它（或把 ``-wal``/``-shm`` 一起拷走）。
- 日期一律 TEXT ``YYYY-MM-DD``；价格为**不复权**原始值；成交量单位为**股**。
- 后复权因子按新浪事件日稀疏存储（``hfq_factors``），消费方按日期 forward-fill；
  后复权价 = 原始价 × factor，前复权价 = 原始价 × factor / 最新 factor。
- 财报存长表（每期每科目一行）+ 每期披露元信息（``financial_meta``：公告日/入库
  时间），支持离线的 point-in-time 过滤——谓词与 ``a_stock`` 在线路径共用同一份
  （见 :func:`disclosure_available`），保证两边判定一致。
- 谁建库：只有入库脚本（scripts/ingest_localdb.py）创建/初始化库文件；本工具
  运行时只读写**已存在**的库，没有库就自动走原在线逻辑，行为与不装本地库时相同。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

__all__ = [
    "SCHEMA_VERSION",
    "REPORT_TYPES",
    "get_local_db_path",
    "connect",
    "open_existing",
    "checkpoint_and_close",
    "set_meta",
    "get_meta",
    "upsert_stock_basic",
    "load_stock_basic",
    "upsert_daily_bars",
    "load_daily_bars",
    "last_bar_date",
    "daily_data_is_fresh",
    "upsert_hfq_factors",
    "load_hfq_factors",
    "upsert_financial",
    "load_financial",
    "financial_data_is_fresh",
    "report_deadline",
    "disclosure_available",
]

SCHEMA_VERSION = 1

#: 三张财报的中文名，与新浪接口 report_type 一致（a_stock 也用同一组名字）
REPORT_TYPES = ("资产负债表", "利润表", "现金流量表")

# 与 a_stock 相同的市场时区：判断"今天"必须按上海算，不能用主机时区
_MARKET_TZ = timezone(timedelta(hours=8))

# 新浪给全市场做批量入库的那一天（见 a_stock 的同款注释：2020-03-13 之前的各期
# 共用这一天的入库时间，能证明"那时已公布"，但不证明原始入库时间）
_SINA_BATCH_IMPORT_DAY = pd.Timestamp("2020-03-13")

# 定期报告法定披露截止：一季报/三季报 1 个月，半年报 2 个月，年报 4 个月
_REPORT_DEADLINE_MONTHS = {3: 1, 6: 2, 9: 1, 12: 4}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_basic (
    code       TEXT PRIMARY KEY,   -- 6 位代码，如 600519
    name       TEXT NOT NULL,      -- 最新证券简称
    exchange   TEXT NOT NULL,      -- SH | SZ | BJ
    updated_at TEXT NOT NULL       -- 北京时间 ISO8601
);

CREATE TABLE IF NOT EXISTS daily_bars (
    code       TEXT NOT NULL,
    date       TEXT NOT NULL,      -- YYYY-MM-DD（交易日）
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,               -- 成交量，单位：股
    amount     REAL,               -- 成交额（元）；新浪日线源不提供，恒为 NULL
    updated_at TEXT NOT NULL,
    PRIMARY KEY (code, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date);

CREATE TABLE IF NOT EXISTS hfq_factors (
    code       TEXT NOT NULL,
    date       TEXT NOT NULL,      -- 除权除息事件日
    factor     REAL NOT NULL,      -- 事件日起适用的累计后复权因子
    updated_at TEXT NOT NULL,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS financials (
    code        TEXT NOT NULL,
    report_type TEXT NOT NULL,     -- 资产负债表 | 利润表 | 现金流量表
    report_date TEXT NOT NULL,     -- 报告期 YYYY-MM-DD
    item        TEXT NOT NULL,     -- 科目名（新浪中文科目；缩进科目已带父级前缀）
    ord         INTEGER NOT NULL,  -- 新浪原始顺序，读回时按它还原列顺序
    value       TEXT,              -- 原始字符串，保真；消费方自行 CAST
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (code, report_type, report_date, item)
);

CREATE TABLE IF NOT EXISTS financial_meta (
    code         TEXT NOT NULL,
    report_type  TEXT NOT NULL,
    report_date  TEXT NOT NULL,
    publish_date TEXT,             -- 新浪 publish_date（YYYYMMDD），可能被改写
    update_time  TEXT,             -- 新浪 update_time（epoch 秒），入库/最近改数时间
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (code, report_type, report_date)
);
"""


# ---------------------------------------------------------------------------
# 连接管理
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """北京时间 ISO8601（入库时间戳统一用它，跨主机可比较）。"""
    return datetime.now(_MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _market_today_str() -> str:
    return datetime.now(_MARKET_TZ).strftime("%Y-%m-%d")


def get_local_db_path() -> str:
    """本地库路径：配置 local_db_path > 环境变量 TRADINGAGENTS_LOCAL_DB > 默认。"""
    from .config import get_config

    path = get_config().get("local_db_path")
    if not path:
        path = os.getenv("TRADINGAGENTS_LOCAL_DB")
    if not path:
        path = os.path.join(
            os.path.expanduser("~"), ".tradingagents", "astock.db"
        )
    return path


def connect(path: str | None = None) -> sqlite3.Connection:
    """打开（必要时创建）本地库并初始化 schema。仅入库脚本等写方使用。"""
    path = path or get_local_db_path()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(_SCHEMA)
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    conn.commit()
    return conn


def open_existing(path: str | None = None) -> sqlite3.Connection | None:
    """打开**已存在**的本地库（工具运行时用）；不存在则返回 None，调用方回退在线。

    以读写方式打开但从不建表、不建文件——本工具只把在线取到的数据回写进已
    有库，库本身由入库脚本负责创建。schema 不是最新时也照样用：新表/新列由
    入库脚本升级，这里读不到就当没有。
    """
    path = path or get_local_db_path()
    if not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(path, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        # 读旧库确认可用（文件损坏/非 SQLite 时立刻暴露，而不是在半路抛错）
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn
    except sqlite3.Error:
        return None


def checkpoint_and_close(conn: sqlite3.Connection) -> None:
    """写方收尾：把 WAL 并回主文件再关闭，之后库目录里只剩 .db 一个文件，可整拷。"""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass
    conn.close()


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


# ---------------------------------------------------------------------------
# stock_basic
# ---------------------------------------------------------------------------


def upsert_stock_basic(conn: sqlite3.Connection, rows) -> int:
    """rows: iterable of (code, name, exchange)。返回写入行数。"""
    ts = _now_iso()
    payload = [(c, n, ex, ts) for c, n, ex in rows]
    conn.executemany(
        "INSERT INTO stock_basic(code, name, exchange, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(code) DO UPDATE SET name=excluded.name, "
        "exchange=excluded.exchange, updated_at=excluded.updated_at",
        payload,
    )
    return len(payload)


def load_stock_basic(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT code, name, exchange FROM stock_basic ORDER BY code", conn
    )


# ---------------------------------------------------------------------------
# daily_bars
# ---------------------------------------------------------------------------


def _bars_rows(code: str, df: pd.DataFrame, ts: str):
    dates = pd.to_datetime(df["Date"], errors="coerce")

    def col(name: str):
        return df[name].tolist() if name in df.columns else [None] * len(df)

    zipped = zip(
        dates,
        col("Open"), col("High"), col("Low"), col("Close"),
        col("Volume"), col("Amount"),
    )
    for d, o, h, low, c, v, a in zipped:
        if pd.isna(d):
            continue
        yield (
            code,
            d.strftime("%Y-%m-%d"),
            *(None if (x is None or pd.isna(x)) else float(x) for x in (o, h, low, c, v, a)),
            ts,
        )


def upsert_daily_bars(conn: sqlite3.Connection, code: str, df: pd.DataFrame) -> int:
    """写入日线（df 需含 Date；Open/High/Low/Close/Volume 可缺省为 NULL）。返回行数。"""
    if df is None or df.empty:
        return 0
    df = df.reset_index(drop=True)
    rows = list(_bars_rows(code, df, _now_iso()))
    conn.executemany(
        "INSERT INTO daily_bars(code, date, open, high, low, close, volume, amount, "
        "updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(code, date) DO UPDATE SET open=excluded.open, high=excluded.high, "
        "low=excluded.low, close=excluded.close, volume=excluded.volume, "
        "amount=excluded.amount, updated_at=excluded.updated_at",
        rows,
    )
    return len(rows)


def load_daily_bars(
    conn: sqlite3.Connection,
    code: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """读日线，返回 Date(已归一化为日期) + Open/High/Low/Close/Volume/Amount。"""
    sql = "SELECT date, open, high, low, close, volume, amount FROM daily_bars WHERE code=?"
    params: list = [code]
    if start:
        sql += " AND date>=?"
        params.append(pd.to_datetime(start).strftime("%Y-%m-%d"))
    if end:
        sql += " AND date<=?"
        params.append(pd.to_datetime(end).strftime("%Y-%m-%d"))
    sql += " ORDER BY date"
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "amount": "Amount",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])


def last_bar_date(conn: sqlite3.Connection, code: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(date) FROM daily_bars WHERE code=?", (code,)
    ).fetchone()
    return row[0] if row and row[0] else None


def daily_data_is_fresh(
    conn: sqlite3.Connection, code: str, need_date: str | None
) -> bool:
    """本地日线能否直接满足"要看到 need_date"的请求。

    与 a_stock 的 CSV 缓存同款口径（mtime 是今天就信）：
    - 库里最新一根 >= need_date：历史分析日或已收盘补数，全覆盖；
    - 否则，只要今天（北京时间）刷新过这只票，就当和在线一样新——盘中 need_date
      是今天时，谁也拿不到收盘 bar，在线源同样只有半根。
    """
    last = last_bar_date(conn, code)
    if last is None:
        return False
    if need_date and last >= pd.to_datetime(need_date).strftime("%Y-%m-%d"):
        return True
    row = conn.execute(
        "SELECT MAX(updated_at) FROM daily_bars WHERE code=?", (code,)
    ).fetchone()
    if not row or not row[0]:
        return False
    return str(row[0])[:10] >= _market_today_str()


# ---------------------------------------------------------------------------
# hfq_factors
# ---------------------------------------------------------------------------


def upsert_hfq_factors(
    conn: sqlite3.Connection, code: str, events: list[tuple[str, float]]
) -> int:
    """events: [(事件日 YYYY-MM-DD, 累计后复权因子)]，只存事件日，读方按日 ffill。"""
    ts = _now_iso()
    rows = [
        (code, pd.to_datetime(d).strftime("%Y-%m-%d"), float(f), ts)
        for d, f in events
        if f is not None and not pd.isna(f)
    ]
    conn.executemany(
        "INSERT INTO hfq_factors(code, date, factor, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(code, date) DO UPDATE SET factor=excluded.factor, "
        "updated_at=excluded.updated_at",
        rows,
    )
    return len(rows)


def load_hfq_factors(conn: sqlite3.Connection, code: str) -> pd.DataFrame:
    """按事件日升序返回 date/factor；消费方对日线日期 ffill 后即可乘出后复权价。"""
    return pd.read_sql_query(
        "SELECT date, factor FROM hfq_factors WHERE code=? ORDER BY date",
        conn,
        params=(code,),
    )


# ---------------------------------------------------------------------------
# 财报（长表 + 披露元信息）
# ---------------------------------------------------------------------------


def upsert_financial(
    conn: sqlite3.Connection, code: str, report_type: str, df: pd.DataFrame
) -> int:
    """把 a_stock 解析出的新浪财报宽表落库（含 _公告日/_入库时间 两个内部列）。

    同一 (code, report_type, report_date) 的旧科目先删再插——两次抓取之间科目名
    可能变化，只靠主键覆盖会留下已消失的旧行。
    """
    if df is None or df.empty or "报告日" not in df.columns:
        return 0
    df = df.reset_index(drop=True)
    ts = _now_iso()
    item_cols = [c for c in df.columns if not c.startswith("_") and c != "报告日"]
    n_rows = 0
    for idx, report_day in enumerate(pd.to_datetime(df["报告日"], errors="coerce")):
        if pd.isna(report_day):
            continue
        rd = report_day.strftime("%Y-%m-%d")
        publish = (
            str(df.at[idx, "_公告日"])
            if "_公告日" in df.columns and pd.notna(df.at[idx, "_公告日"])
            else None
        )
        update = (
            str(df.at[idx, "_入库时间"])
            if "_入库时间" in df.columns and pd.notna(df.at[idx, "_入库时间"])
            else None
        )
        conn.execute(
            "INSERT INTO financial_meta(code, report_type, report_date, publish_date, "
            "update_time, ingested_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(code, report_type, report_date) DO UPDATE SET "
            "publish_date=excluded.publish_date, update_time=excluded.update_time, "
            "ingested_at=excluded.ingested_at",
            (code, report_type, rd, publish, update, ts),
        )
        conn.execute(
            "DELETE FROM financials WHERE code=? AND report_type=? AND report_date=?",
            (code, report_type, rd),
        )
        rows = []
        for ord_, col in enumerate(item_cols):
            v = df.at[idx, col]
            rows.append(
                (code, report_type, rd, col, ord_, None if pd.isna(v) else str(v), ts)
            )
        conn.executemany(
            "INSERT INTO financials(code, report_type, report_date, item, ord, value, "
            "updated_at) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        n_rows += len(rows)
    return n_rows


def financial_data_is_fresh(
    conn: sqlite3.Connection, code: str, report_type: str
) -> bool:
    """今天（北京时间）入库过该 (code, report_type) 就算新鲜，直接用本地。"""
    try:
        row = conn.execute(
            "SELECT MAX(ingested_at) FROM financial_meta WHERE code=? AND report_type=?",
            (code, report_type),
        ).fetchone()
    except sqlite3.Error:
        return False
    if not row or not row[0]:
        return False
    return str(row[0])[:10] >= _market_today_str()


def load_financial(
    conn: sqlite3.Connection,
    code: str,
    report_type: str,
    curr_date: str | None = None,
) -> pd.DataFrame | None:
    """读回报表宽表（报告日 + 科目列，科目顺序与新浪一致）；库里没有则 None。

    curr_date 给定时用 financial_meta 里存的公告日/入库时间做离线 point-in-time
    过滤（与在线路径同一谓词）。返回**全部**报告期（新在前），head(8)/annual 过滤
    由调用方做，与在线行为对齐。
    """
    long_df = pd.read_sql_query(
        "SELECT report_date, item, ord, value FROM financials "
        "WHERE code=? AND report_type=?",
        conn,
        params=(code, report_type),
    )
    if long_df.empty:
        return None
    meta_df = pd.read_sql_query(
        "SELECT report_date, publish_date, update_time FROM financial_meta "
        "WHERE code=? AND report_type=?",
        conn,
        params=(code, report_type),
    )

    periods = sorted(long_df["report_date"].unique(), reverse=True)
    ord_by_item = (
        long_df.drop_duplicates("item").set_index("item")["ord"].to_dict()
    )
    item_cols = sorted(ord_by_item, key=lambda c: ord_by_item[c])

    rows = []
    for rd in periods:
        sub = long_df[long_df["report_date"] == rd].set_index("item")["value"]
        row = {"报告日": pd.Timestamp(rd), "报告日_str": rd}
        for col in item_cols:
            row[col] = sub.get(col)
        rows.append(row)
    df = pd.DataFrame(rows)

    if curr_date:
        meta_map = meta_df.set_index("report_date")
        df["_公告日"] = df["报告日_str"].map(meta_map["publish_date"])
        df["_入库时间"] = df["报告日_str"].map(meta_map["update_time"])
        available = _disclosure_filter_from_raw(df, curr_date)
        df = df[available]

    df = df.sort_values("报告日", ascending=False).reset_index(drop=True)
    return df[["报告日"] + item_cols]


# ---------------------------------------------------------------------------
# 披露判定（与 a_stock 在线路径共用的纯逻辑）
# ---------------------------------------------------------------------------


def report_deadline(report_date: pd.Series) -> pd.Series:
    """报告期 → 法定披露截止日（报告期都是月末，按月末顺延）；非标准报告期保守按 4 个月。"""
    deadline = report_date + pd.offsets.MonthEnd(4)
    for month, lag in _REPORT_DEADLINE_MONTHS.items():
        hit = report_date.dt.month == month
        deadline[hit] = report_date[hit] + pd.offsets.MonthEnd(lag)
    return deadline


def _parse_publish(raw: pd.Series) -> pd.Series:
    return pd.to_datetime(raw, errors="coerce", format="%Y%m%d")


def _parse_update(raw: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(pd.to_numeric(raw, errors="coerce"), unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )


def _disclosure_filter_from_raw(df: pd.DataFrame, curr_date: str) -> pd.Series:
    """对带 报告日/_公告日/_入库时间 列的宽表做 point-in-time 过滤（布尔 Series）。"""
    report = pd.to_datetime(df["报告日"], errors="coerce")
    nat = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    published = (
        _parse_publish(df["_公告日"]) if "_公告日" in df.columns else nat
    )
    updated = (
        _parse_update(df["_入库时间"]) if "_入库时间" in df.columns else nat
    )
    return disclosure_available(
        report, published, updated, pd.to_datetime(curr_date)
    )


def disclosure_available(
    report: pd.Series,
    published: pd.Series,
    updated: pd.Series,
    cutoff: pd.Timestamp,
) -> pd.Series:
    """报告期在 cutoff 当天是否"已公布"。语义与 a_stock 在线路径逐条一致：

    - 公告日早于分析日，或入库时间早于分析日当天 → 已公布（公告日当天收盘后才
      发的也有，所以要严格早于）；
    - 两个日期都晚于报告期一年以上（或都没有）→ 原始公布日已被改写丢失，只能
      假定按时披露，过了法定截止日才算已公布；
    - 新浪 2020-03-13 批量入库时间不算原始入库时间（只认这天、且 3 期及以上共用，
      否则逾期公司一次补发几期会被当成批量日，提前露出）；
    - 公告日不晚于报告期的（上市前数据有写成报告期当天的）不可信，当作没有。
    """
    published = published.where(published > report)
    batch = (updated.dt.normalize() == _SINA_BATCH_IMPORT_DAY) & (
        updated.map(updated.value_counts()) >= 3
    )
    year_later = report + pd.DateOffset(years=1)
    original_lost = (published.isna() | (published >= year_later)) & (
        updated.isna() | (updated >= year_later) | batch
    )
    return (
        (published < cutoff)
        | (updated < cutoff)
        | (original_lost & (report_deadline(report) < cutoff))
    )
