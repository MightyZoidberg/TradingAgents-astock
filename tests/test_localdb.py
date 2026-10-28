# -*- coding: utf-8 -*-
"""localdb（本地 A 股 SQLite 库）单元测试。"""
import pytest
import pandas as pd

from tradingagents.dataflows import localdb


@pytest.fixture
def db(tmp_path):
    conn = localdb.connect(str(tmp_path / "astock.db"))
    yield conn
    localdb.checkpoint_and_close(conn)


def test_connect_creates_all_tables(db):
    names = {
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "meta", "stock_basic", "daily_bars", "hfq_factors",
        "financials", "financial_meta",
    } <= names
    assert localdb.get_meta(db, "schema_version") == "1"


def test_open_existing_returns_none_when_missing(tmp_path):
    assert localdb.open_existing(str(tmp_path / "nope.db")) is None


def test_daily_roundtrip_and_freshness(db):
    df = pd.DataFrame({
        "Date": ["2026-10-26", "2026-10-27"],
        "Open": [10.0, 10.5], "High": [10.8, 10.9],
        "Low": [9.9, 10.2], "Close": [10.5, 10.7],
        "Volume": [1_000_000, 1_200_000],
    })
    assert localdb.upsert_daily_bars(db, "600519", df) == 2

    out = localdb.load_daily_bars(db, "600519")
    assert list(out["Close"]) == [10.5, 10.7]
    assert str(out["Date"].iloc[-1].date()) == "2026-10-27"
    # Amount 列存在但为 NULL（新浪日线源不给成交额）
    assert out["Amount"].isna().all()

    assert localdb.last_bar_date(db, "600519") == "2026-10-27"
    # 覆盖到请求日 → 新鲜；请求日超出库存且今天没刷过 → 只有当请求日<=今天才看 updated_at
    assert localdb.daily_data_is_fresh(db, "600519", "2026-10-27")
    assert localdb.daily_data_is_fresh(db, "600519", "2026-10-26")
    # 没有任何数据的股票
    assert not localdb.daily_data_is_fresh(db, "000001", "2026-10-27")


def test_daily_upsert_replaces_changed_row(db):
    df1 = pd.DataFrame({"Date": ["2026-10-26"], "Close": [10.5], "Volume": [100]})
    localdb.upsert_daily_bars(db, "600519", df1)
    df2 = pd.DataFrame({"Date": ["2026-10-26"], "Close": [11.0], "Volume": [200]})
    localdb.upsert_daily_bars(db, "600519", df2)
    out = localdb.load_daily_bars(db, "600519")
    assert len(out) == 1 and out["Close"].iloc[0] == 11.0


def test_hfq_factor_roundtrip(db):
    events = [("2010-06-30", 1.5), ("2020-06-24", 4.25), ("2026-06-26", 8.88)]
    assert localdb.upsert_hfq_factors(db, "600519", events) == 3
    out = localdb.load_hfq_factors(db, "600519")
    assert list(out["date"]) == ["2010-06-30", "2020-06-24", "2026-06-26"]
    assert list(out["factor"]) == [1.5, 4.25, 8.88]


def _beijing_epoch(day: str, hm: str = "19:00") -> str:
    return str(int(pd.Timestamp(f"{day} {hm}", tz="Asia/Shanghai").timestamp()))


def _sample_statements():
    """两期利润表：2024 年报 2025-03-28 公布；2025 一季报 2025-04-25 晚入库。"""
    return pd.DataFrame({
        "报告日": ["2025-03-31", "2024-12-31"],
        "营业总收入": ["500.0", "1700.0"],
        "财务费用-利息收入": ["1.2", "5.0"],
        "_公告日": [None, "20250328"],
        "_入库时间": [_beijing_epoch("2025-04-25"), _beijing_epoch("2025-03-27")],
    })


def test_financial_roundtrip_preserves_item_order(db):
    localdb.upsert_financial(db, "600519", "利润表", _sample_statements())
    out = localdb.load_financial(db, "600519", "利润表")
    # 新报告期在前，科目顺序与写入时一致（营业总收入 在 财务费用-利息收入 前）
    assert list(out["报告日"].dt.strftime("%Y-%m-%d")) == [
        "2025-03-31", "2024-12-31",
    ]
    assert list(out.columns) == ["报告日", "营业总收入", "财务费用-利息收入"]
    assert out["营业总收入"].iloc[0] == "500.0"


def test_financial_offline_disclosure_filter(db):
    localdb.upsert_financial(db, "600519", "利润表", _sample_statements())

    # 分析日在一季报入库之前：只应看到年报
    out = localdb.load_financial(db, "600519", "利润表", curr_date="2025-04-01")
    assert list(out["报告日"].dt.strftime("%Y-%m-%d")) == ["2024-12-31"]

    # 分析日之后：两期都可见
    out = localdb.load_financial(db, "600519", "利润表", curr_date="2025-04-30")
    assert len(out) == 2

    # 内部日期列不外泄
    assert "_公告日" not in out.columns and "_入库时间" not in out.columns


def test_financial_missing_returns_none(db):
    assert localdb.load_financial(db, "000002", "资产负债表") is None


def test_stock_basic_roundtrip(db):
    localdb.upsert_stock_basic(
        db, [("600519", "贵州茅台", "SH"), ("920000", "安徽凤凰", "BJ")]
    )
    out = localdb.load_stock_basic(db)
    assert len(out) == 2 and out.iloc[1]["exchange"] == "BJ"
