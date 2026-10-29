# -*- coding: utf-8 -*-
"""TradingAgents Web UI 托盘启动器（无控制台窗口）。

用法（快捷方式即指向此文件）::

    pythonw.exe web\\tray.pyw

行为：
- 启动 Streamlit 子进程（headless，日志写 ~/.tradingagents/logs/web-server.log）
- 健康检查就绪后自动打开默认浏览器
- 托盘区常驻图标：双击 = 打开界面；右键菜单 = 状态 / 打开界面 /
  打开日志目录 / 重启服务 / 退出
- 单实例保护：占用固定管理端口；重复启动时只把已有界面在浏览器打开
- 「退出」= 结束 Streamlit 进程树 + 退出托盘

冒烟测试可用环境变量 TRADINGAGENTS_TRAY_NO_BROWSER=1 抑制自动开浏览器。
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "web" / "app.py"
ICON_PATH = PROJECT_ROOT / "assets" / "tray.ico"
LOG_DIR = Path.home() / ".tradingagents" / "logs"
SERVER_LOG = LOG_DIR / "web-server.log"
STATE_FILE = Path.home() / ".tradingagents" / "tray-state.json"

PREFERRED_PORTS = range(8501, 8511)   # 依次尝试；全被占则让系统随机分配
MANAGER_PORT = 48599                  # 单实例互斥用；勿与常用端口冲突
CREATE_NO_WINDOW = 0x08000000
NO_BROWSER = os.getenv("TRADINGAGENTS_TRAY_NO_BROWSER") == "1"

# phase: starting → running → restarting / degraded / stopped
_state = {"phase": "starting", "url": ""}
_proc: subprocess.Popen | None = None
_icon = None

# 先建目录再配日志，否则全新机器上首次运行在 import 阶段就会失败
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "tray-launcher.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",  # 不指定时 pythonw 会用系统 GBK，中文日志变乱码
)
log = logging.getLogger("tray")


# ---------------------------------------------------------------------------
# 图标：深色圆角底 + 绿/红 K 线三根（纯 Pillow 绘制，首次运行生成 .ico）
# ---------------------------------------------------------------------------

def _draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = max(2, size // 6)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=(24, 26, 32, 255))
    # 三根 K 线：阳-阴-阳；坐标按 size 比例缩放
    def candle(x_frac, h_frac, up: bool):
        cx = int(size * x_frac)
        half = max(1, int(size * 0.09))
        body_top = int(size * (0.5 - h_frac / 2))
        body_bot = int(size * (0.5 + h_frac / 2))
        wick_top, wick_bot = body_top - half * 2, body_bot + half * 2
        color = (46, 204, 113, 255) if up else (231, 76, 60, 255)
        d.line([cx, wick_top, cx, wick_bot], fill=color, width=max(1, size // 32))
        d.rectangle([cx - half, body_top, cx + half, body_bot], fill=color)

    candle(0.28, 0.42, True)
    candle(0.50, 0.34, False)
    candle(0.72, 0.50, True)
    return img


def ensure_icon_file() -> Image.Image:
    """生成 assets/tray.ico（多尺寸）并返回托盘用的位图。"""
    img = _draw_icon(256)
    try:
        if not ICON_PATH.exists():
            ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
            img.save(ICON_PATH, sizes=[(s, s) for s in (256, 128, 64, 48, 32, 16)])
    except OSError as e:
        log.warning("写入图标文件失败（不影响运行）: %s", e)
    return img.resize((64, 64))


# ---------------------------------------------------------------------------
# 服务管理
# ---------------------------------------------------------------------------

def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def pick_port() -> int:
    for port in PREFERRED_PORTS:
        if _port_free(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server() -> int:
    """启动 Streamlit 子进程，返回端口。调用方负责后续健康检查。"""
    global _proc
    port = pick_port()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(SERVER_LOG, "ab")
    _proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH),
         "--server.headless", "true",
         "--server.address", "127.0.0.1",
         "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=str(PROJECT_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
    )
    log.info("streamlit 已启动 pid=%s port=%s", _proc.pid, port)
    return port


def wait_ready(port: int, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.monotonic() < deadline:
        if _proc is not None and _proc.poll() is not None:
            log.error("streamlit 启动过程中退出 code=%s", _proc.returncode)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            time.sleep(0.3)
    return False


def stop_server() -> None:
    global _proc
    if _proc is None:
        return
    if _proc.poll() is None:
        # /T 连子进程一起结束；进程树杀掉才不会留下孤儿 streamlit
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(_proc.pid)],
            capture_output=True, creationflags=CREATE_NO_WINDOW,
        )
    _proc = None
    log.info("streamlit 已停止")


def _set_phase(phase: str, url: str = "") -> None:
    _state["phase"], _state["url"] = phase, url or _state["url"]
    if _icon is not None:
        try:
            _icon.update_menu()
        except Exception as e:  # noqa: BLE001 - 菜单刷新失败不致命
            log.warning("update_menu 失败: %s", e)


def _write_state(url: str) -> None:
    try:
        STATE_FILE.write_text(
            json.dumps({"url": url, "pid": os.getpid()}), encoding="utf-8"
        )
    except OSError as e:
        log.warning("写状态文件失败: %s", e)


def _serve_cycle(open_browser: bool) -> None:
    """一条完整的服务生命周期线程：启动 → 等就绪 → 开浏览器 → 监视存活。"""
    port = start_server()
    if not wait_ready(port):
        _set_phase("degraded")
        log.error("服务未就绪，详见 %s", SERVER_LOG)
        return
    url = f"http://127.0.0.1:{port}"
    _write_state(url)
    _set_phase("running", url)
    log.info("服务就绪 %s", url)
    try:
        _icon.notify(f"服务已就绪：{url}")
    except Exception:  # noqa: BLE001 - 通知失败无所谓
        pass
    if open_browser and not NO_BROWSER:
        webbrowser.open(url)
    while _proc is not None and _proc.poll() is None:
        time.sleep(2)
    if _state["phase"] == "running":
        _set_phase("degraded")
        log.error("streamlit 意外退出，可从托盘菜单「重启服务」恢复")


# ---------------------------------------------------------------------------
# 托盘
# ---------------------------------------------------------------------------

def _status_text(_item) -> str:
    phase = _state["phase"]
    if phase == "running":
        return f"运行中 · {_state['url']}"
    return {"starting": "启动中…", "restarting": "重启中…",
            "degraded": "服务异常（看日志）", "stopped": "已停止"}.get(phase, phase)


def _open_ui(_item=None):
    if _state["url"]:
        webbrowser.open(_state["url"])


def _open_logs(_item=None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["explorer", str(LOG_DIR)], creationflags=CREATE_NO_WINDOW)


def _restart(_item=None):
    def worker():
        _set_phase("restarting")
        stop_server()
        _serve_cycle(open_browser=False)

    threading.Thread(target=worker, daemon=True).start()


def _quit(_item=None):
    # 先让托盘消失（icon.run 返回），进程树在 finally 里收尾
    if _icon is not None:
        _icon.stop()


def _acquire_manager_lock() -> bool:
    """占用管理端口作为单实例互斥；失败说明已有实例。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", MANAGER_PORT))
        s.listen(1)
    except OSError:
        s.close()
        return False
    # 故意不 close：套接字存活期间锁一直持有
    globals()["_manager_socket"] = s
    return True


def _focus_existing_instance() -> None:
    """第二个实例：把已有界面打开到前台后退出。"""
    url = f"http://127.0.0.1:{PREFERRED_PORTS.start}"
    try:
        url = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("url", url)
    except (OSError, ValueError):
        pass
    webbrowser.open(url)


def main() -> None:
    global _icon
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("===== 托盘启动器 pid=%s =====", os.getpid())

    if not _acquire_manager_lock():
        log.info("已有实例在运行，转而打开界面后退出")
        _focus_existing_instance()
        return

    menu = Menu(
        MenuItem(_status_text, None, enabled=False),
        Menu.SEPARATOR,
        MenuItem("打开界面", _open_ui, default=True),
        MenuItem("打开日志目录", _open_logs),
        MenuItem("重启服务", _restart),
        Menu.SEPARATOR,
        MenuItem("退出", _quit),
    )
    _icon = Icon("TradingAgents", ensure_icon_file(), "TradingAgents Web UI", menu)

    def setup(icon):
        icon.visible = True
        threading.Thread(
            target=_serve_cycle, args=(True,), daemon=True
        ).start()

    try:
        _icon.run(setup=setup)
    finally:
        _set_phase("stopped")
        stop_server()
        try:
            STATE_FILE.unlink()
        except OSError:
            pass
        log.info("托盘启动器退出")


if __name__ == "__main__":
    main()
