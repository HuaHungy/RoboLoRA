"""Log-friendly progress and monotonic wall-clock estimates; no extra dependencies."""

from contextlib import contextmanager
from datetime import datetime, timedelta
from threading import Event, Thread
from time import monotonic


def duration(seconds):
    if 0 <= seconds < 60:
        return f"{seconds:.2f} 秒"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self, label, total, unit, interval=10):
        self.label, self.total, self.unit = label, total, unit
        self.interval = interval
        self.started = self.last = monotonic()
        self.done = 0
        print(f"[{label}] 0/{total} {unit} | 预计剩余: 采样计时中", flush=True)

    def update(self, done, force=False):
        self.done = done
        now = monotonic()
        if not force and done != self.total and now - self.last < self.interval:
            return
        self.last = now
        elapsed = now - self.started
        remaining = elapsed / done * max(0, self.total - done) if done else None
        eta = duration(remaining) if remaining is not None else "采样计时中"
        finish = (datetime.now() + timedelta(seconds=remaining)).strftime("%m-%d %H:%M:%S") if remaining is not None else "待估计"
        pct = 100 * done / self.total if self.total else 100
        print(f"[{self.label}] {done}/{self.total} {self.unit} ({pct:.1f}%) | "
              f"已用 {duration(elapsed)} | 预计本阶段剩余 {eta} | 预计结束 {finish}", flush=True)


@contextmanager
def timed(label, estimate=None, interval=15):
    """Heartbeat for opaque calls (e.g. sklearn fit), including failure cleanup."""
    started = monotonic()
    stopped = Event()
    hint = f"粗估耗时 {duration(estimate)}（按前一候选 K 外推）" if estimate is not None else "耗时待实测"
    print(f"[{label}] 开始 | {hint}", flush=True)

    def heartbeat():
        while not stopped.wait(interval):
            elapsed = monotonic() - started
            remaining = f"粗估剩余 {duration(estimate - elapsed)}" if estimate is not None and elapsed < estimate else "仍在计算，剩余时间待估计"
            print(f"[{label}] 已用 {duration(elapsed)} | {remaining}", flush=True)

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    success = False
    try:
        yield
        success = True
    finally:
        stopped.set()
        worker.join()
        print(f"[{label}] {'完成' if success else '中断/失败'} | 耗时 {duration(monotonic() - started)}", flush=True)
