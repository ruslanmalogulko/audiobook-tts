"""Запускает команду и убивает её (со всеми потомками), если она съедает слишком много памяти.

  uv run memguard.py --max-gb 12 --min-free-gb 6 -- <команда ...>

Печатает пик памяти. Нужен, чтобы локальные ML-модели не вешали Mac (было: 56 GB -> перезагрузка).
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time


def tree_rss_bytes(root_pid: int) -> int:
    """Суммарный RSS процесса и всех потомков (по ps)."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children, rss = {}, {}
    for line in out.splitlines():
        pid, ppid, kb = map(int, line.split())
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb * 1024
    total, stack = 0, [root_pid]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack += children.get(pid, [])
    return total


def system_available_bytes() -> int:
    """Свободные + неактивные + очищаемые страницы (то, что система отдаст без своппинга)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {k: int(v) for k, v in re.findall(r'"?Pages ([\w ]+)"?:\s+(\d+)', out)}
    return page * (pages.get("free", 0) + pages.get("inactive", 0) + pages.get("purgeable", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gb", type=float, default=12)
    parser.add_argument("--min-free-gb", type=float, default=6)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command

    process = subprocess.Popen(command, start_new_session=True)
    peak, started, reason = 0, time.time(), None
    while process.poll() is None:
        used = tree_rss_bytes(process.pid)
        peak = max(peak, used)
        free = system_available_bytes()
        if used > args.max_gb * 2**30:
            reason = f"процесс занял {used / 2**30:.1f} GB > {args.max_gb} GB"
        elif free < args.min_free_gb * 2**30:
            reason = f"в системе осталось {free / 2**30:.1f} GB < {args.min_free_gb} GB"
        if reason:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            break
        time.sleep(0.5)
    elapsed = time.time() - started
    print(f"[memguard] пик {peak / 2**30:.2f} GB, {elapsed:.0f} с, код {process.returncode}"
          + (f", УБИТ: {reason}" if reason else ""), file=sys.stderr)
    sys.exit(97 if reason else process.returncode)


if __name__ == "__main__":
    main()
