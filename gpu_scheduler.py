# -*- coding: utf-8 -*-
"""CRANE-Z GPU dynamic scheduler (multi-process CUDA contention guard).

Purpose: when multiple processes compete for the same RTX 5080 (16 GB),
decide dynamically whether a new GPU task may start, based on live
nvidia-smi memory/process state. Prevents OOM crashes and wasteful
re-runs (H31: input validation, observable progress, no blind runs).

Usage:
    # as a launcher (recommended): wait until >= MIN_FREE_GB free, then run cmd
    python gpu_scheduler.py run --min-free-gb 4 --cmd "python train_v2.py --all"

    # as a check (exit 0 if safe, 1 if busy)
    python gpu_scheduler.py check --min-free-gb 4

    # query current state
    python gpu_scheduler.py status

Mechanism:
  - Polls nvidia-smi every POLL_SEC (default 20 s), timeout TIME_OUT s.
  - Holds a lock file (LOCK_PATH) so only one launcher runs at a time;
    other launchers queue (wait for the lock, then re-check memory).
  - Writes JSON state snapshots to gpu_scheduler_state.json for observability.
  - Runs the command via subprocess and streams output; exit code propagated.
"""
import argparse
import json
import os
import subprocess
import sys
import time

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "gpu_scheduler_state.json")
LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "gpu_scheduler.lock")

NVIDIA_SMI = "nvidia-smi"


def query_gpu():
    """Return dict with memory used/free and process count via nvidia-smi."""
    try:
        out = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        procs = subprocess.run(
            [NVIDIA_SMI, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        n_procs = len([l for l in procs.stdout.strip().splitlines() if l.strip()]) if procs.returncode == 0 else -1
        return {
            "mem_used_mb": int(parts[0]),
            "mem_total_mb": int(parts[1]),
            "mem_free_mb": int(parts[2]),
            "util_pct": int(parts[3]),
            "gpu_processes": n_procs,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def snapshot(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _pid_alive(pid):
    """Check whether a pid is alive (works on Windows and POSIX)."""
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def acquire_lock(timeout=3600, poll=10):
    """Lock file with stale-lock recovery:
    lock held by a DEAD pid or older than 30 min -> removed immediately;
    otherwise wait for the holder to release.
    """
    deadline = time.time() + timeout
    while True:
        if os.path.exists(LOCK_PATH):
            holder = ""
            stale = False
            try:
                with open(LOCK_PATH) as f:
                    holder = f.read().strip()
                if holder and not _pid_alive(holder):
                    stale = True
            except OSError:
                pass
            age = time.time() - os.path.getmtime(LOCK_PATH)
            if stale or age > 1800:  # dead holder or stale lock
                try:
                    os.remove(LOCK_PATH)
                    print(f"[gpu_scheduler] stale lock removed "
                          f"(holder {holder or '?'} dead / age {age:.0f}s)", flush=True)
                    continue
                except OSError:
                    pass
            if time.time() > deadline:
                return False
            print(f"[gpu_scheduler] lock busy (holder {holder or '?'}, "
                  f"age {age:.0f}s), waiting...", flush=True)
            time.sleep(poll)
            continue
        try:
            with open(LOCK_PATH, "w") as f:
                f.write(str(os.getpid()))
            return True
        except OSError:
            time.sleep(poll)


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def wait_for_gpu(min_free_gb, timeout, poll):
    """Poll until GPU has >= min_free_gb free, or timeout. Returns bool."""
    min_free_mb = int(min_free_gb * 1024)
    deadline = time.time() + timeout
    while True:
        st = query_gpu()
        if st and "error" not in st:
            free_mb = st["mem_free_mb"]
            snapshot({"waiting": True, "min_free_gb": min_free_gb, "last": st})
            print(f"[gpu_scheduler] mem free {free_mb:.0f} MB / {st['mem_total_mb']} MB, "
                  f"procs {st['gpu_processes']}, util {st['util_pct']}%", flush=True)
            if free_mb >= min_free_mb:
                return True
        else:
            print(f"[gpu_scheduler] nvidia-smi unavailable ({st.get('error', '?') if st else '?'}); "
                  f"assuming GPU is free", flush=True)
            return True
        if time.time() > deadline:
            return False
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description="CRANE-Z GPU dynamic scheduler")
    sub = ap.add_subparsers(dest="subcommand", required=True)

    p_run = sub.add_parser("run", help="wait for GPU then run a command")
    p_run.add_argument("--min-free-gb", type=float, default=4.0)
    p_run.add_argument("--timeout", type=int, default=3600)
    p_run.add_argument("--poll", type=int, default=20)
    p_run.add_argument("--cmd", nargs=argparse.REMAINDER, required=True)

    p_chk = sub.add_parser("check", help="exit 0 if GPU free enough, else 1")
    p_chk.add_argument("--min-free-gb", type=float, default=4.0)

    p_st = sub.add_parser("status", help="print current GPU state")

    args = ap.parse_args()

    if args.subcommand == "status":
        st = query_gpu()
        print(json.dumps(st, indent=2, ensure_ascii=False))
        snapshot({"cmd": "status", "last": st})
        return 0

    if args.subcommand == "check":
        st = query_gpu()
        if st and "error" not in st:
            ok = st["mem_free_mb"] >= int(args.min_free_gb * 1024)
            print(f"GPU {'FREE' if ok else 'BUSY'}: {st['mem_free_mb']} MB free "
                  f"({st['gpu_processes']} procs)")
            return 0 if ok else 1
        print("GPU unknown (nvidia-smi failed), treating as FREE")
        return 0

    if args.subcommand == "run":
        if not acquire_lock(timeout=args.timeout):
            print("[gpu_scheduler] could not acquire lock within timeout; aborting", flush=True)
            return 2
        try:
            ok = wait_for_gpu(args.min_free_gb, args.timeout, args.poll)
            if not ok:
                print(f"[gpu_scheduler] GPU not free within {args.timeout}s; aborting", flush=True)
                return 2
            print(f"[gpu_scheduler] GPU free >= {args.min_free_gb} GB; launching: "
                  f"{' '.join(args.cmd)}", flush=True)
            snapshot({"cmd": "run", "launched": " ".join(args.cmd), "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
            rc = subprocess.run(args.cmd).returncode
            print(f"[gpu_scheduler] command finished with exit {rc}", flush=True)
            snapshot({"cmd": "run", "finished": True, "exit": rc,
                      "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
            return rc
        finally:
            release_lock()

    return 0


if __name__ == "__main__":
    sys.exit(main())
