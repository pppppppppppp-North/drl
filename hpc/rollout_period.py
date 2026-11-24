"""Time a running PPO task's iterations without touching it, from its memory maps.

Prints, for each of the user's matching python processes, when a large anonymous mapping appears or
disappears. For the point-in-time design (PPO, n_steps 128, batch 64, 10 epochs) each minibatch update
maps and frees about 230 MB, so the 20 updates of one iteration show as 20 short mappings, followed by a
gap of about 1.2 s with none while the next 128 environment steps are collected. The time between gaps
is the time per iteration. Job 4409, 2026-11-24: 11.0-13.3 s, so 2,344 iterations (300k steps) take
7.2-8.7 hours. It only reads /proc/<pid>/maps. Run it on the compute node with its absolute path:
    python3 -I /abs/path/hpc/rollout_period.py <pattern> <seconds> [interval s] [max pids] [min MB]
"""
import os
import re
import sys
import time

pattern = sys.argv[1].encode()
duration = float(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
max_pids = int(sys.argv[4]) if len(sys.argv) > 4 else 2
min_bytes = int(float(sys.argv[5]) * 2**20) if len(sys.argv) > 5 else 20 * 2**20
header = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")


def pids():
    found = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if os.stat(f"/proc/{p}").st_uid != os.getuid():
                continue
            args = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if args and args[0].endswith(b"python") and any(pattern in a for a in args):
            found.append(int(p))
    return sorted(found)[:max_pids]


def mappings(pid):
    """Large anonymous mappings, from /proc/<pid>/maps (cheap enough to read several times a second)."""
    out = set()
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            m = header.match(line)
            start, end = int(m.group(1), 16), int(m.group(2), 16)
            if end - start >= min_bytes and len(line.split()) == 5:
                out.add((start, end))
    return out


chosen = pids()
print("pids", chosen, flush=True)
seen = {p: mappings(p) for p in chosen}
t0 = time.time()
for p in chosen:
    for s, e in sorted(seen[p]):
        print(f"{p} {0:8.1f}s have {s:x} {(e - s) / 2**20:8.1f} MB", flush=True)
while time.time() - t0 < duration:
    time.sleep(interval)
    t = time.time() - t0
    for p in chosen:
        try:
            now = mappings(p)
        except OSError:
            print(f"{p} {t:8.1f}s gone", flush=True)
            continue
        for k in sorted(now - seen[p]):
            print(f"{p} {t:8.1f}s new  {k[0]:x} {(k[1] - k[0]) / 2**20:8.1f} MB", flush=True)
        for k in sorted(seen[p] - now):
            print(f"{p} {t:8.1f}s gone {k[0]:x} {(k[1] - k[0]) / 2**20:8.1f} MB", flush=True)
        seen[p] = now
