"""Measure how fast each SAC/TD3/DDPG worker writes its replay buffer (steps per second).

Finds the observation array (a 939,840 kB anonymous mapping: 100,000 steps x 9,624 bytes), clears the
processes' soft-dirty bits, waits, and reads /proc/<pid>/pagemap to see which pages were written since.
Only touches the user's own processes; the clear costs one minor page fault per page written next.
Usage: python3 buffer_rate.py <wait seconds> [interval count]
"""
import os, sys, time, struct, glob

STEP = 9624
SIZE = 100_000 * STEP
PAGE = 4096
wait = float(sys.argv[1]) if len(sys.argv) > 1 else 300
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
uid = os.getuid()

def buffers():
    out = {}
    for d in glob.glob("/proc/[0-9]*"):
        try:
            if os.stat(d).st_uid != uid or b"python" not in open(d + "/comm", "rb").read():
                continue
            for line in open(d + "/maps"):
                a, b = (int(x, 16) for x in line.split()[0].split("-"))
                # NumPy advises huge pages from the first page boundary, so the array's region is
                # split; take the anonymous region of about the buffer's size.
                if len(line.split()) == 5 and 0.97 * SIZE < b - a <= SIZE:
                    out[int(d[6:])] = (a, b)
        except OSError:
            pass
    return out

def clear(pids):
    for p in pids:
        with open(f"/proc/{p}/clear_refs", "w") as f:
            f.write("4")

def dirty(p, span):
    start, end = span
    n = (end - start) // PAGE
    with open(f"/proc/{p}/pagemap", "rb") as f:
        f.seek(start // PAGE * 8)
        data = f.read(n * 8)
    return [i for i, (e,) in enumerate(struct.iter_unpack("<Q", data)) if e >> 55 & 1]

bufs = buffers()
print(f"{len(bufs)} buffers found", flush=True)
clear(bufs)
t0 = time.time()
for r in range(rounds):
    time.sleep(wait)
    t = time.time()
    for p, a in sorted(bufs.items()):
        pages = dirty(p, a)
        if not pages:
            print(p, round(t - t0), "none"); continue
        # Steps written: the dirty pages form one run (or two when the write position wrapped).
        lo, hi = pages[0], pages[-1]
        gaps = [(x, y) for x, y in zip(pages, pages[1:]) if y - x > 1]
        if gaps:  # wrapped: end is the page before the biggest gap
            hi = max(gaps, key=lambda g: g[1] - g[0])[0]
        print(p, round(t - t0), len(pages), "pages", f"pos~{hi * PAGE // STEP}",
              f"rate~{len(pages) * PAGE / STEP / (t - t0):.2f} steps/s", flush=True)
