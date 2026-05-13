import os
import sys
import subprocess
import time
from datetime import datetime

log_dir = "logs/experiment"
os.makedirs(log_dir, exist_ok=True)

runs = [
    (2, 3, 2, 4),
    (2, 3, 2, 5),
    (2, 3, 2, 6),
    (2, 1, 2, 4),
    (2, 2, 2, 4),
    (2, 3, 1, 6),
    (2, 3, 1, 5)
]

MAX_PARALLEL = 1
running = []

def start_run(i, a, b, c):
    log_file = os.path.join(log_dir, f"run_{i}_{a}_{b}_{c}_left.log")

    cmd = [
        sys.executable,
        "run_left.py",
        str(i),
        str(a),
        str(b),
        str(c)
    ]

    f = open(log_file, "w", encoding="utf-8")
    f.write(f"Started: {datetime.now()}\n")
    f.write(f"Command: {' '.join(cmd)}\n")
    f.write("=" * 80 + "\n\n")
    f.flush()

    print(f"Started ( {i}, {a}, {b}, {c})")

    p = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT
    )

    return p, f, i, a, b, c


pending = runs.copy()

while pending or running:

    while pending and len(running) < MAX_PARALLEL:
        i, a, b, c = pending.pop(0)
        running.append(start_run(i, a, b, c))

    still_running = []

    for p, f, i, a, b, c in running:
        if p.poll() is None:
            still_running.append((p, f, i, a, b, c))
        else:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"Finished: {datetime.now()}\n")
            f.write(f"Exit code: {p.returncode}\n")
            f.close()

            print(f"Finished ({i}, {a}, {b}, {c})")

    running = still_running

    time.sleep(2)