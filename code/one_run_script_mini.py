import os
import sys
import subprocess
from datetime import datetime


# takes model index +1 as command line argument
i = sys.argv[1] if len(sys.argv) > 1 else exit(1)

log_dir = "logs/experiment/"
os.makedirs(log_dir, exist_ok=True)

for a in range(1, 4):      # 1–3,     word type
    for b in range(1, 3):  # 1–2      word count
        #for c in range(1, 8):  # 1–7  perturbation type
        if (a,b) in ((1,1), (1,2), (2,1)):
            continue

        log_file = os.path.join(log_dir, f"run_{i}_{a}_{b}_6.log")

        cmd = [sys.executable, "ollama_async_skip_timeout.py", str(i), str(a), str(b), str(6)]

        print(f"Running: {' '.join(cmd)}")
        print(f"Log: {log_file}")

        with open(log_file, "w") as f:
            f.write(f"Started at: {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("-" * 50 + "\n")

            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT
            )

            process.wait()  # ensures sequential execution

            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Finished at: {datetime.now()}\n")