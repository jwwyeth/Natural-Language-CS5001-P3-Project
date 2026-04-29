import os
import sys
import subprocess
from datetime import datetime


i = sys.argv[1] if len(sys.argv) > 1 else exit(1)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

for a in range(1, 5):      # 1–4
    for b in range(1, 4):  # 1–3
        for c in range(1, 5):  # 1–4

            log_file = os.path.join(log_dir, f"run_{a}_{b}_{c}_{i}.log")

            cmd = [sys.executable, "ollama_cloud_async.py", str(a), str(b), str(c), str(i)]

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