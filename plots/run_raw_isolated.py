#!/usr/bin/env python3

import os
import subprocess
import glob
import re

# List of benchmark directories
BENCHMARK_DIRS = [
    os.path.expanduser("~/osu-micro-ucc"),
    os.path.expanduser("~/osu-micro-nccl"),
]

NUM_TRIALS = 3
DATA_DIR = "data"

def next_output_file():
    # ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    # find existing files matching run_N.txt
    pattern = re.compile(r'run_(\d+)\.txt$')
    existing = glob.glob(os.path.join(DATA_DIR, 'run_*.txt'))
    nums = []
    for f in existing:
        m = pattern.search(os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    next_n = max(nums) + 1 if nums else 1
    return os.path.join(DATA_DIR, f"run_{next_n}.txt")

def rotate_files():
    # keep only the 10 newest files
    files = glob.glob(os.path.join(DATA_DIR, 'run_*.txt'))
    if len(files) <= 10:
        return
    files.sort(key=lambda f: os.path.getmtime(f))
    for old in files[:-10]:
        try:
            os.remove(old)
        except OSError:
            pass

def main():
    output_file = next_output_file()
    with open(output_file, "w") as outf:
        for bench_dir in BENCHMARK_DIRS:
            name = os.path.basename(bench_dir)
            if not os.path.isdir(bench_dir):
                print(f"Warning: directory {bench_dir} not found, skipping.")
                continue

            for trial in range(1, NUM_TRIALS + 1):
                print(f"[{name}] Running trial {trial}/{NUM_TRIALS}...")
                outf.write(f"=== {name} Trial {trial} ===\n")
                proc = subprocess.run(
                    "./run.sh",
                    shell=True,
                    cwd=bench_dir,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                outf.write(proc.stdout)
                outf.write("\n")

    rotate_files()
    print(f"\n✓ All done! Raw outputs written to '{output_file}'")

if __name__ == "__main__":
    main()

