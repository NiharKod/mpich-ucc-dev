#!/usr/bin/env python3
"""
Simple script to run each benchmark's run.sh and append all raw output into a single file.

Usage:
  python run_all_benchmarks.py --dirs ~/osu-micro-nccl ~/osu-micro-ucc --output all_output.txt
"""
import argparse
import subprocess
import os
import sys

def run_and_append(dir_path, outfile):
    script_path = os.path.join(os.path.expanduser(dir_path), 'run.sh')
    if not os.path.isfile(script_path):
        print(f"[WARN] {script_path} not found, skipping.", file=sys.stderr)
        return
    print(f"[INFO] Running {script_path}...")
    try:
        result = subprocess.run([script_path], cwd=os.path.dirname(script_path),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_path} failed with error:\n{e.stderr}", file=sys.stderr)
        return

    # Write header and output
    outfile.write(f"===== OUTPUT for {dir_path} =====\n")
    outfile.write(result.stdout)
    outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks and append raw outputs into one file.")
    parser.add_argument('--dirs', nargs='+', required=True, help="Benchmark directories containing run.sh")
    parser.add_argument('--output', required=True, help="Path to the output file to append all logs")
    args = parser.parse_args()

    # Open output file once
    try:
        with open(os.path.expanduser(args.output), 'w') as out:
            for d in args.dirs:
                run_and_append(d, out)
    except OSError as e:
        print(f"[ERROR] Cannot write to {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] All outputs written to {args.output}")

if __name__ == '__main__':
    main()

