#!/usr/bin/env python3

import os
import glob
import re
import csv

# Directories
RAW_DIR = "data"
CLEAN_DIR = "data-cleaned"

# Patterns
RAW_PATTERN = re.compile(r'run_(\d+)\.txt$')
HEADER_PATTERN = re.compile(r'^===\s+(\S+)\s+Trial\s+(\d+)\s+===')
DATA_PATTERN = re.compile(r'^\s*(\d+)\s+([\d.]+)')

# Ensure clean directory exists
os.makedirs(CLEAN_DIR, exist_ok=True)

def get_latest_raw_file():
    files = glob.glob(os.path.join(RAW_DIR, 'run_*.txt'))
    if not files:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}")
    # Sort by extracted run number
    files.sort(key=lambda f: int(RAW_PATTERN.search(os.path.basename(f)).group(1)))
    return files[-1]

def parse_raw_file(filepath):
    rows = []
    current_bench = None
    current_trial = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Check for header
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                current_bench = header_match.group(1)
                current_trial = int(header_match.group(2))
                continue
            # Skip comments and errors
            if line.startswith('#') or line.startswith('['):
                continue
            # Match size-latency
            data_match = DATA_PATTERN.match(line)
            if data_match and current_bench and current_trial is not None:
                size = int(data_match.group(1))
                latency = float(data_match.group(2))
                rows.append({
                    'benchmark': current_bench,
                    'trial': current_trial,
                    'size': size,
                    'latency_us': latency
                })
    return rows

def write_clean_csv(rows, run_number):
    output_file = os.path.join(CLEAN_DIR, f"run_{run_number}.csv")
    with open(output_file, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=['benchmark', 'trial', 'size', 'latency_us'])
        writer.writeheader()
        writer.writerows(rows)
    return output_file

def rotate_cleaned():
    files = glob.glob(os.path.join(CLEAN_DIR, 'run_*.csv'))
    if len(files) <= 10:
        return
    # Sort by run number
    files.sort(key=lambda f: int(RAW_PATTERN.search(os.path.basename(f)).group(1)))
    # Remove oldest beyond 10
    for old in files[:-10]:
        try:
            os.remove(old)
        except OSError:
            pass

def main():
    raw_file = get_latest_raw_file()
    run_number = RAW_PATTERN.search(os.path.basename(raw_file)).group(1)
    print(f"Parsing raw data from {raw_file} (run {run_number})...")
    rows = parse_raw_file(raw_file)
    clean_file = write_clean_csv(rows, run_number)
    rotate_cleaned()
    print(f"Cleaned CSV written to {clean_file}")

if __name__ == "__main__":
    main()

