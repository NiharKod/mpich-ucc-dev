#!/usr/bin/env python3

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# Directories
CLEAN_DIR = "data-cleaned"
GRAPH_DIR = "graphs"

# Filename patterns
CSV_PATTERN = re.compile(r'run_(\d+)\.csv$')
IMAGE_PATTERN = re.compile(r'run_(\d+)\.png$')

# Ensure output directory exists
os.makedirs(GRAPH_DIR, exist_ok=True)

def get_latest_clean_file():
    files = glob.glob(os.path.join(CLEAN_DIR, 'run_*.csv'))
    if not files:
        raise FileNotFoundError(f"No cleaned CSVs found in {CLEAN_DIR}")
    files.sort(key=lambda f: int(CSV_PATTERN.search(os.path.basename(f)).group(1)))
    return files[-1]

def rotate_graphs():
    files = glob.glob(os.path.join(GRAPH_DIR, 'run_*.png'))
    if len(files) <= 10:
        return
    files.sort(key=lambda f: int(IMAGE_PATTERN.search(os.path.basename(f)).group(1)))
    for old in files[:-10]:
        try:
            os.remove(old)
        except OSError:
            pass

def main():
    # Locate latest cleaned CSV
    latest_csv = get_latest_clean_file()
    run_number = CSV_PATTERN.search(os.path.basename(latest_csv)).group(1)
    print(f"Generating graph for run {run_number} from {latest_csv}...")

    # Load and average across trials
    df = pd.read_csv(latest_csv)
    avg_df = df.groupby(['benchmark', 'size'], as_index=False)['latency_us'].mean()

    # Pivot for plotting
    pivot = avg_df.pivot(index='size', columns='benchmark', values='latency_us')
    pivot = pivot.sort_index()

    # Plot
    plt.figure()
    for bench in pivot.columns:
        plt.plot(pivot.index, pivot[bench], marker='o', linestyle='-', label=bench)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Message size (bytes)')
    plt.ylabel('Avg Latency (μs)')
    plt.title(f'OSU Allreduce Latency Run {run_number}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save graph
    output_path = os.path.join(GRAPH_DIR, f'run_{run_number}.png')
    plt.savefig(output_path)
    plt.close()
    rotate_graphs()
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    main()

