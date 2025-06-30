#!/usr/bin/env python3

import os
import subprocess
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_output(output):
    data = {}
    pattern = re.compile(r'^\s*(\d+)\s+([\d\.]+)')
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = pattern.match(line)
        if m:
            size = int(m.group(1))
            latency = float(m.group(2))
            data[size] = latency
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Run predefined OSU benchmarks and plot average latency comparisons on one graph.')
    parser.add_argument('trials', type=int, help='Number of trials to average')
    args = parser.parse_args()

    # Hard-coded benchmark directories
    directories = [
        '/home/nkodkani/osu-micro-ucc',
        '/home/nkodkani/osu-micro-nccl',
        '/home/nkodkani/osu-micro-native',
        '/home/nkodkani/osu-micro-native-nodevice'
    ]

    cwd = os.getcwd()
    results = {}

    for directory in directories:
        build_name = os.path.basename(os.path.abspath(directory))
        os.chdir(directory)
        all_runs = []
        cmd = './run.sh'
        for i in range(args.trials):
            print(f'[{build_name}] Running trial {i+1}/{args.trials}...')
            proc = subprocess.run(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if proc.returncode != 0:
                print(f'[{build_name}] Trial {i+1} failed:\n{proc.stderr}')
                os.chdir(cwd)
                return
            all_runs.append(parse_output(proc.stdout))
        os.chdir(cwd)
        # Compute average latency
        df = pd.DataFrame(all_runs).T
        df['avg_latency'] = df.mean(axis=1)
        results[build_name] = df['avg_latency']

    # Combine into a single DataFrame
    comp_df = pd.DataFrame(results).sort_index()

    # Plot
    plt.figure()
    for col in comp_df.columns:
        plt.plot(comp_df.index, comp_df[col], marker='o', linestyle='-', label=col)
    plt.xscale('log')
    plt.xlabel('Message size (bytes)')
    plt.ylabel('Avg Latency (us)')
    plt.title('OSU Allreduce Latency Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = 'latency_comparison.png'
    plt.savefig(output_file)
    print(f'Comparison plot saved to {output_file}')

if __name__ == '__main__':
    main()
