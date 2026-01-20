#!/usr/bin/env python
import sys
import argparse
import pandas as pd
import re

"""
Compare microstate H-bond occupancy results among multiple files.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microstate H-bond Occupancy Comparison")
    parser.add_argument("files", nargs="+", type=str, help="Paths to H bond count files to compare")
    args = parser.parse_args()

    data_list = []
    for file in args.files:
        with open(file) as f:
            first_line = f.readline()
        match = re.search(r"Total microstate count:\s*(\d+)", first_line)
        if not match:
            print(f"Error: Could not parse total microstate count from first line of file '{file}'.")
            sys.exit(1)
        total_microstates = int(match.group(1))
        df = pd.read_csv(file, sep=r"\s+", comment="#")  # ignore comment lines
        donor_acceptor_pairs = df[["Donor_ConfID", "Acceptor_ConfID"]].values
        counts = df["Count"].values
        fractions = counts / total_microstates
 
        data_list.append((file, dict(zip(map(tuple, donor_acceptor_pairs), fractions))))

    # Compare the data
    all_pairs = set()
    for _, data in data_list:
        all_pairs.update(data.keys())

    print(" ".join([f"{'Donor_ConfID':<16}", f"{'Acceptor_ConfID':<16} "] + [f"{re.sub(r'\.[^.]+$', '', file):<16}" for file, _ in data_list] + ["std_dev"]))
    for pair in all_pairs:
        row = [str(pair[0]), str(pair[1])]
        occs = []
        for _, data in data_list:
            occ = data.get(pair, 0.0)
            occs.append(occ)
            row.append(f"{occ:10.6f}")
        std_dev = pd.Series(occs).std()
        row.append(f"{std_dev:12.6f}")
        print("    ".join(row))