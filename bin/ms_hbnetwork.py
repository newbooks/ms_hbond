#!/usr/bin/env python

import argparse
import numpy as np
import numba as nb
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute hydrogen bond network statistics from mcce.")
    parser.add_argument("-i", choices=["matrix", "adj", "adjnumba"], default="matrix", help="Implementation method for hydrogen bond network computation")
    parser.add_argument("file", help="Input file name for microstates")
    return parser.parse_args()


def read_head3_lst(filename="head3.lst"):
    confids = []
    confid_to_index = {}
    with open(filename, 'r') as f:
        next(f)  # Skip header line
        for index, line in enumerate(f):
            parts = line.split()
            confid = parts[1]
            confids.append(confid)
            confid_to_index[confid] = index

    return confids, confid_to_index


def dornor_acceptor_list(fname="step2_out_hah.txt"):
    da = []
    with open(fname, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.split()
            da.append((parts[0], parts[1]))  # (donor_confid, acceptor_confid)

    return da


def compute_hbnetwork_matrix(input_file):
    # Read head3.lst to get confids and mapping
    fname = "head3.lst"
    logging.info(f"Reading {fname} for conformer IDs.")
    confids, confid_to_index = read_head3_lst(fname)

    # Read hydrogen bond matrix from step2_out_hah.txt
    fname = "step2_out_hah.txt"
    logging.info(f"Reading hydrogen bond donor-acceptor pairs from {fname}.")
    da_list = dornor_acceptor_list(fname)

    # Initialize hydrogen bond lookup matrix, hb_matrix
    logging.info("Initializing hydrogen bond lookup matrix.")
    n_confs = len(confids)
    hb_matrix = np.zeros((n_confs, n_confs), dtype=bool)
    for donor_confid, acceptor_confid in da_list:
        if donor_confid in confid_to_index and acceptor_confid in confid_to_index:   # This filters BK confids
            donor_index = confid_to_index[donor_confid]
            acceptor_index = confid_to_index[acceptor_confid]
            hb_matrix[donor_index, acceptor_index] = True



    


def compute_hbnetwork_adj(input_file):
    confids, confid_to_index = read_head3_lst("head3.lst")
    n_confs = len(confids)
    hb_adj_list = {confid: [] for confid in confids}

    # Placeholder for actual hydrogen bond computation logic
    # Fill hb_adj_list based on hydrogen bond interactions

    print("Hydrogen Bond Network Adjacency List:")
    for confid, neighbors in hb_adj_list.items():
        print(f"{confid}: {neighbors}")


def compute_hbnetwork_adj_numba(input_file):
    confids, confid_to_index = read_head3_lst("head3.lst")
    n_confs = len(confids)
    hb_adj_list = {confid: [] for confid in confids}

    # Placeholder for actual hydrogen bond computation logic using Numba
    # Fill hb_adj_list based on hydrogen bond interactions

    print("Hydrogen Bond Network Adjacency List (Numba):")
    for confid, neighbors in hb_adj_list.items():
        print(f"{confid}: {neighbors}")


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.file
    method = args.i

    if method == "matrix":
        logging.info("Using matrix method for hydrogen bond network computation.")
        compute_hbnetwork_matrix(input_file)
    elif method == "adj":
        compute_hbnetwork_adj(input_file)
    elif method == "adjnumba":
        compute_hbnetwork_adj_numba(input_file)
    else:
        raise ValueError(f"Unknown method: {method}")
