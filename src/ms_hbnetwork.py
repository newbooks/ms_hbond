#!/usr/bin/env python

import sys
import argparse
import numpy as np
from numba import njit
import logging
from collections import defaultdict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute hydrogen bond network statistics from mcce.")
    parser.add_argument("-i", choices=["matrix", "adj", "numba"], default="numba", help="Implementation method for hydrogen bond network computation")
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


def donor_acceptor_list(fname="step2_out_hah.txt"):
    da = []
    with open(fname, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.split()
            da.append((parts[0], parts[1]))  # (donor_confid, acceptor_confid)

    return da


class MatrixImplementation:
    def __init__(self):
        self.confids, self.confid_to_index = read_head3_lst("head3.lst")
        self.da_list = donor_acceptor_list("step2_out_hah.txt")
        self.hb_matrix = self._initialize_hb_matrix()
        self.hb_count = np.zeros((len(self.confids), len(self.confids)), dtype=int)  # n_confs x n_confs, donor on rows, acceptor on columns
        self.total_ms_count = 0

    def _initialize_hb_matrix(self):
        # Initialize hydrogen bond lookup matrix, hb_matrix
        logging.info("Initializing hydrogen bond lookup matrix.")
        n_confs = len(self.confids)
        hb_matrix = np.zeros((n_confs, n_confs), dtype=bool)
        for donor_confid, acceptor_confid in self.da_list:
            try:
                donor_index = self.confid_to_index[donor_confid]
                acceptor_index = self.confid_to_index[acceptor_confid]
                hb_matrix[donor_index, acceptor_index] = True
            except KeyError:
                pass
        return hb_matrix

    def process_microstate(self, microstate, count):
        # Compute hydrogen bond network for the given microstate using hb_matrix
        reduced_matrix = self.hb_matrix[np.ix_(microstate, microstate)]
        d, a = np.nonzero(reduced_matrix)
        np.add.at(
            self.hb_count,
            (microstate[d], microstate[a]),
            count
        )
        self.total_ms_count += count

    def dump_hb_count(self, fname="hbnetwork_count.txt"):
        # Dump the hydrogen bond count matrix to a text file
        donor_idxs, acceptor_idxs = np.nonzero(self.hb_count)
        with open(fname, 'w') as f:
            f.write(f"# Matrix Implementation - Total microstate count: {self.total_ms_count}\n")
            f.write("Donor_ConfID  Acceptor_ConfID  Count\n")
            for d_idx, a_idx in zip(donor_idxs, acceptor_idxs):
                count = self.hb_count[d_idx, a_idx]
                donor_confid = self.confids[d_idx]
                acceptor_confid = self.confids[a_idx]
                f.write(f"{donor_confid}  {acceptor_confid}  {count}\n")
        logging.info(f"Dumped hydrogen bond count matrix to {fname}.")
        

class AdjImplementation:
    def __init__(self):
        self.confids, self.confid_to_index = read_head3_lst("head3.lst")
        self.da_list = donor_acceptor_list("step2_out_hah.txt")
        self.hb_adj = self._initialize_hb_adjacency()
        self.hb_count = defaultdict(int)  # (donor_idx, acceptor_idx) -> count
        self.total_ms_count = 0

    def _initialize_hb_adjacency(self):
        hb_adj = defaultdict(set)  # donor_idx -> set of acceptor_idx
        for donor_confid, acceptor_confid in self.da_list:
            try:
                d = self.confid_to_index[donor_confid]
                a = self.confid_to_index[acceptor_confid]
                hb_adj[d].add(a)
            except KeyError:
                pass
        return hb_adj
    
    def process_microstate(self, microstate, count):
        microstate_set = set(microstate)
        for d in microstate:
            for a in self.hb_adj.get(d, []):
                if a in microstate_set:
                    self.hb_count[(d, a)] += count
        self.total_ms_count += count

    def dump_hb_count(self, fname="hbnetwork_count.txt"):
        with open(fname, 'w') as f:
            f.write(f"# Adjacency List Implementation - Total microstate count: {self.total_ms_count}\n")
            f.write("Donor_ConfID  Acceptor_ConfID  Count\n")
            for (d_idx, a_idx), count in self.hb_count.items():
                donor_confid = self.confids[d_idx]
                acceptor_confid = self.confids[a_idx]
                f.write(f"{donor_confid}  {acceptor_confid}  {count}\n")
        logging.info(f"Dumped hydrogen bond count adjacency list to {fname}.")


@njit
def _process_microstate_numba(
    microstate,
    count,
    hb_adj_indices,
    hb_adj_indptr,
    ms_mask,
    hb_count,
):
    for i in range(len(microstate)):
        d = microstate[i]
        start = hb_adj_indptr[d]
        end = hb_adj_indptr[d + 1]

        for p in range(start, end):
            a = hb_adj_indices[p]
            if ms_mask[a]:
                hb_count[d, a] += count


class AdjImplementationNumba:
    def __init__(self):
        self.confids, self.confid_to_index = read_head3_lst("head3.lst")
        self.da_list = donor_acceptor_list("step2_out_hah.txt")

        self.n_confs = len(self.confids)

        # Build adjacency in CSR-like format
        self.hb_adj_indices, self.hb_adj_indptr = self._build_csr_adjacency()

        # Dense count matrix (Numba-friendly)
        self.hb_count = np.zeros((self.n_confs, self.n_confs), dtype=np.int64)

        self.total_ms_count = 0

    def _build_csr_adjacency(self):
        """
        Convert donor->acceptor adjacency list into CSR-like arrays
        """
        tmp_adj = defaultdict(set)

        for donor_confid, acceptor_confid in self.da_list:
            try:
                d = self.confid_to_index[donor_confid]
                a = self.confid_to_index[acceptor_confid]
                tmp_adj[d].add(a)
            except KeyError:
                # Some donor/acceptor conformer IDs from the DA list are not present
                # in head3.lst; skip those pairs when building the adjacency.
                pass

        indptr = np.zeros(self.n_confs + 1, dtype=np.int32)
        indices = []

        for d in range(self.n_confs):
            neighbors = tmp_adj.get(d, ())
            indptr[d + 1] = indptr[d] + len(neighbors)
            indices.extend(sorted(neighbors))

        return np.array(indices, dtype=np.int32), indptr

    def process_microstate(self, microstate, count):
        """
        Processes a single microstate and updates hydrogen bond counts.
        """
        microstate = np.asarray(microstate, dtype=np.int32)

        # Boolean membership mask
        ms_mask = np.zeros(self.n_confs, dtype=np.uint8)
        ms_mask[microstate] = 1

        _process_microstate_numba(
            microstate,
            count,
            self.hb_adj_indices,
            self.hb_adj_indptr,
            ms_mask,
            self.hb_count,
        )

        self.total_ms_count += count

    def dump_hb_count(self, fname="hbnetwork_count.txt"):
        """
        Write the hydrogen bond count matrix to a text file in a human-readable format.

        The output file begins with a header line containing the total microstate
        count, followed by tabular data with donor and acceptor conformer IDs
        and their corresponding hydrogen bond counts.
        """
        with open(fname, "w") as f:
            f.write(
                f"# Adjacency+Numba Implementation - Total microstate count: "
                f"{self.total_ms_count}\n"
            )
            f.write("Donor_ConfID  Acceptor_ConfID  Count\n")

            donor_idxs, acceptor_idxs = np.nonzero(self.hb_count)
            for d, a in zip(donor_idxs, acceptor_idxs):
                f.write(
                    f"{self.confids[d]}  "
                    f"{self.confids[a]}  "
                    f"{self.hb_count[d, a]}\n"
                )

        logging.info(f"Dumped hydrogen bond count adjacency list (Numba) to {fname}.")


def compute_hbnetwork(input_file, implementation):
    # Read microstate data from input_file
    logging.info(f"Reading microstate data from {input_file}, processing the microstate header.")
    with open(input_file, 'r') as f:
        # First line, temperature, pH and Eh
        line = next(f)
        parts = line.strip().split(",")
        temperature = float(parts[0].split(":")[1])
        pH = float(parts[1].split(":")[1])
        Eh = float(parts[2].split(":")[1])
        logging.info(f"Temperature: {temperature}, pH: {pH}, Eh: {Eh}")
        
        # Skip two lines: Method, header 
        for _ in range(2): next(f)        
        
        # Fixed conformers line, convert to a list that will be attached to the end of microstate
        line = next(f)
        parts = line.strip().split(":")
        n_fixed_confs = int(parts[0]) 
        fixed_confs = np.array([int(conf) for conf in parts[1].split()])  # Ignore "Fixed_conformers:" label
        logging.info(f"Detected {len(fixed_confs)} fixed occupied conformers, expected {n_fixed_confs}.")
        if len(fixed_confs) != n_fixed_confs:
            logging.warning("Number of fixed conformers does not match the expected count.")
            exit(1)

        # Header, skip
        next(f)
        
        # Free conformers line, conformers are grouped by free residues
        line = next(f)
        parts = line.strip().split(":")
        n_free_residues = int(parts[0])
        free_residue_confs_str = parts[1].split(";")  # Ignore "Free_conformers:" label
        free_residue_confs = [
            [int(conf) for conf in residue_str.strip().split()]
            for residue_str in free_residue_confs_str
            if residue_str.strip().split()
        ]
        logging.info(
            f"Detected {len(free_residue_confs)} free residues with conformers, expected {n_free_residues}."
        )
        if len(free_residue_confs) != n_free_residues:
            logging.warning("Number of free residues does not match the expected count.")
            sys.exit(1)
        
        # Compose a reverse lookup: conformer index -> microstate index (free residue list)
        free_confs = np.array([int(conf) for residue_confs in free_residue_confs for conf in residue_confs])
        max_conf = max(free_confs)
        iconf_to_microstate_index = [-1] * (max_conf + 1)
        for microstate_index, residue_confs in enumerate(free_residue_confs):
            for conf in residue_confs:
                iconf = int(conf)
                iconf_to_microstate_index[iconf] = microstate_index

        #------------------------------------------------------------------------------------------------
        # Congratulations, we have parsed the microstate header and have all the data structures we need
        #------------------------------------------------------------------------------------------------
        # Now process each microstate line
        Continue_reading = True
        while Continue_reading:
            # Look for pattern "MC:0", "MC:1", etc.
            iter_flag = False
            line = next(f, None)
            if line is None:
                break
            if not line.startswith("MC:"):
                continue
            else:
                MC_mark = line.strip()
                iter_flag = True
            
            while iter_flag and Continue_reading:
                # Initial MC mark, count, and ms
                logging.info(f"Processing microstate group {MC_mark}.")                                
                # Once detected the MC: header, read the microstate in the next line
                microstate_line = next(f)
                ms_ini = np.array(microstate_line.split(":")[1].split()) 
                # Collect the count from the next line
                count_line = next(f)
                this_ms_count = int(count_line.split(",")[1].strip()) 
                this_ms = np.array(ms_ini, dtype=int)
                this_ms = np.concatenate((this_ms, fixed_confs))  # Update this ms to append fixed conformers
                implementation.process_microstate(this_ms, this_ms_count)
                line_counter = 1

                # Go the the next microstate till EOF or next MC. Empty lines are ignored
                while True:
                    try:
                        line = next(f)
                    except StopIteration:
                        Continue_reading = False
                        break
                    if line.startswith("MC:"):
                        # Found the next microstate header, break to outer loop
                        new_MC_mark = line.strip()
                        break
                    if line.strip() == "":
                        continue  # Ignore empty lines

                    parts = line.strip().split(",")
                    this_ms_count = int(parts[1])
                    flipped_confs = [int(a) for a in parts[2].split()]
                    for iconf in flipped_confs:
                        this_ms[iconf_to_microstate_index[iconf]] = iconf   # this_ms is modified in place
                    implementation.process_microstate(this_ms, this_ms_count)
                    line_counter += 1

                logging.info(f"Processed {line_counter} lines for microstate group {MC_mark}.")
                MC_mark = new_MC_mark if Continue_reading else None


    # After processing all microstates, output the hydrogen bond network statistics
    logging.info("Finished processing all microstates. Outputting hydrogen bond network statistics.")
    implementation.dump_hb_count("hbnetwork_count.txt")



if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.file
    method = args.i

    if method == "matrix":
        logging.info("Using matrix method for hydrogen bond network computation.")
        implementation = MatrixImplementation()
    elif method == "adj":
        implementation = AdjImplementation()
    elif method == "numba":
        implementation = AdjImplementationNumba()
    else:
        raise ValueError(f"Unknown method: {method}")

    compute_hbnetwork(input_file, implementation)