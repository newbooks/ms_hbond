#!/usr/bin/env python

import sys
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


def donor_acceptor_list(fname="step2_out_hah.txt"):
    da = []
    with open(fname, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.split()
            da.append((parts[0], parts[1]))  # (donor_confid, acceptor_confid)

    return da


class Matrix_Implementation:
    def __init__(self):
        self.confids, self.confid_to_index = read_head3_lst("head3.lst")
        self.da_list = donor_acceptor_list("step2_out_hah.txt")
        self.hb_matrix = self._initialize_hb_matrix()

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


class Adj_Implementation:
    def __init__(self):
        pass
    # Placeholder for adjacency list implementation


class AdjNumba_Implementation:
    def __init__(self):
        pass
    # Placeholder for numba-optimized adjacency list implementation


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
        fixed_confs = np.array(parts[1].split())  # Ignore "Fixed_conformers:" label
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
            residue_str.strip().split()
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
        free_confs = [int(conf) for residue_confs in free_residue_confs for conf in residue_confs]
        max_conf = max(free_confs)
        iconf_to_microstate_index = [-1] * (max_conf + 1)
        for microstate_index, residue_confs in enumerate(free_residue_confs):
            for conf in residue_confs:
                iconf = int(conf)
                iconf_to_microstate_index[iconf] = microstate_index

        #------------------------------------------------------------------------------------------------
        # Congratulations, we have parsed the microstate header and have all the data structures we need
        #------------------------------------------------------------------------------------------------
        logging.info("Finished parsing microstate header.")


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
                line_counter = 1
                # Compute hydrogen bond network for this microstate using hb_matrix
                # Placeholder for actual hydrogen bond network computation

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
                        this_ms[iconf_to_microstate_index[iconf]] = iconf
                    line_counter += 1

                logging.info(f"Processed {line_counter} lines for microstate group {MC_mark}.")
                MC_mark = new_MC_mark if Continue_reading else None





if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.file
    method = args.i

    if method == "matrix":
        logging.info("Using matrix method for hydrogen bond network computation.")
        implementation = Matrix_Implementation()
    elif method == "adj":
        implementation = Adj_Implementation()
    elif method == "adjnumba":
        implementation = AdjNumba_Implementation()
    else:
        raise ValueError(f"Unknown method: {method}")

    compute_hbnetwork(input_file, implementation)