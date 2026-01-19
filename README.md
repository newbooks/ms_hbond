# ms_hbond

## Goal
Hydrogen bond detection in microstates is a critical tool in MCCE_tools. Given the current performance constraints, this project aims to explore different implementations of microstate hydrogen bond detection and benchmark the speed of various algorithms.

Implementations under evaluation:

- **Matrix approach**: Hydrogen bond networks are stored as a lookup matrix. For each microstate, the matrix is reduced and NumPy vectorized operations are used to identify hydrogen bonds. The counter is also matrix-based.
- **Adjacency list**: Hydrogen bond networks are stored as an adjacency list. Since hydrogen bonds among conformers are sparse, this approach reduces unnecessary operations and saves computation time. The counter is a Python dictionary.
- **Adjacency list with Numba**: Adjacency list operations involve loops and lose the advantage of NumPy vectorization. Numba introduces C-level efficiency to recover performance. The counter is a Numba typed dictionary.

## Hypothesis:

Adjacency list with Numba > Matrix implementation > Adjacency list

## Benchmark results

| Implementation | 1/4 Size | 1/2 Size | Full Size |
| Matrix | | | |
| Adjacency | | | |
| Adjacency + Numba | | | |



