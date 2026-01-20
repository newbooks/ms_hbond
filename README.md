# ms_hbond

## Additional requirement
The fastest implementation uses Numba module. Please install numba by
```
pip install numba
```  

## Goal
Hydrogen bond detection in microstates is a critical tool in MCCE_tools. Given the current performance constraints, this project aims to explore different implementations of microstate hydrogen bond detection and benchmark the speed of various algorithms.

Implementations under evaluation:

- **Matrix approach**: Hydrogen bond networks are stored as a lookup matrix. For each microstate, the matrix is reduced and NumPy vectorized operations are used to identify hydrogen bonds. The counter is also matrix-based.
- **Adjacency list**: Hydrogen bond networks are stored as an adjacency list. Since hydrogen bonds among conformers are sparse, this approach reduces unnecessary operations and saves computation time. The counter is a Python dictionary.
- **Adjacency list with Numba**: Adjacency list operations involve loops and lose the advantage of NumPy vectorization. Numba introduces C-level efficiency to recover performance. The counter is a dense matrix used in matrix implementation for the best performance.

## Hypothesis:

Adjacency list with Numba > Matrix implementation > Adjacency list

## Benchmark results

| Implementation | 4lzt | Jose 30m | Jose unfinished |
|---|---|---|---|
| Matrix | 51s | 9m | > 8h |
| Adjacency | 19s | 1m16s | 25m5s |
| Numba | 4s | 6s | 44s |

## Conclusion

**Performance:** Numba implementation significantly outperforms both adjacency list and matrix approaches across all datasets.

**Scalability:** Numba demonstrates superior scalability, followed by the adjacency list approach, with the matrix implementation showing the steepest performance degradation as dataset size increases.

