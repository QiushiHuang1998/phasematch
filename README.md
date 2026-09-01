Genetic search for supercell matching between two crystals with:
- HNF parametrization (lower-triangular, positive diag; 0<=b<d, 0<=c,e<f)
- Composition alignment via formula-units base ratio (d1_base, d2_base)
- Determinant sampling: det(H1)=t*d1_base, det(H2)=t*d2_base with small t
- Per-species Hungarian matching (periodic min-image)
- Gram (metric) penalty for lattice shape similarity
- Fast supercell replication (corrected: f' = (f @ H^{-1} + n @ H^{-1}) mod 1)
- Geometry-based deduplication (Gram-key), strong memoization
- Elitist GA with tournament selection + crossover + mutation
- Robust initialization with fallback diagonal enumeration (no empty population)
- Export top-K POSCARs + print & save best H1/H2 (txt/npy/json)

Version 5 keeps the original script's fast HNF + GA idea, but fixes the
main correctness issues:
  - command line inputs instead of hard-coded POSCAR names only;
  - row-HNF-aware crossover/mutation, so children remain valid HNF matrices;
  - Angstrom-based periodic shuffle distances instead of raw fractional RMSD;
  - origin-shift search before Hungarian assignment;
  - common unimodular Q reduction for equivalent, less skewed supercells;
  - neighbor-image Cartesian distance checks for skewed cells;
  - optional spglib-based structure deduplication;
  - exported POSCAR pairs are lattice-normalized and reordered by the actual atom assignment;
  - optional bounded exhaustive mode for small determinant ranges;
  - per-candidate metrics are saved for reproducibility.
  - raw HNF SNF tags are exported together with an exact intrinsic-period audit.

It is still a heuristic matcher in GA mode. Use --mode enumerate when the
bounded search space is small enough and completeness inside those bounds is
required.

Acknowledgement: The CSM enumeration/lattice-normalization design benefited
from studying the crystmatch program and related work by Fangcheng Wang,
Xin-Zheng Li, and co-workers.

Author: Qiu-Shi Huang 

Ref：

[1]https://www.pnas.org/doi/10.1073/pnas.2318341121

[2]https://link.aps.org/doi/10.1103/PhysRevLett.133.226101
