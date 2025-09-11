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
Author: Qiu-Shi Huang 
Ref：
[1]https://www.pnas.org/doi/10.1073/pnas.2318341121
[2]https://link.aps.org/doi/10.1103/PhysRevLett.133.226101
