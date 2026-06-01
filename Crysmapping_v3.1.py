#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust supercell matching between two crystals.

This version keeps the original script's fast HNF + GA idea, but fixes the
main correctness issues:
  - command line inputs instead of hard-coded POSCAR names only;
  - row-HNF-aware crossover/mutation, so children remain valid HNF matrices;
  - Angstrom-based periodic shuffle distances instead of raw fractional RMSD;
  - origin-shift search before Hungarian assignment;
  - independent unimodular Q reduction for equivalent, less skewed supercells;
  - neighbor-image Cartesian distance checks for skewed cells;
  - optional spglib-based structure deduplication;
  - exported POSCAR pairs are reordered by the actual atom assignment;
  - optional bounded exhaustive mode for small determinant ranges;
  - per-candidate metrics are saved for reproducibility.

It is still a heuristic matcher in GA mode. Use --mode enumerate when the
bounded search space is small enough and completeness inside those bounds is
required.

Author: Qiu-Shi Huang 
Ref：
[1]https://www.pnas.org/doi/10.1073/pnas.2318341121
[2]https://link.aps.org/doi/10.1103/PhysRevLett.133.226101
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure
    PYMATGEN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    Lattice = None
    Structure = None
    PYMATGEN_IMPORT_ERROR = exc

try:
    import spglib
except Exception:  # pragma: no cover - optional runtime dependency
    spglib = None


HnfFlat = Tuple[int, ...]
Individual = Tuple[HnfFlat, HnfFlat]


@dataclass
class Config:
    file_a: Path
    file_b: Path
    output_dir: Path
    seed: Optional[int]
    mode: str
    pop_size: int
    init_keep: int
    generations: int
    substitute_number: int
    elite_keep: int
    top_k: int
    det_mult_min: int
    det_mult_max: int
    diag_max: int
    weight_geom: float
    weight_rmsd: float
    weight_shape: float
    max_origin_shifts: int
    reduce_cell: bool
    image_shell: int
    dedupe_mode: str
    symprec: float
    exhaustive_limit: int


@dataclass
class Problem:
    cfg: Config
    rng: random.Random
    struct_a: Structure
    struct_b: Structure
    lattice_a: np.ndarray
    lattice_b: np.ndarray
    labels_a: np.ndarray
    labels_b: np.ndarray
    frac_a: np.ndarray
    frac_b: np.ndarray
    species: Tuple[str, ...]
    z_a: int
    z_b: int
    d_a_base: int
    d_b_base: int
    fitness_cache: Dict[Individual, "EvalResult"]
    eq_cache: set


@dataclass
class MatchResult:
    rmsd: float
    rmsd2: float
    shift: np.ndarray
    order_a: np.ndarray
    order_b: np.ndarray
    distances: np.ndarray


@dataclass
class EvalResult:
    fitness: float
    rmsd: float
    rmsd2: float
    geom: float
    shape: float
    shift: Tuple[float, float, float]
    det_a: int
    det_b: int
    q_a: Tuple[int, ...]
    q_b: Tuple[int, ...]


def divisors(n: int) -> List[int]:
    n = int(n)
    if n <= 0:
        return []
    small: List[int] = []
    large: List[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i * i != n:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def gcd_array(values: Sequence[int]) -> int:
    vals = [int(x) for x in values if int(x) != 0]
    if not vals:
        return 1
    g = vals[0]
    for value in vals[1:]:
        g = math.gcd(g, value)
    return abs(g) or 1


def row_lattice_matrix(struct: Structure) -> np.ndarray:
    return np.array(struct.lattice.matrix, dtype=float)


def labels_and_frac(struct: Structure) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.array([str(sp) for sp in struct.species], dtype=object)
    frac = np.array(struct.frac_coords, dtype=float)
    frac -= np.floor(frac)
    return labels, frac


def species_counts(struct: Structure, species: Sequence[str]) -> np.ndarray:
    counts = struct.composition.get_el_amt_dict()
    return np.array([int(round(counts.get(s, 0))) for s in species], dtype=int)


def metric_tensor(lattice: np.ndarray) -> np.ndarray:
    return lattice @ lattice.T


def gram_penalty(lattice_a: np.ndarray, lattice_b: np.ndarray) -> float:
    """Dimensionless metric mismatch between two row-lattice matrices."""
    g_a = metric_tensor(lattice_a)
    g_b = metric_tensor(lattice_b)
    denom = max(1e-14, 0.5 * (np.sum(g_a * g_a) + np.sum(g_b * g_b)))
    return float(np.sum((g_a - g_b) ** 2) / denom)


def strain_penalty(lattice_a: np.ndarray, lattice_b: np.ndarray) -> float:
    """Basis-invariant strain penalty from the deformation singular values."""
    try:
        deformation = np.linalg.solve(lattice_a, lattice_b)
        singular_values = np.linalg.svd(deformation, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1e30
    return float(np.mean((singular_values - 1.0) ** 2))


def cell_shape_score(lattice: np.ndarray) -> float:
    """Smaller means shorter and more orthogonal basis vectors."""
    lengths = np.linalg.norm(lattice, axis=1)
    volume = abs(float(np.linalg.det(lattice)))
    if volume < 1e-14 or np.any(lengths < 1e-14):
        return 1e30
    orthogonality_defect = float(np.prod(lengths) / volume)
    spread = float(np.max(lengths) / max(np.min(lengths), 1e-14))
    return (orthogonality_defect - 1.0) ** 2 + 0.05 * (spread - 1.0) ** 2


def pair_shape_score(lattice_a: np.ndarray, lattice_b: np.ndarray) -> float:
    return cell_shape_score(lattice_a) + cell_shape_score(lattice_b)


def _gram_schmidt_rows(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = basis.shape[0]
    b_star = np.zeros_like(basis, dtype=float)
    mu = np.zeros((n, n), dtype=float)
    norm = np.zeros(n, dtype=float)
    for i in range(n):
        vec = basis[i].astype(float).copy()
        for j in range(i):
            if norm[j] <= 1e-30:
                continue
            mu[i, j] = float(np.dot(basis[i], b_star[j]) / norm[j])
            vec -= mu[i, j] * b_star[j]
        b_star[i] = vec
        norm[i] = float(np.dot(vec, vec))
    return b_star, mu, norm


def lll_reduction_q(lattice: np.ndarray, delta: float = 0.75, max_iter: int = 200) -> np.ndarray:
    """Return an integer unimodular Q such that Q @ lattice is LLL-like reduced."""
    basis = np.array(lattice, dtype=float).copy()
    q = np.eye(3, dtype=int)
    k = 1
    steps = 0
    while k < 3 and steps < max_iter:
        steps += 1
        _, mu, norm = _gram_schmidt_rows(basis)
        for j in range(k - 1, -1, -1):
            r = int(round(mu[k, j]))
            if r:
                basis[k] -= r * basis[j]
                q[k] -= r * q[j]
                _, mu, norm = _gram_schmidt_rows(basis)

        if norm[k] + 1e-14 >= (delta - mu[k, k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            basis[[k, k - 1]] = basis[[k - 1, k]]
            q[[k, k - 1]] = q[[k - 1, k]]
            k = max(k - 1, 1)

    if round(np.linalg.det(q)) < 0:
        q[0] *= -1
    return q


def _valid_unimodular(q: np.ndarray) -> bool:
    return q.shape == (3, 3) and abs(int(round(np.linalg.det(q)))) == 1


def pair_reduction_q(lattice_a: np.ndarray, lattice_b: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Choose a common integer basis transform Q for both matched supercells."""
    identity = np.eye(3, dtype=int)
    if not enabled:
        return identity

    references = [lattice_a, lattice_b, 0.5 * (lattice_a + lattice_b)]
    candidates = [identity]
    for ref in references:
        q = lll_reduction_q(ref)
        if _valid_unimodular(q):
            candidates.append(q)

    best_q = identity
    best_score = pair_shape_score(lattice_a, lattice_b)
    for q in candidates:
        score = pair_shape_score(q @ lattice_a, q @ lattice_b)
        if score + 1e-12 < best_score:
            best_score = score
            best_q = q

    # Small pair-aware hill climb with elementary row operations.
    improved = True
    while improved:
        improved = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                vec_i_a, vec_j_a = best_q[i] @ lattice_a, best_q[j] @ lattice_a
                vec_i_b, vec_j_b = best_q[i] @ lattice_b, best_q[j] @ lattice_b
                denom = np.dot(vec_j_a, vec_j_a) + np.dot(vec_j_b, vec_j_b)
                if denom <= 1e-30:
                    continue
                mu = (np.dot(vec_i_a, vec_j_a) + np.dot(vec_i_b, vec_j_b)) / denom
                for k in {int(round(mu)), int(math.floor(mu)), int(math.ceil(mu))}:
                    if k == 0 or abs(k) > 8:
                        continue
                    trial = best_q.copy()
                    trial[i] -= k * trial[j]
                    if not _valid_unimodular(trial):
                        continue
                    score = pair_shape_score(trial @ lattice_a, trial @ lattice_b)
                    if score + 1e-12 < best_score:
                        best_q = trial
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    if round(np.linalg.det(best_q)) < 0:
        best_q[0] *= -1
    return best_q.astype(int)


def independent_reduction_qs(
    lattice_a: np.ndarray,
    lattice_b: np.ndarray,
    enabled: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose independent unimodular transforms for the two equivalent supercells."""
    identity = np.eye(3, dtype=int)
    if not enabled:
        return identity, identity

    def candidates(lattice: np.ndarray) -> List[np.ndarray]:
        out = [identity]
        q_lll = lll_reduction_q(lattice)
        if _valid_unimodular(q_lll):
            out.append(q_lll)
        q_pair = pair_reduction_q(lattice, lattice, enabled=True)
        if _valid_unimodular(q_pair):
            out.append(q_pair)
        unique: List[np.ndarray] = []
        seen = set()
        for q in out:
            key = tuple(int(x) for x in q.reshape(-1).tolist())
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return unique

    best_qa, best_qb = identity, identity
    best_score = strain_penalty(lattice_a, lattice_b) + 0.5 * pair_shape_score(lattice_a, lattice_b)
    for qa in candidates(lattice_a):
        for qb in candidates(lattice_b):
            lat_a = qa @ lattice_a
            lat_b = qb @ lattice_b
            score = strain_penalty(lat_a, lat_b) + 0.5 * pair_shape_score(lat_a, lat_b)
            if score + 1e-12 < best_score:
                best_score = score
                best_qa, best_qb = qa, qb
    return best_qa.astype(int), best_qb.astype(int)


def apply_basis_q(lattice: np.ndarray, frac: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply equivalent basis transform lattice' = Q lattice, frac' = frac Q^-1."""
    if not _valid_unimodular(q):
        raise ValueError(f"Q must be unimodular, got:\n{q}")
    inv_q = np.rint(np.linalg.inv(q)).astype(int)
    new_lattice = q @ lattice
    new_frac = frac @ inv_q
    new_frac -= np.floor(new_frac)
    return new_lattice, new_frac


def canonical_geom_key(lattice: np.ndarray, ndigits: int = 8) -> Tuple[float, ...]:
    g = metric_tensor(lattice)
    lengths = np.linalg.norm(lattice, axis=1)
    values = (
        g[0, 0],
        g[1, 1],
        g[2, 2],
        g[0, 1],
        g[1, 2],
        g[0, 2],
        *sorted(lengths.tolist()),
    )
    return tuple(round(float(x), ndigits) for x in values)


def det_int(matrix: np.ndarray) -> int:
    return int(round(np.linalg.det(matrix)))


def hnf_flat(matrix: np.ndarray) -> HnfFlat:
    return tuple(int(x) for x in matrix.reshape(-1).tolist())


def hnf_from_flat(values: HnfFlat) -> np.ndarray:
    return np.array(values, dtype=int).reshape(3, 3)


def is_valid_hnf(matrix: np.ndarray, det_target: Optional[int] = None, diag_max: Optional[int] = None) -> bool:
    h = np.array(matrix, dtype=int)
    if h.shape != (3, 3):
        return False
    if h[0, 1] != 0 or h[0, 2] != 0 or h[1, 2] != 0:
        return False
    a, d, f = int(h[0, 0]), int(h[1, 1]), int(h[2, 2])
    if a <= 0 or d <= 0 or f <= 0:
        return False
    if diag_max is not None and max(a, d, f) > diag_max:
        return False
    # Row-lattice convention uses L' = H @ L.  For lower row-HNF, entries
    # below each diagonal are reduced modulo that column's diagonal element.
    if not (0 <= int(h[1, 0]) < a):
        return False
    if not (0 <= int(h[2, 0]) < a and 0 <= int(h[2, 1]) < d):
        return False
    if det_target is not None and a * d * f != int(det_target):
        return False
    return True


def hnf_matrices_with_det(det_target: int, diag_max: int) -> Iterator[np.ndarray]:
    """Yield all lower row-HNF matrices within a determinant bound."""
    det_target = int(det_target)
    for a in divisors(det_target):
        rem = det_target // a
        for d in divisors(rem):
            f = rem // d
            if max(a, d, f) > diag_max:
                continue
            for b in range(a):
                for c in range(a):
                    for e in range(d):
                        yield np.array([[a, 0, 0], [b, d, 0], [c, e, f]], dtype=int)


def count_hnfs_with_det(det_target: int, diag_max: int) -> int:
    total = 0
    for a in divisors(det_target):
        rem = det_target // a
        for d in divisors(rem):
            f = rem // d
            if max(a, d, f) <= diag_max:
                total += a * a * d
    return total


def sample_hnf_with_det(det_target: int, diag_max: int, rng: random.Random) -> Optional[np.ndarray]:
    triples: List[Tuple[int, int, int]] = []
    for a in divisors(det_target):
        rem = det_target // a
        for d in divisors(rem):
            f = rem // d
            if max(a, d, f) <= diag_max:
                triples.append((a, d, f))
    if not triples:
        return None
    a, d, f = rng.choice(triples)
    b = 0 if a == 1 else rng.randrange(a)
    c = 0 if a == 1 else rng.randrange(a)
    e = 0 if d == 1 else rng.randrange(d)
    return np.array([[a, 0, 0], [b, d, 0], [c, e, f]], dtype=int)


def make_supercell_fast(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    labels: np.ndarray,
    hnf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.array(hnf, dtype=int)
    if not is_valid_hnf(h):
        raise ValueError(f"Invalid HNF matrix:\n{h}")

    a, d, f = int(h[0, 0]), int(h[1, 1]), int(h[2, 2])
    new_lattice = h @ lattice
    inv_h = np.linalg.inv(h.astype(float))

    base = frac_coords @ inv_h
    base -= np.floor(base)

    shifts_int = np.stack(
        np.meshgrid(np.arange(a), np.arange(d), np.arange(f), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    shifts = shifts_int @ inv_h

    det_h = a * d * f
    frac = (base[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    frac -= np.floor(frac)
    labels_rep = np.repeat(labels, det_h)
    return new_lattice, labels_rep, frac


def wrap_frac(diff: np.ndarray) -> np.ndarray:
    return diff - np.round(diff)


def cartesian_periodic_cost(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    lattice_a: np.ndarray,
    lattice_b: np.ndarray,
    image_shell: int = 1,
) -> np.ndarray:
    """Symmetric Angstrom min-image cost, robust for skewed cells."""
    base = frac_a[:, None, :] - frac_b[None, :, :]
    shell = max(0, int(image_shell))
    offsets = np.array(
        [
            (i, j, k)
            for i in range(-shell, shell + 1)
            for j in range(-shell, shell + 1)
            for k in range(-shell, shell + 1)
        ],
        dtype=float,
    )
    best = None
    for offset in offsets:
        diff = base + offset
        cart_a = diff @ lattice_a
        cart_b = diff @ lattice_b
        d2 = 0.5 * (np.sum(cart_a * cart_a, axis=2) + np.sum(cart_b * cart_b, axis=2))
        if best is None:
            best = d2
        else:
            best = np.minimum(best, d2)
    return np.sqrt(np.maximum(best, 0.0))


def unique_shift(shift: np.ndarray, ndigits: int = 8) -> Tuple[float, float, float]:
    shift = shift - np.floor(shift)
    return tuple(round(float(x), ndigits) for x in shift)


def origin_shift_candidates(
    frac_a: np.ndarray,
    labels_a: np.ndarray,
    frac_b: np.ndarray,
    labels_b: np.ndarray,
    species: Sequence[str],
    max_candidates: int,
    rng: Optional[random.Random] = None,
) -> List[np.ndarray]:
    """Generate plausible origin shifts f_b + shift ~= f_a."""
    seen = {unique_shift(np.zeros(3))}
    shifts = [np.zeros(3, dtype=float)]
    if max_candidates <= 1:
        return shifts

    per_species_budget = max(1, max_candidates // max(1, len(species)))
    for sp in species:
        idx_a = np.where(labels_a == sp)[0]
        idx_b = np.where(labels_b == sp)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        local: List[Tuple[float, np.ndarray]] = []
        for ia in idx_a:
            for ib in idx_b:
                shift = frac_a[int(ia)] - frac_b[int(ib)]
                shift -= np.floor(shift)
                centered = shift - np.round(shift)
                local.append((float(np.dot(centered, centered)), shift))
        local.sort(key=lambda item: item[0])
        for _, shift in local[:per_species_budget]:
            key = unique_shift(shift)
            if key in seen:
                continue
            seen.add(key)
            shifts.append(shift)
            if len(shifts) >= max_candidates:
                return shifts

    return shifts[:max_candidates]


def hungarian_match_for_shift(
    frac_a: np.ndarray,
    labels_a: np.ndarray,
    frac_b: np.ndarray,
    labels_b: np.ndarray,
    lattice_a: np.ndarray,
    lattice_b: np.ndarray,
    species: Sequence[str],
    shift: np.ndarray,
    image_shell: int = 1,
) -> Optional[MatchResult]:
    shifted_b = frac_b + shift
    shifted_b -= np.floor(shifted_b)

    order_a: List[int] = []
    order_b: List[int] = []
    all_distances: List[float] = []

    for sp in species:
        idx_a = np.where(labels_a == sp)[0]
        idx_b = np.where(labels_b == sp)[0]
        if len(idx_a) != len(idx_b):
            return None
        if len(idx_a) == 0:
            continue
        cost = cartesian_periodic_cost(
            frac_a[idx_a],
            shifted_b[idx_b],
            lattice_a,
            lattice_b,
            image_shell=image_shell,
        )
        rows, cols = linear_sum_assignment(cost)
        order_a.extend(idx_a[rows].tolist())
        order_b.extend(idx_b[cols].tolist())
        all_distances.extend(cost[rows, cols].tolist())

    if not all_distances:
        return None

    distances = np.array(all_distances, dtype=float)
    rmsd2 = float(np.mean(distances * distances))
    return MatchResult(
        rmsd=float(math.sqrt(rmsd2)),
        rmsd2=rmsd2,
        shift=shift.copy(),
        order_a=np.array(order_a, dtype=int),
        order_b=np.array(order_b, dtype=int),
        distances=distances,
    )


def best_hungarian_match(
    problem: Problem,
    lattice_a: np.ndarray,
    labels_a: np.ndarray,
    frac_a: np.ndarray,
    lattice_b: np.ndarray,
    labels_b: np.ndarray,
    frac_b: np.ndarray,
) -> Optional[MatchResult]:
    candidates = origin_shift_candidates(
        frac_a,
        labels_a,
        frac_b,
        labels_b,
        problem.species,
        problem.cfg.max_origin_shifts,
        problem.rng,
    )
    best: Optional[MatchResult] = None
    for shift in candidates:
        match = hungarian_match_for_shift(
            frac_a,
            labels_a,
            frac_b,
            labels_b,
            lattice_a,
            lattice_b,
            problem.species,
            shift,
            image_shell=problem.cfg.image_shell,
        )
        if match is None:
            continue
        if best is None or match.rmsd2 < best.rmsd2:
            best = match
    return best


def spglib_structure_key(
    lattice: np.ndarray,
    labels: np.ndarray,
    frac: np.ndarray,
    species: Sequence[str],
    symprec: float,
) -> Tuple[object, ...]:
    """Canonical-ish key for symmetry-equivalent supercells."""
    species_to_number = {sp: idx + 1 for idx, sp in enumerate(species)}
    numbers = np.array([species_to_number[str(label)] for label in labels], dtype=int)
    positions = frac - np.floor(frac)

    if spglib is not None:
        try:
            std = spglib.standardize_cell(
                (np.array(lattice, dtype=float), positions, numbers),
                to_primitive=False,
                no_idealize=True,
                symprec=float(symprec),
            )
            if std is not None:
                lattice, positions, numbers = std
                positions = np.array(positions, dtype=float)
                positions -= np.floor(positions)
                numbers = np.array(numbers, dtype=int)
        except Exception:
            pass

    rows = []
    for number, pos in zip(numbers.tolist(), positions.tolist()):
        rows.append((int(number), *(round(float(x), 6) for x in pos)))
    rows.sort()
    return (canonical_geom_key(np.array(lattice, dtype=float), ndigits=6), tuple(rows))


def prepared_candidate_cells(
    problem: Problem,
    h_a: np.ndarray,
    h_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build supercells, then apply a common Q reduction for nicer equivalent cells."""
    sc_lat_a, sc_labels_a, sc_frac_a = make_supercell_fast(
        problem.lattice_a, problem.frac_a, problem.labels_a, h_a
    )
    sc_lat_b, sc_labels_b, sc_frac_b = make_supercell_fast(
        problem.lattice_b, problem.frac_b, problem.labels_b, h_b
    )

    q_a, q_b = independent_reduction_qs(sc_lat_a, sc_lat_b, enabled=problem.cfg.reduce_cell)
    sc_lat_a, sc_frac_a = apply_basis_q(sc_lat_a, sc_frac_a, q_a)
    sc_lat_b, sc_frac_b = apply_basis_q(sc_lat_b, sc_frac_b, q_b)
    lattice_penalty = strain_penalty(sc_lat_a, sc_lat_b)
    shape_penalty = pair_shape_score(sc_lat_a, sc_lat_b)
    return q_a, q_b, sc_lat_a, sc_labels_a, sc_frac_a, sc_lat_b, sc_labels_b, sc_frac_b, lattice_penalty, shape_penalty


def pair_t(problem: Problem, ind: Individual) -> Optional[int]:
    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    d_a = abs(det_int(h_a))
    d_b = abs(det_int(h_b))
    if d_a * problem.z_a != d_b * problem.z_b:
        return None
    if d_a % problem.d_a_base != 0 or d_b % problem.d_b_base != 0:
        return None
    t_a = d_a // problem.d_a_base
    t_b = d_b // problem.d_b_base
    return t_a if t_a == t_b else None


def eq_key_pair(problem: Problem, ind: Individual) -> Tuple[object, ...]:
    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    if problem.cfg.dedupe_mode == "none":
        return ind

    try:
        q_a, q_b, lat_a, labels_a, frac_a, lat_b, labels_b, frac_b, _, _ = prepared_candidate_cells(problem, h_a, h_b)
    except Exception:
        return ("invalid", ind)

    base = (
        abs(det_int(h_a)),
        abs(det_int(h_b)),
        canonical_geom_key(lat_a),
        canonical_geom_key(lat_b),
    )
    if problem.cfg.dedupe_mode == "symmetry":
        return (
            *base,
            spglib_structure_key(lat_a, labels_a, frac_a, problem.species, problem.cfg.symprec),
            spglib_structure_key(lat_b, labels_b, frac_b, problem.species, problem.cfg.symprec),
        )
    return (
        *base,
            tuple(int(x) for x in q_a.reshape(-1).tolist()),
            tuple(int(x) for x in q_b.reshape(-1).tolist()),
        )


def evaluate(problem: Problem, ind: Individual, with_match: bool = False) -> Tuple[EvalResult, Optional[MatchResult]]:
    if not with_match and ind in problem.fitness_cache:
        return problem.fitness_cache[ind], None

    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    d_a = abs(det_int(h_a))
    d_b = abs(det_int(h_b))
    identity_q = tuple(int(x) for x in np.eye(3, dtype=int).reshape(-1).tolist())

    bad = EvalResult(1e30, 1e15, 1e30, 1e30, 1e30, (0.0, 0.0, 0.0), d_a, d_b, identity_q, identity_q)

    if not is_valid_hnf(h_a, diag_max=problem.cfg.diag_max):
        return bad, None
    if not is_valid_hnf(h_b, diag_max=problem.cfg.diag_max):
        return bad, None
    if d_a * problem.z_a != d_b * problem.z_b:
        return bad, None

    try:
        (
            q_a,
            q_b,
            sc_lat_a,
            sc_labels_a,
            sc_frac_a,
            sc_lat_b,
            sc_labels_b,
            sc_frac_b,
            geom,
            shape,
        ) = prepared_candidate_cells(
            problem,
            h_a,
            h_b,
        )
    except Exception:
        return bad, None

    if sc_frac_a.shape[0] != sc_frac_b.shape[0]:
        return bad, None
    for sp in problem.species:
        if np.sum(sc_labels_a == sp) != np.sum(sc_labels_b == sp):
            return bad, None

    match = best_hungarian_match(
        problem, sc_lat_a, sc_labels_a, sc_frac_a, sc_lat_b, sc_labels_b, sc_frac_b
    )
    if match is None:
        return bad, None

    fitness = (
        problem.cfg.weight_rmsd * match.rmsd2
        + problem.cfg.weight_geom * geom
        + problem.cfg.weight_shape * shape
    )
    result = EvalResult(
        fitness=float(fitness),
        rmsd=float(match.rmsd),
        rmsd2=float(match.rmsd2),
        geom=float(geom),
        shape=float(shape),
        shift=tuple(float(x) for x in match.shift),
        det_a=int(d_a),
        det_b=int(d_b),
        q_a=tuple(int(x) for x in q_a.reshape(-1).tolist()),
        q_b=tuple(int(x) for x in q_b.reshape(-1).tolist()),
    )
    problem.fitness_cache[ind] = result
    return result, match if with_match else None


def random_individual(problem: Problem, t: Optional[int] = None) -> Optional[Individual]:
    if t is None:
        t = problem.rng.randint(problem.cfg.det_mult_min, problem.cfg.det_mult_max)
    det_a = int(t) * problem.d_a_base
    det_b = int(t) * problem.d_b_base
    h_a = sample_hnf_with_det(det_a, problem.cfg.diag_max, problem.rng)
    h_b = sample_hnf_with_det(det_b, problem.cfg.diag_max, problem.rng)
    if h_a is None or h_b is None:
        return None
    return hnf_flat(h_a), hnf_flat(h_b)


def combine_hnf(a: np.ndarray, b: np.ndarray, rng: random.Random) -> np.ndarray:
    """HNF-aware crossover for two matrices with the same determinant."""
    if tuple(np.diag(a).tolist()) != tuple(np.diag(b).tolist()):
        return a.copy() if rng.random() < 0.5 else b.copy()
    child = a.copy()
    for i, j in ((1, 0), (2, 0), (2, 1)):
        child[i, j] = a[i, j] if rng.random() < 0.5 else b[i, j]
    return child


def crossover_individual(problem: Problem, left: Individual, right: Individual) -> Individual:
    t_left = pair_t(problem, left)
    t_right = pair_t(problem, right)
    if t_left is None or t_right is None or t_left != t_right:
        base = left if problem.rng.random() < 0.5 else right
        return mutate_individual(problem, base, force_matrix_resample=False)

    h_a_left, h_b_left = hnf_from_flat(left[0]), hnf_from_flat(left[1])
    h_a_right, h_b_right = hnf_from_flat(right[0]), hnf_from_flat(right[1])
    child_a = combine_hnf(h_a_left, h_a_right, problem.rng)
    child_b = combine_hnf(h_b_left, h_b_right, problem.rng)
    child = (hnf_flat(child_a), hnf_flat(child_b))
    if pair_t(problem, child) != t_left:
        return left if problem.rng.random() < 0.5 else right
    return child


def mutate_shear(h: np.ndarray, rng: random.Random) -> np.ndarray:
    child = h.copy()
    choices: List[Tuple[int, int, int]] = []
    if child[0, 0] > 1:
        choices.append((1, 0, int(child[0, 0])))
        choices.append((2, 0, int(child[0, 0])))
    if child[1, 1] > 1:
        choices.append((2, 1, int(child[1, 1])))
    if choices:
        i, j, modulus = rng.choice(choices)
        child[i, j] = (int(child[i, j]) + rng.choice([-2, -1, 1, 2])) % modulus
    return child


def mutate_individual(
    problem: Problem,
    ind: Individual,
    force_matrix_resample: bool = False,
) -> Individual:
    t = pair_t(problem, ind)
    if t is None:
        fresh = random_individual(problem)
        return fresh if fresh is not None else ind

    if force_matrix_resample or problem.rng.random() < 0.12:
        if problem.rng.random() < 0.35:
            t = max(
                problem.cfg.det_mult_min,
                min(problem.cfg.det_mult_max, t + problem.rng.choice([-1, 1])),
            )
        fresh = random_individual(problem, t=t)
        return fresh if fresh is not None else ind

    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    if problem.rng.random() < 0.5:
        h_a = mutate_shear(h_a, problem.rng)
    else:
        h_b = mutate_shear(h_b, problem.rng)
    return hnf_flat(h_a), hnf_flat(h_b)


def generate_population(problem: Problem, n: int) -> List[Individual]:
    pop: List[Individual] = []
    seen = set()
    max_attempts = max(20000, n * 2000)
    attempts = 0

    while len(pop) < n and attempts < max_attempts:
        attempts += 1
        ind = random_individual(problem)
        if ind is None or ind in seen:
            continue
        key = eq_key_pair(problem, ind)
        if key in problem.eq_cache:
            continue
        seen.add(ind)
        problem.eq_cache.add(key)
        pop.append(ind)

    if len(pop) < n:
        print(f"[Warn] Initial population truncated to {len(pop)}.", flush=True)
    return pop


def select_parent(problem: Problem, pop: Sequence[Individual], k: int = 3) -> Individual:
    contestants = problem.rng.sample(list(pop), k=min(k, len(pop)))
    scored = [(evaluate(problem, ind)[0].fitness, ind) for ind in contestants]
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def evolve_population(problem: Problem, pop: List[Individual]) -> List[Individual]:
    scored = [(evaluate(problem, ind)[0].fitness, ind) for ind in pop]
    scored.sort(key=lambda item: item[0])
    elites = [ind for _, ind in scored[: problem.cfg.elite_keep]]

    replace_count = min(problem.cfg.substitute_number, len(pop))
    worst_indices = np.argsort([score for score, _ in scored])[-replace_count:]

    # worst_indices indexes scored, not pop. Map back by identity.
    worst_set = {id(scored[int(i)][1]) for i in worst_indices}
    for idx, old in enumerate(pop):
        if id(old) not in worst_set:
            continue
        parent_a = select_parent(problem, pop)
        parent_b = select_parent(problem, pop)
        child = crossover_individual(problem, parent_a, parent_b)
        if problem.rng.random() < 0.85:
            child = mutate_individual(problem, child)
        for _ in range(6):
            key = eq_key_pair(problem, child)
            if key not in problem.eq_cache:
                problem.eq_cache.add(key)
                break
            child = mutate_individual(problem, child, force_matrix_resample=True)
        pop[idx] = child

    rescored = [(evaluate(problem, ind)[0].fitness, i) for i, ind in enumerate(pop)]
    rescored.sort(key=lambda item: item[0], reverse=True)
    for elite, (_, idx) in zip(elites, rescored[: len(elites)]):
        pop[idx] = elite
    return pop


def top_k(problem: Problem, pop: Sequence[Individual], k: int) -> List[Tuple[EvalResult, Individual]]:
    scored = [(evaluate(problem, ind)[0], ind) for ind in pop]
    scored.sort(key=lambda item: item[0].fitness)
    return scored[:k]


def count_exhaustive_pairs(problem: Problem) -> int:
    total = 0
    for t in range(problem.cfg.det_mult_min, problem.cfg.det_mult_max + 1):
        n_a = count_hnfs_with_det(t * problem.d_a_base, problem.cfg.diag_max)
        n_b = count_hnfs_with_det(t * problem.d_b_base, problem.cfg.diag_max)
        total += n_a * n_b
    return total


def enumerate_population(problem: Problem) -> List[Individual]:
    total = count_exhaustive_pairs(problem)
    if total > problem.cfg.exhaustive_limit:
        raise RuntimeError(
            f"Exhaustive space has {total} pairs, above --exhaustive-limit "
            f"{problem.cfg.exhaustive_limit}. Reduce --det-mult-max/--diag-max "
            "or use --mode ga."
        )
    pop: List[Individual] = []
    for t in range(problem.cfg.det_mult_min, problem.cfg.det_mult_max + 1):
        hnfs_a = list(hnf_matrices_with_det(t * problem.d_a_base, problem.cfg.diag_max))
        hnfs_b = list(hnf_matrices_with_det(t * problem.d_b_base, problem.cfg.diag_max))
        for h_a in hnfs_a:
            for h_b in hnfs_b:
                pop.append((hnf_flat(h_a), hnf_flat(h_b)))
    return pop


def export_candidate(problem: Problem, rank: int, ind: Individual, result: EvalResult) -> Dict[str, object]:
    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    q_a, q_b, lat_a, labels_a, frac_a, lat_b, labels_b, frac_b, _, _ = prepared_candidate_cells(problem, h_a, h_b)
    _, match = evaluate(problem, ind, with_match=True)
    if match is None:
        raise RuntimeError("Cannot export candidate without a valid match.")

    labels_a_ord = labels_a[match.order_a]
    labels_b_ord = labels_b[match.order_b]
    frac_a_ord = frac_a[match.order_a]
    frac_b_shifted = frac_b + match.shift
    frac_b_shifted -= np.floor(frac_b_shifted)
    frac_b_ord = frac_b_shifted[match.order_b]

    problem.cfg.output_dir.mkdir(parents=True, exist_ok=True)
    struct_a = Structure(Lattice(lat_a), labels_a_ord.tolist(), frac_a_ord.tolist(), coords_are_cartesian=False)
    struct_b = Structure(Lattice(lat_b), labels_b_ord.tolist(), frac_b_ord.tolist(), coords_are_cartesian=False)
    struct_a.to(filename=str(problem.cfg.output_dir / f"{rank:02d}-POSCAR-i.vasp"), fmt="poscar")
    struct_b.to(filename=str(problem.cfg.output_dir / f"{rank:02d}-POSCAR-f.vasp"), fmt="poscar")

    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H1_hnf.txt", h_a, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H2_hnf.txt", h_b, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-Q_A.txt", q_a, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-Q_B.txt", q_b, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H1_display_QH.txt", q_a @ h_a, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H2_display_QH.txt", q_b @ h_b, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-shuffle-distances.txt", match.distances, fmt="%.10f")

    return {
        "rank": rank,
        "fitness": result.fitness,
        "rmsd_ang": result.rmsd,
        "rmsd2_ang2": result.rmsd2,
        "lattice_penalty": result.geom,
        "gram_penalty": result.geom,
        "shape_penalty": result.shape,
        "det_H1": result.det_a,
        "det_H2": result.det_b,
        "origin_shift": [float(x) for x in match.shift],
        "Q_A": q_a.tolist(),
        "Q_B": q_b.tolist(),
        "H1_hnf": h_a.tolist(),
        "H2_hnf": h_b.tolist(),
        "H1_display_QH": (q_a @ h_a).tolist(),
        "H2_display_QH": (q_b @ h_b).tolist(),
    }


def save_summary(problem: Problem, records: List[Dict[str, object]]) -> None:
    out = problem.cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "file_a": str(problem.cfg.file_a),
                "file_b": str(problem.cfg.file_b),
                "species": list(problem.species),
                "z_a": problem.z_a,
                "z_b": problem.z_b,
                "d_a_base": problem.d_a_base,
                "d_b_base": problem.d_b_base,
                "config": {
                    "mode": problem.cfg.mode,
                    "seed": problem.cfg.seed,
                    "det_mult_min": problem.cfg.det_mult_min,
                    "det_mult_max": problem.cfg.det_mult_max,
                    "diag_max": problem.cfg.diag_max,
                    "weight_rmsd": problem.cfg.weight_rmsd,
                    "weight_geom": problem.cfg.weight_geom,
                    "weight_shape": problem.cfg.weight_shape,
                    "max_origin_shifts": problem.cfg.max_origin_shifts,
                    "reduce_cell": problem.cfg.reduce_cell,
                    "image_shell": problem.cfg.image_shell,
                    "dedupe_mode": problem.cfg.dedupe_mode,
                    "symprec": problem.cfg.symprec,
                },
                "candidates": records,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "rank",
        "fitness",
        "rmsd_ang",
        "rmsd2_ang2",
        "lattice_penalty",
        "shape_penalty",
        "det_H1",
        "det_H2",
        "origin_shift",
    ]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fieldnames}
            row["origin_shift"] = json.dumps(row["origin_shift"])
            writer.writerow(row)

    if records:
        np.savetxt(out / "best_H1_hnf.txt", np.array(records[0]["H1_hnf"], dtype=int), fmt="%d")
        np.savetxt(out / "best_H2_hnf.txt", np.array(records[0]["H2_hnf"], dtype=int), fmt="%d")
        np.savetxt(out / "best_Q_A.txt", np.array(records[0]["Q_A"], dtype=int), fmt="%d")
        np.savetxt(out / "best_Q_B.txt", np.array(records[0]["Q_B"], dtype=int), fmt="%d")
        np.savetxt(out / "best_H1_display_QH.txt", np.array(records[0]["H1_display_QH"], dtype=int), fmt="%d")
        np.savetxt(out / "best_H2_display_QH.txt", np.array(records[0]["H2_display_QH"], dtype=int), fmt="%d")


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Robust HNF supercell matcher for two crystals.")
    parser.add_argument("--file-a", default="POSCAR1", type=Path)
    parser.add_argument("--file-b", default="POSCAR2", type=Path)
    parser.add_argument("--output-dir", default=Path("best_solutions_v2"), type=Path)
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--mode", choices=("ga", "enumerate"), default="ga")
    parser.add_argument("--pop-size", default=10000, type=int)
    parser.add_argument("--init-keep", default=1000, type=int)
    parser.add_argument("--generations", default=60, type=int)
    parser.add_argument("--substitute-number", default=192, type=int)
    parser.add_argument("--elite-keep", default=32, type=int)
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--det-mult-min", default=1, type=int)
    parser.add_argument("--det-mult-max", default=12, type=int)
    parser.add_argument("--diag-max", default=12, type=int)
    parser.add_argument("--weight-geom", default=1.0, type=float)
    parser.add_argument("--weight-rmsd", default=1.0, type=float)
    parser.add_argument(
        "--weight-shape",
        default=0.05,
        type=float,
        help="Penalty for skewed/elongated equivalent supercell bases after Q reduction.",
    )
    parser.add_argument("--max-origin-shifts", default=64, type=int)
    parser.add_argument(
        "--no-reduce-cell",
        action="store_true",
        help="Do not apply the common unimodular Q reduction before scoring/export.",
    )
    parser.add_argument(
        "--image-shell",
        default=1,
        type=int,
        help="Neighboring fractional image shell for Cartesian min-image distances.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=("symmetry", "geom", "none"),
        default="symmetry",
        help="Deduplicate candidates by spglib-standardized supercell, reduced geometry, or not at all.",
    )
    parser.add_argument("--symprec", default=1e-5, type=float)
    parser.add_argument("--exhaustive-limit", default=250000, type=int)
    args = parser.parse_args(argv)

    det_mult_min = max(1, args.det_mult_min)
    det_mult_max = max(det_mult_min, args.det_mult_max)

    return Config(
        file_a=args.file_a,
        file_b=args.file_b,
        output_dir=args.output_dir,
        seed=args.seed,
        mode=args.mode,
        pop_size=max(1, args.pop_size),
        init_keep=max(1, args.init_keep),
        generations=max(0, args.generations),
        substitute_number=max(1, args.substitute_number),
        elite_keep=max(1, args.elite_keep),
        top_k=max(1, args.top_k),
        det_mult_min=det_mult_min,
        det_mult_max=det_mult_max,
        diag_max=max(1, args.diag_max),
        weight_geom=float(args.weight_geom),
        weight_rmsd=float(args.weight_rmsd),
        weight_shape=float(args.weight_shape),
        max_origin_shifts=max(1, args.max_origin_shifts),
        reduce_cell=not args.no_reduce_cell,
        image_shell=max(0, args.image_shell),
        dedupe_mode=args.dedupe_mode,
        symprec=max(1e-12, float(args.symprec)),
        exhaustive_limit=max(1, args.exhaustive_limit),
    )


def build_problem(cfg: Config) -> Problem:
    if PYMATGEN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "pymatgen is required to read/write POSCAR files. "
            f"Original import error: {PYMATGEN_IMPORT_ERROR}"
        )
    struct_a = Structure.from_file(str(cfg.file_a))
    struct_b = Structure.from_file(str(cfg.file_b))

    lattice_a = row_lattice_matrix(struct_a)
    lattice_b = row_lattice_matrix(struct_b)
    labels_a, frac_a = labels_and_frac(struct_a)
    labels_b, frac_b = labels_and_frac(struct_b)

    species = tuple(sorted(set(struct_a.composition.get_el_amt_dict()) | set(struct_b.composition.get_el_amt_dict())))
    counts_a = species_counts(struct_a, species)
    counts_b = species_counts(struct_b, species)
    z_a = gcd_array(counts_a)
    z_b = gcd_array(counts_b)
    formula_a = (counts_a // z_a).astype(int)
    formula_b = (counts_b // z_b).astype(int)
    if not np.array_equal(formula_a, formula_b):
        raise ValueError(
            "Incompatible stoichiometric ratios:\n"
            f"  A counts={counts_a.tolist()}, Z={z_a}, formula={formula_a.tolist()}\n"
            f"  B counts={counts_b.tolist()}, Z={z_b}, formula={formula_b.tolist()}"
        )

    gcd_z = math.gcd(int(z_a), int(z_b))
    d_a_base = z_b // gcd_z
    d_b_base = z_a // gcd_z
    rng = random.Random(cfg.seed) if cfg.seed is not None else random.Random()

    return Problem(
        cfg=cfg,
        rng=rng,
        struct_a=struct_a,
        struct_b=struct_b,
        lattice_a=lattice_a,
        lattice_b=lattice_b,
        labels_a=labels_a,
        labels_b=labels_b,
        frac_a=frac_a,
        frac_b=frac_b,
        species=species,
        z_a=z_a,
        z_b=z_b,
        d_a_base=d_a_base,
        d_b_base=d_b_base,
        fitness_cache={},
        eq_cache=set(),
    )


def run_ga(problem: Problem) -> List[Individual]:
    print("[Info] Building initial population ...", flush=True)
    population = generate_population(problem, problem.cfg.pop_size)
    if not population:
        raise RuntimeError("Initial population is empty. Increase --det-mult-max or --diag-max.")

    scored = [(evaluate(problem, ind)[0].fitness, ind) for ind in population]
    scored.sort(key=lambda item: item[0])
    population = [ind for _, ind in scored[: min(problem.cfg.init_keep, len(scored))]]
    print(f"[Info] Init best fitness = {scored[0][0]:.8e} | pop = {len(population)}", flush=True)

    for gen in range(1, problem.cfg.generations + 1):
        population = evolve_population(problem, population)
        best = top_k(problem, population, 1)[0][0]
        print(
            f"[Gen {gen:03d}] fitness={best.fitness:.8e} "
            f"rmsd={best.rmsd:.6f} A lattice={best.geom:.6e} shape={best.shape:.6e}",
            flush=True,
        )
    return population


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    problem = build_problem(cfg)

    print(f"[Info] Files: A='{cfg.file_a}', B='{cfg.file_b}'", flush=True)
    print(f"[Info] Species: {list(problem.species)}", flush=True)
    print(f"[Info] Formula units: Z_A={problem.z_a}, Z_B={problem.z_b}", flush=True)
    print(
        f"[Info] Base det pair: det(H1)=t*{problem.d_a_base}, "
        f"det(H2)=t*{problem.d_b_base}, t={cfg.det_mult_min}..{cfg.det_mult_max}",
        flush=True,
    )
    print(
        f"[Info] Q reduction={cfg.reduce_cell}, dedupe={cfg.dedupe_mode}, "
        f"image_shell={cfg.image_shell}, spglib={'yes' if spglib is not None else 'no'}",
        flush=True,
    )

    if cfg.mode == "enumerate":
        total = count_exhaustive_pairs(problem)
        print(f"[Info] Exhaustive candidate pairs inside bounds: {total}", flush=True)
        population = enumerate_population(problem)
    else:
        population = run_ga(problem)

    results = top_k(problem, population, cfg.top_k)
    print("\n[Result] Top candidates:", flush=True)
    records: List[Dict[str, object]] = []
    for rank, (result, ind) in enumerate(results, 1):
        print(
            f"  #{rank:02d}: fitness={result.fitness:.8e} "
            f"rmsd={result.rmsd:.6f} A lattice={result.geom:.6e} shape={result.shape:.6e} "
            f"det(H1)={result.det_a} det(H2)={result.det_b}",
            flush=True,
        )
        records.append(export_candidate(problem, rank, ind, result))

    save_summary(problem, records)
    if records:
        print("\n[Best] H1 HNF:")
        print(np.array(records[0]["H1_hnf"], dtype=int))
        print("[Best] H2 HNF:")
        print(np.array(records[0]["H2_hnf"], dtype=int))
        print("[Best] Q-reduced display H1 = Q_A @ H1:")
        print(np.array(records[0]["H1_display_QH"], dtype=int))
        print("[Best] Q-reduced display H2 = Q_B @ H2:")
        print(np.array(records[0]["H2_display_QH"], dtype=int))
    print(f"\n[Done] Exported matched POSCARs and metrics to '{cfg.output_dir}'.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1)
