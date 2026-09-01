#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust supercell matching between two crystals.

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
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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

try:
    from snf_csm_invariant import (
        SaturationLimitError,
        resolve_atomic_period_columns,
        row_hnf_3x3,
        saturate_supercell_columns,
    )
    INTRINSIC_PERIOD_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency/runtime dependent
    SaturationLimitError = RuntimeError
    resolve_atomic_period_columns = None
    row_hnf_3x3 = None
    saturate_supercell_columns = None
    INTRINSIC_PERIOD_IMPORT_ERROR = exc


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
    diag_max: Optional[int]
    weight_geom: float
    weight_rmsd: float
    rmsd_scale_ang: Optional[float]
    weight_shape: float
    max_origin_shifts: int
    reduce_cell: bool
    image_shell: int
    dedupe_mode: str
    symprec: float
    exhaustive_limit: int
    export_lattice_mode: str
    intrinsic_period_audit: bool
    intrinsic_period_max_points: int
    enumerate_stop_after_t: Optional[int] = None
    enumerate_resume: bool = False
    ga_stop_after_generation: Optional[int] = None
    ga_resume: bool = False


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
    diag_max_source: str
    diag_max_no_truncation_bound: int
    rmsd_scale_ang: float
    rmsd_scale_source: str
    mean_nn_a_ang: float
    mean_nn_b_ang: float
    fitness_cache: Dict[object, "EvalResult"]
    eq_cache: set


@dataclass
class MatchResult:
    rmsd: float
    rmsd2: float
    shift: np.ndarray
    order_a: np.ndarray
    order_b: np.ndarray
    translations_b: np.ndarray
    distances: np.ndarray


@dataclass
class EvalResult:
    fitness: float
    rmsd: float
    rmsd2: float
    rmsd2_normalized: float
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


def gcd_ints(values: Iterable[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, abs(int(value)))
    return g


def det3_int(matrix: np.ndarray) -> int:
    m = np.asarray(matrix, dtype=object)
    return int(
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def smith_invariant_3x3(matrix: np.ndarray) -> Tuple[int, int, int]:
    """SNF invariant factors for a nonsingular 3x3 integer matrix.

    For full-rank integer matrices, d1 is gcd of entries, d1*d2 is gcd of
    2x2 minors, and d1*d2*d3 is abs(det). This avoids an extra SymPy runtime
    dependency and is enough for HNF supercell matrices.
    """
    m = np.asarray(matrix, dtype=int)
    if m.shape != (3, 3):
        raise ValueError(f"SNF tag expects a 3x3 matrix, got shape {m.shape}.")

    det_abs = abs(det3_int(m))
    if det_abs == 0:
        raise ValueError("SNF tag expects a nonsingular supercell matrix.")

    d1 = gcd_ints(m.ravel())
    minors_2x2: List[int] = []
    for r0 in range(3):
        for r1 in range(r0 + 1, 3):
            for c0 in range(3):
                for c1 in range(c0 + 1, 3):
                    minors_2x2.append(int(m[r0, c0] * m[r1, c1] - m[r0, c1] * m[r1, c0]))
    g2 = gcd_ints(minors_2x2)
    if d1 <= 0 or g2 <= 0 or g2 % d1 != 0 or det_abs % g2 != 0:
        raise ValueError(f"Cannot derive SNF invariant from matrix:\n{m}")

    return int(d1), int(g2 // d1), int(det_abs // g2)


def snf_key(snf: Sequence[int]) -> str:
    return "x".join(str(int(x)) for x in snf)


def add_snf_tags(record: Dict[str, object]) -> Dict[str, object]:
    h1 = np.array(record["H1_hnf"], dtype=int)
    h2 = np.array(record["H2_hnf"], dtype=int)
    snf_h1 = smith_invariant_3x3(h1)
    snf_h2 = smith_invariant_3x3(h2)
    record["snf_H1"] = list(snf_h1)
    record["snf_H2"] = list(snf_h2)
    record["snf_pair_key"] = f"H1:{snf_key(snf_h1)}|H2:{snf_key(snf_h2)}"
    return record


def _add_period_signature_fields(
    record: Dict[str, object],
    prefix: str,
    signature: object,
) -> None:
    """Serialize an exact saturated-period signature into one candidate row."""

    source_hnf = getattr(signature, "source_hnf_row")
    target_hnf = getattr(signature, "target_hnf_row")
    relative_q = getattr(signature, "relative_q_row")
    source_snf = getattr(signature, "source_snf")
    target_snf = getattr(signature, "target_snf")
    record.update(
        {
            f"{prefix}_H1_hnf": [list(row) for row in source_hnf],
            f"{prefix}_H2_hnf": [list(row) for row in target_hnf],
            f"{prefix}_relative_Q_row": [list(row) for row in relative_q],
            f"{prefix}_snf_H1": list(source_snf),
            f"{prefix}_snf_H2": list(target_snf),
            f"{prefix}_snf_pair_key": getattr(signature, "snf_pair_key"),
            f"{prefix}_coupled_imt_key": getattr(signature, "coupled_imt_key"),
            f"{prefix}_source_index": int(getattr(signature, "source_index")),
            f"{prefix}_target_index": int(getattr(signature, "target_index")),
            f"{prefix}_reduction_factor": int(
                getattr(signature, "source_reduction_factor")
            ),
        }
    )


def add_intrinsic_period_tags(
    record: Dict[str, object],
    source_basis_row: np.ndarray,
    target_basis_row: np.ndarray,
    source_fractional: np.ndarray,
    target_fractional: np.ndarray,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    *,
    relative_q_row: Optional[np.ndarray] = None,
    enabled: bool = True,
    max_points: int = 250_000,
) -> Dict[str, object]:
    """Attach exact SLM and atom-correspondence periods to one exported match.

    ``Q_A`` and ``Q_B`` in older output are display-only common basis changes.
    They are deliberately not treated as the relative IMT coupling.  The
    current candidate generator has relative ``Q=I``; it is recorded explicitly
    so a future nonidentity-Q generator cannot be confused with display cleanup.
    """

    if "snf_pair_key" not in record:
        add_snf_tags(record)
    raw_q = (
        np.eye(3, dtype=int)
        if relative_q_row is None
        else np.asarray(relative_q_row, dtype=int)
    )
    if raw_q.shape != (3, 3) or abs(det_int(raw_q)) != 1:
        raise ValueError("relative_q_row must be a 3x3 unimodular integer matrix.")

    record.update(
        {
            "raw_supercell_snf_pair_key": record["snf_pair_key"],
            "relative_Q_row": raw_q.tolist(),
            "intrinsic_period_exact": False,
            "atomic_period_verified": False,
            "intrinsic_period_status": "disabled" if not enabled else "unverified",
            "intrinsic_period_error": "",
        }
    )
    if not enabled:
        return record
    if INTRINSIC_PERIOD_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Intrinsic-period auditing requires snf_csm_invariant.py and sympy. "
            f"Original import error: {INTRINSIC_PERIOD_IMPORT_ERROR}"
        )
    assert saturate_supercell_columns is not None
    assert resolve_atomic_period_columns is not None

    source_row = np.asarray(source_basis_row, dtype=int)
    target_row = np.asarray(target_basis_row, dtype=int)
    if source_row.shape != (3, 3) or target_row.shape != (3, 3):
        raise ValueError("Intrinsic-period auditing expects two 3x3 row bases.")

    try:
        lattice_signature = saturate_supercell_columns(
            source_row.T.tolist(),
            target_row.T.tolist(),
            max_residue_points=max_points,
        )
        atomic_signature = resolve_atomic_period_columns(
            source_row.T.tolist(),
            target_row.T.tolist(),
            source_fractional,
            target_fractional,
            [str(label) for label in source_labels],
            [str(label) for label in target_labels],
            max_residue_points=max_points,
            max_coset_points=max_points,
        )
        _add_period_signature_fields(record, "slm", lattice_signature)
        _add_period_signature_fields(record, "atomic", atomic_signature)
        record.update(
            {
                "intrinsic_period_exact": True,
                "atomic_period_verified": bool(
                    getattr(atomic_signature, "atomic_period_verified")
                ),
                "slm_and_atomic_period_agree": (
                    lattice_signature.coupled_imt_key
                    == atomic_signature.coupled_imt_key
                ),
                "intrinsic_period_status": (
                    "verified_reduced"
                    if atomic_signature.source_reduction_factor > 1
                    else "verified_minimal"
                ),
            }
        )
    except Exception as exc:
        record.update(
            {
                "intrinsic_period_exact": False,
                "atomic_period_verified": False,
                "intrinsic_period_status": "unverified",
                "intrinsic_period_error": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


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


def mean_nearest_neighbor_distance(structure: Structure) -> float:
    """Mean per-site nearest periodic-neighbor distance in Angstrom."""
    cutoff = max(float(length) for length in structure.lattice.abc) * (1.0 + 1e-10)
    neighbor_groups = structure.get_all_neighbors(cutoff)
    nearest: List[float] = []
    for neighbors in neighbor_groups:
        distances = [
            float(neighbor.nn_distance)
            for neighbor in neighbors
            if float(neighbor.nn_distance) > 1e-8
        ]
        if not distances:
            raise ValueError("Cannot determine a nonzero periodic nearest-neighbor distance.")
        nearest.append(min(distances))
    if not nearest:
        raise ValueError("Cannot determine an RMSD scale for an empty structure.")
    return float(np.mean(nearest))


def resolve_rmsd_scale(
    requested: Optional[float],
    structure_a: Structure,
    structure_b: Structure,
) -> Tuple[float, str, float, float]:
    mean_nn_a = mean_nearest_neighbor_distance(structure_a)
    mean_nn_b = mean_nearest_neighbor_distance(structure_b)
    if requested is None:
        return 0.5 * (mean_nn_a + mean_nn_b), "auto-mean-nearest-neighbor", mean_nn_a, mean_nn_b
    scale = float(requested)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("--rmsd-scale-ang must be a positive finite number.")
    return scale, "manual", mean_nn_a, mean_nn_b


def fitness_terms(
    rmsd2_ang2: float,
    rmsd_scale_ang: float,
    lattice_penalty: float,
    shape_penalty: float,
    weight_rmsd: float,
    weight_geom: float,
    weight_shape: float,
) -> Tuple[float, float, float, float, float]:
    """Return normalized RMSD2, three weighted terms, and total fitness."""
    scale = float(rmsd_scale_ang)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("RMSD normalization scale must be positive and finite.")
    normalized_rmsd2 = float(rmsd2_ang2) / (scale * scale)
    atom_term = float(weight_rmsd) * normalized_rmsd2
    geom_term = float(weight_geom) * float(lattice_penalty)
    shape_term = float(weight_shape) * float(shape_penalty)
    return normalized_rmsd2, atom_term, geom_term, shape_term, atom_term + geom_term + shape_term


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
    """Choose independent unimodular transforms for legacy diagnostics."""
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


def rotation_free_final_lattice(lattice_a: np.ndarray, lattice_b: np.ndarray) -> np.ndarray:
    """Rotate final row-lattice into the polar, rotation-free frame of lattice_a."""
    c_a = np.asarray(lattice_a, dtype=float).T
    c_b = np.asarray(lattice_b, dtype=float).T
    u, _, vt = np.linalg.svd(c_b @ np.linalg.inv(c_a), full_matrices=False)
    rotation = u @ vt
    return np.asarray(lattice_b, dtype=float) @ rotation


def ab_cartesian_frame(lattice: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal frame fixed by the a axis and the a-b plane."""
    rows = np.asarray(lattice, dtype=float)
    if rows.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 lattice, got {rows.shape}.")
    e_a = rows[0] / np.linalg.norm(rows[0])
    b_perp = rows[1] - float(np.dot(rows[1], e_a)) * e_a
    b_perp_norm = float(np.linalg.norm(b_perp))
    if b_perp_norm <= 1e-12:
        raise ValueError("Cannot define an a-b frame from collinear lattice vectors.")
    e_b = b_perp / b_perp_norm
    e_c = np.cross(e_a, e_b)
    return np.vstack((e_a, e_b, e_c))


def initial_to_final_ab_rotation(lattice_a: np.ndarray, lattice_b: np.ndarray) -> np.ndarray:
    """Proper Cartesian rotation aligning initial a and a-b frame to the final one."""
    frame_a = ab_cartesian_frame(lattice_a)
    frame_b = ab_cartesian_frame(lattice_b)
    rotation = frame_a.T @ frame_b
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("Computed endpoint-axis alignment is not orthogonal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
        raise ValueError("Computed endpoint-axis alignment is not a proper rotation.")
    return rotation


def neb_aligned_lattices(
    lattice_a: np.ndarray,
    lattice_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the final frame and rigidly rotate the initial endpoint into it."""
    rotation_a = initial_to_final_ab_rotation(lattice_a, lattice_b)
    return (
        np.asarray(lattice_a, dtype=float) @ rotation_a,
        np.asarray(lattice_b, dtype=float),
        rotation_a,
    )


def half_distorted_lattice(lattice_a: np.ndarray, lattice_b: np.ndarray) -> np.ndarray:
    """Crystmatch-style half-distorted row-lattice from the polar stretch."""
    c_a = np.asarray(lattice_a, dtype=float).T
    c_b = np.asarray(lattice_b, dtype=float).T
    _, sigma, vt = np.linalg.svd(c_b @ np.linalg.inv(c_a), full_matrices=False)
    c_half = vt.T @ np.diag(np.sqrt(np.maximum(sigma, 0.0))) @ vt @ c_a
    return c_half.T


def normalized_export_lattices(
    lattice_a: np.ndarray,
    lattice_b: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return endpoint lattices for exported matched POSCARs."""
    if mode == "raw":
        return np.asarray(lattice_a, dtype=float), np.asarray(lattice_b, dtype=float)
    if mode == "neb":
        aligned_a, aligned_b, _ = neb_aligned_lattices(lattice_a, lattice_b)
        return aligned_a, aligned_b
    if mode == "norot":
        return np.asarray(lattice_a, dtype=float), rotation_free_final_lattice(lattice_a, lattice_b)
    if mode == "medium":
        lattice = half_distorted_lattice(lattice_a, lattice_b)
        return lattice, lattice.copy()
    raise ValueError(f"Unknown export lattice mode: {mode}")


def write_poscar_direct(
    filename: Path,
    comment: str,
    lattice: np.ndarray,
    labels: np.ndarray,
    frac: np.ndarray,
) -> None:
    """Write a POSCAR without wrapping fractional coordinates into [0, 1)."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    frac = np.asarray(frac, dtype=float)
    species: List[str] = []
    counts: List[int] = []
    for label in labels.tolist():
        label = str(label)
        if species and species[-1] == label:
            counts[-1] += 1
        else:
            species.append(label)
            counts.append(1)
    with filename.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{comment}\n")
        fh.write("1.0\n")
        for row in np.asarray(lattice, dtype=float):
            fh.write(f"{row[0]:.12f} {row[1]:.12f} {row[2]:.12f}\n")
        fh.write(" ".join(species) + "\n")
        fh.write(" ".join(str(x) for x in counts) + "\n")
        fh.write("Direct\n")
        for row in frac:
            fh.write(f"{row[0]:.12f} {row[1]:.12f} {row[2]:.12f}\n")


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


def resolve_diag_max(
    requested: Optional[int],
    det_mult_max: int,
    d_a_base: int,
    d_b_base: int,
) -> Tuple[int, str, int]:
    """Resolve the HNF diagonal cap and return its provenance.

    The largest requested determinant is also the smallest diagonal cap that
    retains every HNF in the requested determinant range: a determinant-D HNF
    can have diagonal (D, 1, 1).
    """
    no_truncation_bound = max(1, int(det_mult_max)) * max(
        1, int(d_a_base), int(d_b_base)
    )
    if requested is None:
        return no_truncation_bound, "auto", no_truncation_bound
    return max(1, int(requested)), "manual", no_truncation_bound


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


def make_supercell_integer(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    labels: np.ndarray,
    transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a supercell from any orientation-preserving integer row basis.

    The normal HNF path remains fast.  For ``Q @ H`` bases, an exact row-HNF
    is used only to enumerate quotient translations; fractional coordinates are
    expressed in the *requested* ``Q @ H`` basis, not silently converted back
    to HNF.  This is required for a relative IMT Q to affect the match.
    """

    matrix = np.asarray(transform, dtype=int)
    if matrix.shape != (3, 3) or det_int(matrix) <= 0:
        raise ValueError("Supercell transform must be a 3x3 integer matrix with positive determinant.")
    if is_valid_hnf(matrix):
        return make_supercell_fast(lattice, frac_coords, labels, matrix)
    if row_hnf_3x3 is None:
        raise RuntimeError(
            "General integer-supercell construction requires snf_csm_invariant.py and sympy. "
            f"Original import error: {INTRINSIC_PERIOD_IMPORT_ERROR}"
        )

    quotient_hnf = np.asarray(row_hnf_3x3(matrix.tolist()), dtype=int)
    a, d, f = (int(quotient_hnf[0, 0]), int(quotient_hnf[1, 1]), int(quotient_hnf[2, 2]))
    det_matrix = abs(det_int(matrix))
    if a * d * f != det_matrix:
        raise ValueError("Row-HNF quotient enumeration has an inconsistent determinant.")

    inverse = np.linalg.inv(matrix.astype(float))
    base = np.asarray(frac_coords, dtype=float) @ inverse
    base -= np.floor(base)
    shifts_int = np.stack(
        np.meshgrid(np.arange(a), np.arange(d), np.arange(f), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    shifts = shifts_int @ inverse
    frac = (base[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    frac -= np.floor(frac)
    labels_rep = np.repeat(labels, det_matrix)
    return matrix @ lattice, labels_rep, frac


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


def matched_translation_offsets(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    lattice_a: np.ndarray,
    lattice_b: np.ndarray,
    image_shell: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best integer translations k so frac_b + k follows frac_a."""
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
    if offsets.size == 0:
        offsets = np.zeros((1, 3), dtype=float)
    diff = frac_b[:, None, :] + offsets[None, :, :] - frac_a[:, None, :]
    cart_a = diff @ lattice_a
    cart_b = diff @ lattice_b
    d2 = 0.5 * (np.sum(cart_a * cart_a, axis=2) + np.sum(cart_b * cart_b, axis=2))
    best = np.argmin(d2, axis=1)
    return offsets[best].round().astype(int), np.sqrt(np.maximum(d2[np.arange(len(frac_a)), best], 0.0))


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
    translations_b: List[List[int]] = []
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
        trans, pair_distances = matched_translation_offsets(
            frac_a[idx_a][rows],
            shifted_b[idx_b][cols],
            lattice_a,
            lattice_b,
            image_shell=image_shell,
        )
        order_a.extend(idx_a[rows].tolist())
        order_b.extend(idx_b[cols].tolist())
        translations_b.extend(trans.tolist())
        all_distances.extend(pair_distances.tolist())

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
        translations_b=np.array(translations_b, dtype=int),
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
    """Build supercells, then apply one common Q basis for both endpoints."""
    sc_lat_a, sc_labels_a, sc_frac_a = make_supercell_fast(
        problem.lattice_a, problem.frac_a, problem.labels_a, h_a
    )
    sc_lat_b, sc_labels_b, sc_frac_b = make_supercell_fast(
        problem.lattice_b, problem.frac_b, problem.labels_b, h_b
    )

    q = pair_reduction_q(sc_lat_a, sc_lat_b, enabled=problem.cfg.reduce_cell)
    q_a, q_b = q, q
    sc_lat_a, sc_frac_a = apply_basis_q(sc_lat_a, sc_frac_a, q_a)
    sc_lat_b, sc_frac_b = apply_basis_q(sc_lat_b, sc_frac_b, q_b)
    lattice_penalty = strain_penalty(sc_lat_a, sc_lat_b)
    shape_penalty = pair_shape_score(sc_lat_a, sc_lat_b)
    return q_a, q_b, sc_lat_a, sc_labels_a, sc_frac_a, sc_lat_b, sc_labels_b, sc_frac_b, lattice_penalty, shape_penalty


def prepared_imt_candidate_cells(
    problem: Problem,
    h_a: np.ndarray,
    h_b: np.ndarray,
    relative_q_row: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build an IMT candidate whose target basis is Q_relative @ H_B @ L_B.

    The relative Q is mechanism data.  It is intentionally distinct from the
    common display Q selected later for both endpoints.
    """

    relative_q = np.asarray(relative_q_row, dtype=int)
    if relative_q.shape != (3, 3) or det_int(relative_q) != 1:
        raise ValueError("relative_q_row must be an orientation-preserving unimodular 3x3 matrix.")
    sc_lat_a, sc_labels_a, sc_frac_a = make_supercell_fast(
        problem.lattice_a, problem.frac_a, problem.labels_a, h_a
    )
    sc_lat_b, sc_labels_b, sc_frac_b = make_supercell_integer(
        problem.lattice_b,
        problem.frac_b,
        problem.labels_b,
        relative_q @ h_b,
    )

    q_display = pair_reduction_q(sc_lat_a, sc_lat_b, enabled=problem.cfg.reduce_cell)
    q_a, q_b = q_display, q_display
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

    bad = EvalResult(
        1e30,
        1e15,
        1e30,
        1e30,
        1e30,
        1e30,
        (0.0, 0.0, 0.0),
        d_a,
        d_b,
        identity_q,
        identity_q,
    )

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

    normalized_rmsd2, _, _, _, fitness = fitness_terms(
        match.rmsd2,
        problem.rmsd_scale_ang,
        geom,
        shape,
        problem.cfg.weight_rmsd,
        problem.cfg.weight_geom,
        problem.cfg.weight_shape,
    )
    result = EvalResult(
        fitness=float(fitness),
        rmsd=float(match.rmsd),
        rmsd2=float(match.rmsd2),
        rmsd2_normalized=normalized_rmsd2,
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


def evaluate_imt_candidate(
    problem: Problem,
    ind: Individual,
    relative_q_row: np.ndarray,
    with_match: bool = False,
) -> Tuple[EvalResult, Optional[MatchResult]]:
    """Evaluate one bounded IMT candidate without conflating Q with display cleanup."""

    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    relative_q = np.asarray(relative_q_row, dtype=int)
    q_flat = hnf_flat(relative_q)
    cache_key = (ind, q_flat)
    if not with_match and cache_key in problem.fitness_cache:
        return problem.fitness_cache[cache_key], None

    d_a = abs(det_int(h_a))
    d_b = abs(det_int(h_b))
    identity_q = tuple(int(x) for x in np.eye(3, dtype=int).reshape(-1).tolist())
    bad = EvalResult(
        1e30,
        1e15,
        1e30,
        1e30,
        1e30,
        1e30,
        (0.0, 0.0, 0.0),
        d_a,
        d_b,
        identity_q,
        identity_q,
    )
    if not is_valid_hnf(h_a, diag_max=problem.cfg.diag_max):
        return bad, None
    if not is_valid_hnf(h_b, diag_max=problem.cfg.diag_max):
        return bad, None
    if relative_q.shape != (3, 3) or det_int(relative_q) != 1:
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
        ) = prepared_imt_candidate_cells(problem, h_a, h_b, relative_q)
    except Exception:
        return bad, None
    if sc_frac_a.shape[0] != sc_frac_b.shape[0]:
        return bad, None
    for species in problem.species:
        if np.sum(sc_labels_a == species) != np.sum(sc_labels_b == species):
            return bad, None

    match = best_hungarian_match(
        problem, sc_lat_a, sc_labels_a, sc_frac_a, sc_lat_b, sc_labels_b, sc_frac_b
    )
    if match is None:
        return bad, None
    normalized_rmsd2, _, _, _, fitness = fitness_terms(
        match.rmsd2,
        problem.rmsd_scale_ang,
        geom,
        shape,
        problem.cfg.weight_rmsd,
        problem.cfg.weight_geom,
        problem.cfg.weight_shape,
    )
    result = EvalResult(
        fitness=float(fitness),
        rmsd=float(match.rmsd),
        rmsd2=float(match.rmsd2),
        rmsd2_normalized=normalized_rmsd2,
        geom=float(geom),
        shape=float(shape),
        shift=tuple(float(x) for x in match.shift),
        det_a=int(d_a),
        det_b=int(d_b),
        q_a=tuple(int(x) for x in q_a.reshape(-1).tolist()),
        q_b=tuple(int(x) for x in q_b.reshape(-1).tolist()),
    )
    problem.fitness_cache[cache_key] = result
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
    report_step = max(1, n // 10)
    next_report = report_step
    started = time.perf_counter()
    last_report = started

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
        now = time.perf_counter()
        if len(pop) >= next_report or now - last_report >= 60.0:
            print(
                f"[GA Init] sampled={len(pop)}/{n} attempts={attempts} "
                f"elapsed={format_duration(now - started)}",
                flush=True,
            )
            while next_report <= len(pop):
                next_report += report_step
            last_report = now

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
        accepted = False
        for _ in range(6):
            key = eq_key_pair(problem, child)
            if key not in problem.eq_cache:
                problem.eq_cache.add(key)
                accepted = True
                break
            child = mutate_individual(problem, child, force_matrix_resample=True)
        if accepted:
            pop[idx] = child

    rescored = [(evaluate(problem, ind)[0].fitness, i) for i, ind in enumerate(pop)]
    rescored.sort(key=lambda item: item[0], reverse=True)
    active_keys = {eq_key_pair(problem, ind) for ind in pop}
    for elite, (_, idx) in zip(elites, rescored[: len(elites)]):
        elite_key = eq_key_pair(problem, elite)
        if elite_key in active_keys:
            continue
        replaced_key = eq_key_pair(problem, pop[idx])
        active_keys.discard(replaced_key)
        pop[idx] = elite
        active_keys.add(elite_key)
    return pop


def top_k(problem: Problem, pop: Sequence[Individual], k: int) -> List[Tuple[EvalResult, Individual]]:
    scored = [(evaluate(problem, ind)[0], ind) for ind in pop]
    scored.sort(key=lambda item: item[0].fitness)
    selected: List[Tuple[EvalResult, Individual]] = []
    seen_keys = set()
    key_cache: Dict[Individual, object] = {}
    for result, ind in scored:
        if ind not in key_cache:
            key_cache[ind] = (
                ind
                if problem.cfg.dedupe_mode == "none"
                else eq_key_pair(problem, ind)
            )
        key = key_cache[ind]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append((result, ind))
        if len(selected) >= k:
            break
    return selected


def count_exhaustive_pairs(problem: Problem) -> int:
    total = 0
    for t in range(problem.cfg.det_mult_min, problem.cfg.det_mult_max + 1):
        n_a = count_hnfs_with_det(t * problem.d_a_base, problem.cfg.diag_max)
        n_b = count_hnfs_with_det(t * problem.d_b_base, problem.cfg.diag_max)
        total += n_a * n_b
    return total


def exhaustive_pairs_for_t(problem: Problem, t: int) -> int:
    n_a = count_hnfs_with_det(t * problem.d_a_base, problem.cfg.diag_max)
    n_b = count_hnfs_with_det(t * problem.d_b_base, problem.cfg.diag_max)
    return n_a * n_b


def eval_result_to_dict(result: EvalResult) -> Dict[str, object]:
    return {
        "fitness": result.fitness,
        "rmsd": result.rmsd,
        "rmsd2": result.rmsd2,
        "rmsd2_normalized": result.rmsd2_normalized,
        "geom": result.geom,
        "shape": result.shape,
        "shift": list(result.shift),
        "det_a": result.det_a,
        "det_b": result.det_b,
        "q_a": list(result.q_a),
        "q_b": list(result.q_b),
    }


def eval_result_from_dict(payload: Dict[str, object]) -> EvalResult:
    return EvalResult(
        fitness=float(payload["fitness"]),
        rmsd=float(payload["rmsd"]),
        rmsd2=float(payload["rmsd2"]),
        rmsd2_normalized=float(payload["rmsd2_normalized"]),
        geom=float(payload["geom"]),
        shape=float(payload["shape"]),
        shift=tuple(float(x) for x in payload["shift"]),
        det_a=int(payload["det_a"]),
        det_b=int(payload["det_b"]),
        q_a=tuple(int(x) for x in payload["q_a"]),
        q_b=tuple(int(x) for x in payload["q_b"]),
    )


def ranked_candidate_to_dict(candidate: Tuple[EvalResult, Individual]) -> Dict[str, object]:
    result, ind = candidate
    return {
        "H1_hnf_flat": list(ind[0]),
        "H2_hnf_flat": list(ind[1]),
        "result": eval_result_to_dict(result),
    }


def ranked_candidate_from_dict(payload: Dict[str, object]) -> Tuple[EvalResult, Individual]:
    result = eval_result_from_dict(dict(payload["result"]))
    ind = (
        tuple(int(x) for x in payload["H1_hnf_flat"]),
        tuple(int(x) for x in payload["H2_hnf_flat"]),
    )
    return result, ind


def update_unique_top(
    problem: Problem,
    pool: Dict[object, Tuple[EvalResult, Individual]],
    candidate: Tuple[EvalResult, Individual],
    k: int,
) -> None:
    """Maintain the exact best k semantic classes seen so far."""
    result, ind = candidate
    key = ind if problem.cfg.dedupe_mode == "none" else eq_key_pair(problem, ind)
    previous = pool.get(key)
    if previous is not None:
        if result.fitness < previous[0].fitness:
            pool[key] = candidate
        return
    if len(pool) < k:
        pool[key] = candidate
        return
    worst_key = max(pool, key=lambda item: pool[item][0].fitness)
    if result.fitness < pool[worst_key][0].fitness:
        del pool[worst_key]
        pool[key] = candidate


def sorted_unique_top(
    pool: Dict[object, Tuple[EvalResult, Individual]],
) -> List[Tuple[EvalResult, Individual]]:
    return sorted(pool.values(), key=lambda item: item[0].fitness)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumeration_signature(problem: Problem) -> Dict[str, object]:
    cfg = problem.cfg
    return {
        "schema_version": 1,
        "file_a": str(cfg.file_a.resolve()),
        "file_b": str(cfg.file_b.resolve()),
        "file_a_sha256": file_sha256(cfg.file_a),
        "file_b_sha256": file_sha256(cfg.file_b),
        "z_a": problem.z_a,
        "z_b": problem.z_b,
        "d_a_base": problem.d_a_base,
        "d_b_base": problem.d_b_base,
        "det_mult_min": cfg.det_mult_min,
        "det_mult_max": cfg.det_mult_max,
        "diag_max": cfg.diag_max,
        "top_k": cfg.top_k,
        "weight_rmsd": cfg.weight_rmsd,
        "weight_geom": cfg.weight_geom,
        "weight_shape": cfg.weight_shape,
        "rmsd_scale_ang": problem.rmsd_scale_ang,
        "max_origin_shifts": cfg.max_origin_shifts,
        "reduce_cell": cfg.reduce_cell,
        "image_shell": cfg.image_shell,
        "dedupe_mode": cfg.dedupe_mode,
        "symprec": cfg.symprec,
    }


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def run_enumeration(
    problem: Problem,
    timing: Optional[Dict[str, object]] = None,
) -> List[Tuple[EvalResult, Individual]]:
    cfg = problem.cfg
    total = count_exhaustive_pairs(problem)
    if total > cfg.exhaustive_limit:
        raise RuntimeError(
            f"Exhaustive space has {total} pairs, above --exhaustive-limit "
            f"{cfg.exhaustive_limit}. Reduce --det-mult-max/--diag-max "
            "or use --mode ga."
        )
    if cfg.enumerate_stop_after_t is not None and not (
        cfg.det_mult_min <= cfg.enumerate_stop_after_t <= cfg.det_mult_max
    ):
        raise ValueError(
            "--enumerate-stop-after-t must lie inside the requested "
            "--det-mult-min/--det-mult-max range."
        )

    signature = enumeration_signature(problem)
    checkpoint_path = cfg.output_dir / "enumeration_progress.json"
    stages_dir = cfg.output_dir / "enumeration_stages"
    global_pool: Dict[object, Tuple[EvalResult, Individual]] = {}
    completed_t: List[int] = []
    stage_records: List[Dict[str, object]] = []
    evaluated_total = 0
    resumed = False

    if cfg.enumerate_resume:
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Cannot resume enumeration: checkpoint not found at {checkpoint_path}."
            )
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        if checkpoint.get("signature") != signature:
            raise RuntimeError(
                "Cannot resume enumeration: the input files or search/scoring settings "
                "do not match the checkpoint."
            )
        completed_t = [int(value) for value in checkpoint.get("completed_t", [])]
        stage_records = list(checkpoint.get("stages", []))
        evaluated_total = int(checkpoint.get("evaluated_pairs", 0))
        for payload in checkpoint.get("global_top", []):
            update_unique_top(
                problem,
                global_pool,
                ranked_candidate_from_dict(payload),
                cfg.top_k,
            )
        resumed = True
        print(
            f"[Enumerate] Resuming after completed t={completed_t or 'none'}; "
            f"retained global top={len(global_pool)}",
            flush=True,
        )

    expected_prefix = list(
        range(cfg.det_mult_min, cfg.det_mult_min + len(completed_t))
    )
    if completed_t != expected_prefix:
        raise RuntimeError(
            "Enumeration checkpoint has a non-contiguous completed_t sequence and "
            "cannot be resumed safely."
        )

    status = "running"
    active_t: Optional[int] = None

    def checkpoint_payload() -> Dict[str, object]:
        return {
            "schema_version": 1,
            "signature": signature,
            "status": status,
            "completed_t": completed_t,
            "next_t": (
                completed_t[-1] + 1 if completed_t else cfg.det_mult_min
            ),
            "active_t": active_t,
            "candidate_pairs_total": total,
            "evaluated_pairs": evaluated_total,
            "stages": stage_records,
            "global_top": [
                ranked_candidate_to_dict(item) for item in sorted_unique_top(global_pool)
            ],
            "updated_at": local_timestamp(),
        }

    if not cfg.enumerate_resume:
        write_json_atomic(checkpoint_path, checkpoint_payload())

    try:
        start_t = completed_t[-1] + 1 if completed_t else cfg.det_mult_min
        for t in range(start_t, cfg.det_mult_max + 1):
            active_t = t
            stage_total = exhaustive_pairs_for_t(problem, t)
            stage_started = time.perf_counter()
            last_progress = stage_started
            evaluated_stage = 0
            stage_pool: Dict[object, Tuple[EvalResult, Individual]] = {}
            hnfs_b = tuple(
                hnf_matrices_with_det(t * problem.d_b_base, cfg.diag_max)
            )
            print(
                f"[Enumerate] t={t} started: det(H1)={t * problem.d_a_base}, "
                f"det(H2)={t * problem.d_b_base}, pairs={stage_total}",
                flush=True,
            )

            for h_a in hnf_matrices_with_det(t * problem.d_a_base, cfg.diag_max):
                h_a_flat = hnf_flat(h_a)
                for h_b in hnfs_b:
                    ind = (h_a_flat, hnf_flat(h_b))
                    result = evaluate(problem, ind)[0]
                    update_unique_top(problem, stage_pool, (result, ind), cfg.top_k)
                    problem.fitness_cache.pop(ind, None)
                    evaluated_stage += 1
                    now = time.perf_counter()
                    if now - last_progress >= 60.0:
                        elapsed = now - stage_started
                        rate = evaluated_stage / elapsed if elapsed > 0.0 else 0.0
                        print(
                            f"[Enumerate] t={t} progress={evaluated_stage}/{stage_total} "
                            f"({100.0 * evaluated_stage / max(1, stage_total):.1f}%) "
                            f"rate={rate:.2f} pair/s elapsed={format_duration(elapsed)}",
                            flush=True,
                        )
                        last_progress = now

            stage_top = sorted_unique_top(stage_pool)
            for candidate in stage_top:
                update_unique_top(problem, global_pool, candidate, cfg.top_k)
            stage_elapsed = time.perf_counter() - stage_started
            evaluated_total += evaluated_stage
            completed_t.append(t)
            stage_record = {
                "t": t,
                "det_H1": t * problem.d_a_base,
                "det_H2": t * problem.d_b_base,
                "candidate_pairs": stage_total,
                "evaluated_pairs": evaluated_stage,
                "elapsed_seconds": stage_elapsed,
                "elapsed_hms": format_duration(stage_elapsed),
                "stage_top": [ranked_candidate_to_dict(item) for item in stage_top],
                "global_top": [
                    ranked_candidate_to_dict(item)
                    for item in sorted_unique_top(global_pool)
                ],
            }
            stage_records.append(stage_record)
            active_t = None
            status = (
                "complete" if t == cfg.det_mult_max else "stage_complete"
            )
            write_json_atomic(
                stages_dir / f"t_{t:04d}.json",
                {
                    "schema_version": 1,
                    "signature": signature,
                    "stage": stage_record,
                    "updated_at": local_timestamp(),
                },
            )
            write_json_atomic(checkpoint_path, checkpoint_payload())
            best = stage_top[0][0].fitness if stage_top else float("inf")
            global_best = (
                sorted_unique_top(global_pool)[0][0].fitness
                if global_pool
                else float("inf")
            )
            print(
                f"[Enumerate] t={t} complete: evaluated={evaluated_stage}, "
                f"stage_best={best:.8e}, global_best={global_best:.8e}, "
                f"elapsed={format_duration(stage_elapsed)}",
                flush=True,
            )
            print(
                f"[Enumerate] Checkpoint written: {checkpoint_path}",
                flush=True,
            )

            if cfg.enumerate_stop_after_t == t and t < cfg.det_mult_max:
                status = "stopped_after_stage"
                write_json_atomic(checkpoint_path, checkpoint_payload())
                print(
                    f"[Enumerate] Normal stop requested after t={t}. "
                    "Use --enumerate-resume to continue.",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        status = "interrupted_during_stage"
        write_json_atomic(checkpoint_path, checkpoint_payload())
        print(
            "\n[Enumerate] Interrupted. Completed stages remain resumable; "
            "the active incomplete t stage will be recomputed.",
            flush=True,
        )
        raise

    if completed_t and completed_t[-1] == cfg.det_mult_max:
        status = "complete"
    write_json_atomic(checkpoint_path, checkpoint_payload())
    if timing is not None:
        timing.update(
            {
                "mode": "enumerate",
                "status": status,
                "resumed": resumed,
                "completed_t": list(completed_t),
                "evaluated_pairs": evaluated_total,
                "candidate_pairs_total": total,
                "stages": stage_records,
                "checkpoint": str(checkpoint_path),
            }
        )
    return sorted_unique_top(global_pool)


def export_candidate(
    problem: Problem,
    rank: int,
    ind: Individual,
    result: EvalResult,
    relative_q_row: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    h_a = hnf_from_flat(ind[0])
    h_b = hnf_from_flat(ind[1])
    relative_q = (
        np.eye(3, dtype=int)
        if relative_q_row is None
        else np.asarray(relative_q_row, dtype=int)
    )
    if relative_q.shape != (3, 3) or det_int(relative_q) != 1:
        raise ValueError("relative_q_row must be an orientation-preserving unimodular matrix.")
    if np.array_equal(relative_q, np.eye(3, dtype=int)):
        q_a, q_b, lat_a, labels_a, frac_a, lat_b, labels_b, frac_b, _, _ = prepared_candidate_cells(problem, h_a, h_b)
        _, match = evaluate(problem, ind, with_match=True)
    else:
        q_a, q_b, lat_a, labels_a, frac_a, lat_b, labels_b, frac_b, _, _ = prepared_imt_candidate_cells(
            problem,
            h_a,
            h_b,
            relative_q,
        )
        _, match = evaluate_imt_candidate(problem, ind, relative_q, with_match=True)
    if match is None:
        raise RuntimeError("Cannot export candidate without a valid match.")

    labels_a_ord = labels_a[match.order_a]
    labels_b_ord = labels_b[match.order_b]
    frac_a_ord = frac_a[match.order_a]
    if not np.array_equal(labels_a_ord, labels_b_ord):
        raise RuntimeError("Matched species order is inconsistent during export.")
    frac_b_shifted = frac_b + match.shift
    frac_b_shifted -= np.floor(frac_b_shifted)
    frac_b_ord = frac_b_shifted[match.order_b] + match.translations_b
    export_lat_a, export_lat_b = normalized_export_lattices(
        lat_a,
        lat_b,
        problem.cfg.export_lattice_mode,
    )
    initial_axis_rotation = (
        initial_to_final_ab_rotation(lat_a, lat_b)
        if problem.cfg.export_lattice_mode == "neb"
        else np.eye(3, dtype=float)
    )
    aligned_a_frame = ab_cartesian_frame(export_lat_a)
    final_frame = ab_cartesian_frame(export_lat_b)
    a_axis_alignment_cosine = float(np.dot(aligned_a_frame[0], final_frame[0]))
    b_axis_alignment_cosine = float(
        np.dot(
            export_lat_a[1] / np.linalg.norm(export_lat_a[1]),
            export_lat_b[1] / np.linalg.norm(export_lat_b[1]),
        )
    )
    ab_plane_alignment_cosine = float(np.dot(aligned_a_frame[2], final_frame[2]))

    problem.cfg.output_dir.mkdir(parents=True, exist_ok=True)
    poscar_initial = problem.cfg.output_dir / f"{rank:02d}-POSCAR-i.vasp"
    poscar_final = problem.cfg.output_dir / f"{rank:02d}-POSCAR-f.vasp"
    write_poscar_direct(
        poscar_initial,
        f"SNF match {rank:02d} initial ({problem.cfg.export_lattice_mode})",
        export_lat_a,
        labels_a_ord,
        frac_a_ord,
    )
    write_poscar_direct(
        poscar_final,
        f"SNF match {rank:02d} final ({problem.cfg.export_lattice_mode})",
        export_lat_b,
        labels_b_ord,
        frac_b_ord,
    )

    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H1_hnf.txt", h_a, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H2_hnf.txt", h_b, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-Q_relative.txt", relative_q, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-Q_A.txt", q_a, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-Q_B.txt", q_b, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H1_display_QH.txt", q_a @ h_a, fmt="%d")
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-H2_display_QH.txt", q_b @ relative_q @ h_b, fmt="%d")
    np.savetxt(
        problem.cfg.output_dir / f"{rank:02d}-R_initial_to_final_axes.txt",
        initial_axis_rotation,
        fmt="%.12f",
    )
    np.savetxt(problem.cfg.output_dir / f"{rank:02d}-shuffle-distances.txt", match.distances, fmt="%.10f")

    normalized_rmsd2, atom_term, geom_term, shape_term, checked_fitness = fitness_terms(
        result.rmsd2,
        problem.rmsd_scale_ang,
        result.geom,
        result.shape,
        problem.cfg.weight_rmsd,
        problem.cfg.weight_geom,
        problem.cfg.weight_shape,
    )
    if not np.isclose(checked_fitness, result.fitness, rtol=1e-12, atol=1e-14):
        raise RuntimeError("Exported fitness terms do not reproduce the evaluated fitness.")

    record = add_snf_tags({
        "rank": rank,
        "poscar_initial": str(poscar_initial),
        "poscar_final": str(poscar_final),
        "fitness": result.fitness,
        "rmsd_ang": result.rmsd,
        "rmsd2_ang2": result.rmsd2,
        "rmsd_scale_ang": problem.rmsd_scale_ang,
        "rmsd2_normalized": normalized_rmsd2,
        "weighted_rmsd_contribution": atom_term,
        "weighted_lattice_contribution": geom_term,
        "weighted_shape_contribution": shape_term,
        "lattice_penalty": result.geom,
        "gram_penalty": result.geom,
        "shape_penalty": result.shape,
        "det_H1": result.det_a,
        "det_H2": result.det_b,
        "origin_shift": [float(x) for x in match.shift],
        "B_integer_translations": match.translations_b.tolist(),
        "export_lattice_mode": problem.cfg.export_lattice_mode,
        "initial_to_final_axes_rotation": initial_axis_rotation.tolist(),
        "a_axis_alignment_cosine": a_axis_alignment_cosine,
        "b_axis_alignment_cosine": b_axis_alignment_cosine,
        "ab_plane_alignment_cosine": ab_plane_alignment_cosine,
        "preserve_unwrapped_fractional_coordinates": True,
        "Q_A": q_a.tolist(),
        "Q_B": q_b.tolist(),
        "H1_hnf": h_a.tolist(),
        "H2_hnf": h_b.tolist(),
        "H1_display_QH": (q_a @ h_a).tolist(),
        "H2_display_QH": (q_b @ relative_q @ h_b).tolist(),
    })
    # Q_A/Q_B are a common display reduction.  The actual current relative
    # IMT coupling remains Q=I and is recorded separately by the audit.
    record["display_Q_A"] = q_a.tolist()
    record["display_Q_B"] = q_b.tolist()
    return add_intrinsic_period_tags(
        record,
        q_a @ h_a,
        q_b @ relative_q @ h_b,
        frac_a_ord,
        frac_b_ord,
        labels_a_ord,
        labels_b_ord,
        relative_q_row=relative_q,
        enabled=problem.cfg.intrinsic_period_audit,
        max_points=problem.cfg.intrinsic_period_max_points,
    )


def build_snf_classes(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    classes: Dict[str, Dict[str, object]] = {}
    for record in records:
        if "snf_pair_key" not in record:
            add_snf_tags(record)
        key = str(record["snf_pair_key"])
        entry = classes.setdefault(
            key,
            {
                "snf_pair_key": key,
                "snf_H1": record["snf_H1"],
                "snf_H2": record["snf_H2"],
                "count": 0,
                "ranks": [],
                "best_rank": None,
                "best_fitness": None,
                "best_rmsd_ang": None,
                "best_lattice_penalty": None,
                "best_shape_penalty": None,
                "det_H1": record.get("det_H1"),
                "det_H2": record.get("det_H2"),
            },
        )
        entry["count"] = int(entry["count"]) + 1
        ranks = entry["ranks"]
        assert isinstance(ranks, list)
        ranks.append(record.get("rank"))
        fitness = float(record.get("fitness", float("inf")))
        best_fitness = entry["best_fitness"]
        if best_fitness is None or fitness < float(best_fitness):
            entry["best_rank"] = record.get("rank")
            entry["best_fitness"] = fitness
            entry["best_rmsd_ang"] = record.get("rmsd_ang")
            entry["best_lattice_penalty"] = record.get("lattice_penalty")
            entry["best_shape_penalty"] = record.get("shape_penalty")

    return sorted(
        classes.values(),
        key=lambda item: (
            float(item["best_fitness"]) if item["best_fitness"] is not None else float("inf"),
            str(item["snf_pair_key"]),
        ),
    )


def build_intrinsic_period_classes(
    records: Sequence[Dict[str, object]],
    class_key: str,
) -> List[Dict[str, object]]:
    """Group only candidate rows with an exact atom-period verification."""

    classes: Dict[str, Dict[str, object]] = {}
    for record in records:
        if not record.get("atomic_period_verified"):
            continue
        key_value = record.get(class_key)
        if key_value is None:
            continue
        key = str(key_value)
        entry = classes.setdefault(
            key,
            {
                "class_key": key,
                "atomic_snf_pair_key": record.get("atomic_snf_pair_key"),
                "atomic_coupled_imt_key": record.get("atomic_coupled_imt_key"),
                "count": 0,
                "ranks": [],
                "best_rank": None,
                "best_fitness": None,
                "best_rmsd_ang": None,
                "best_lattice_penalty": None,
                "best_shape_penalty": None,
                "min_atomic_reduction_factor": None,
                "max_atomic_reduction_factor": None,
            },
        )
        entry["count"] = int(entry["count"]) + 1
        ranks = entry["ranks"]
        assert isinstance(ranks, list)
        ranks.append(record.get("rank"))
        reduction_factor = int(record["atomic_reduction_factor"])
        minimum = entry["min_atomic_reduction_factor"]
        maximum = entry["max_atomic_reduction_factor"]
        entry["min_atomic_reduction_factor"] = (
            reduction_factor if minimum is None else min(int(minimum), reduction_factor)
        )
        entry["max_atomic_reduction_factor"] = (
            reduction_factor if maximum is None else max(int(maximum), reduction_factor)
        )
        fitness = float(record.get("fitness", float("inf")))
        best_fitness = entry["best_fitness"]
        if best_fitness is None or fitness < float(best_fitness):
            entry["best_rank"] = record.get("rank")
            entry["best_fitness"] = fitness
            entry["best_rmsd_ang"] = record.get("rmsd_ang")
            entry["best_lattice_penalty"] = record.get("lattice_penalty")
            entry["best_shape_penalty"] = record.get("shape_penalty")

    return sorted(
        classes.values(),
        key=lambda item: (
            float(item["best_fitness"]) if item["best_fitness"] is not None else float("inf"),
            str(item["class_key"]),
        ),
    )


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_timing_receipt(problem: Problem, timing: Dict[str, object]) -> None:
    out = problem.cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_timing.json").open("w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=2, ensure_ascii=False)

    summary_path = out / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        summary["timing"] = timing
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)


def save_summary(
    problem: Problem,
    records: List[Dict[str, object]],
    timing: Optional[Dict[str, object]] = None,
) -> None:
    out = problem.cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    snf_classes = build_snf_classes(records)
    atomic_snf_classes = build_intrinsic_period_classes(records, "atomic_snf_pair_key")
    atomic_imt_classes = build_intrinsic_period_classes(records, "atomic_coupled_imt_key")
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
                "timing": dict(timing or {}),
                "config": {
                    "mode": problem.cfg.mode,
                    "seed": problem.cfg.seed,
                    "ga_stop_after_generation": problem.cfg.ga_stop_after_generation,
                    "ga_resume": problem.cfg.ga_resume,
                    "det_mult_min": problem.cfg.det_mult_min,
                    "det_mult_max": problem.cfg.det_mult_max,
                    "enumerate_stop_after_t": problem.cfg.enumerate_stop_after_t,
                    "enumerate_resume": problem.cfg.enumerate_resume,
                    "diag_max": problem.cfg.diag_max,
                    "diag_max_source": problem.diag_max_source,
                    "diag_max_no_truncation_bound": problem.diag_max_no_truncation_bound,
                    "weight_rmsd": problem.cfg.weight_rmsd,
                    "weight_geom": problem.cfg.weight_geom,
                    "weight_shape": problem.cfg.weight_shape,
                    "rmsd_scale_ang": problem.rmsd_scale_ang,
                    "rmsd_scale_source": problem.rmsd_scale_source,
                    "mean_nn_a_ang": problem.mean_nn_a_ang,
                    "mean_nn_b_ang": problem.mean_nn_b_ang,
                    "fitness_formula": (
                        "weight_rmsd * (rmsd_ang / rmsd_scale_ang)^2 + "
                        "weight_geom * lattice_penalty + weight_shape * shape_penalty"
                    ),
                    "max_origin_shifts": problem.cfg.max_origin_shifts,
                    "reduce_cell": problem.cfg.reduce_cell,
                    "image_shell": problem.cfg.image_shell,
                    "dedupe_mode": problem.cfg.dedupe_mode,
                    "symprec": problem.cfg.symprec,
                    "export_lattice_mode": problem.cfg.export_lattice_mode,
                    "neb_axis_alignment": (
                        "keep final Cartesian frame; rigidly rotate initial a axis and "
                        "a-b plane into the final endpoint frame"
                        if problem.cfg.export_lattice_mode == "neb"
                        else "disabled"
                    ),
                    "intrinsic_period_audit": problem.cfg.intrinsic_period_audit,
                    "intrinsic_period_max_points": problem.cfg.intrinsic_period_max_points,
                },
                "candidates": records,
                "raw_hnf_snf_classes": snf_classes,
                "atomic_period_snf_classes": atomic_snf_classes,
                "atomic_period_imt_classes": atomic_imt_classes,
                "snf_classes": snf_classes,
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
        "rmsd_scale_ang",
        "rmsd2_normalized",
        "weighted_rmsd_contribution",
        "weighted_lattice_contribution",
        "weighted_shape_contribution",
        "lattice_penalty",
        "shape_penalty",
        "det_H1",
        "det_H2",
        "poscar_initial",
        "poscar_final",
        "H1_hnf",
        "H2_hnf",
        "H1_display_QH",
        "H2_display_QH",
        "snf_H1",
        "snf_H2",
        "snf_pair_key",
        "raw_supercell_snf_pair_key",
        "relative_Q_row",
        "intrinsic_period_exact",
        "atomic_period_verified",
        "intrinsic_period_status",
        "intrinsic_period_error",
        "slm_and_atomic_period_agree",
        "slm_H1_hnf",
        "slm_H2_hnf",
        "slm_relative_Q_row",
        "slm_snf_H1",
        "slm_snf_H2",
        "slm_snf_pair_key",
        "slm_coupled_imt_key",
        "slm_source_index",
        "slm_target_index",
        "slm_reduction_factor",
        "atomic_H1_hnf",
        "atomic_H2_hnf",
        "atomic_relative_Q_row",
        "atomic_snf_H1",
        "atomic_snf_H2",
        "atomic_snf_pair_key",
        "atomic_coupled_imt_key",
        "atomic_source_index",
        "atomic_target_index",
        "atomic_reduction_factor",
        "origin_shift",
        "export_lattice_mode",
        "initial_to_final_axes_rotation",
        "a_axis_alignment_cosine",
        "b_axis_alignment_cosine",
        "ab_plane_alignment_cosine",
        "preserve_unwrapped_fractional_coordinates",
    ]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fieldnames}
            row["origin_shift"] = json.dumps(row["origin_shift"])
            for key in (
                "H1_hnf",
                "H2_hnf",
                "H1_display_QH",
                "H2_display_QH",
                "relative_Q_row",
                "initial_to_final_axes_rotation",
                "slm_H1_hnf",
                "slm_H2_hnf",
                "slm_relative_Q_row",
                "atomic_H1_hnf",
                "atomic_H2_hnf",
                "atomic_relative_Q_row",
            ):
                if row[key] is not None:
                    row[key] = json.dumps(row[key])
            for key in (
                "snf_H1",
                "snf_H2",
                "slm_snf_H1",
                "slm_snf_H2",
                "atomic_snf_H1",
                "atomic_snf_H2",
            ):
                if row[key] is not None:
                    row[key] = snf_key(row[key])
            writer.writerow(row)

    class_fieldnames = [
        "snf_pair_key",
        "snf_H1",
        "snf_H2",
        "count",
        "ranks",
        "best_rank",
        "best_fitness",
        "best_rmsd_ang",
        "best_lattice_penalty",
        "best_shape_penalty",
        "det_H1",
        "det_H2",
    ]
    with (out / "snf_classes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=class_fieldnames)
        writer.writeheader()
        for entry in snf_classes:
            row = {key: entry.get(key) for key in class_fieldnames}
            row["snf_H1"] = snf_key(row["snf_H1"])
            row["snf_H2"] = snf_key(row["snf_H2"])
            row["ranks"] = json.dumps(row["ranks"])
            writer.writerow(row)

    intrinsic_class_fieldnames = [
        "class_key",
        "atomic_snf_pair_key",
        "atomic_coupled_imt_key",
        "count",
        "ranks",
        "best_rank",
        "best_fitness",
        "best_rmsd_ang",
        "best_lattice_penalty",
        "best_shape_penalty",
        "min_atomic_reduction_factor",
        "max_atomic_reduction_factor",
    ]
    for filename, entries in (
        ("atomic_period_snf_classes.csv", atomic_snf_classes),
        ("atomic_period_imt_classes.csv", atomic_imt_classes),
    ):
        with (out / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=intrinsic_class_fieldnames)
            writer.writeheader()
            for entry in entries:
                row = {key: entry.get(key) for key in intrinsic_class_fieldnames}
                row["ranks"] = json.dumps(row["ranks"])
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
    parser.add_argument(
        "--ga-stop-after-generation",
        default=None,
        type=int,
        help=(
            "In GA mode, checkpoint and stop normally after this generation. "
            "Use 0 to stop after initial-population scoring."
        ),
    )
    parser.add_argument(
        "--ga-resume",
        action="store_true",
        help=(
            "Resume GA mode from output-dir/ga_progress.json. Inputs and all "
            "GA/search/scoring settings must match the checkpoint."
        ),
    )
    parser.add_argument("--substitute-number", default=192, type=int)
    parser.add_argument("--elite-keep", default=32, type=int)
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--det-mult-min", default=1, type=int)
    parser.add_argument("--det-mult-max", default=4, type=int)
    parser.add_argument(
        "--diag-max",
        default=None,
        type=int,
        help=(
            "HNF diagonal cap. By default it is resolved after reading both cells as "
            "det_mult_max * max(d_a_base, d_b_base), retaining every HNF in the "
            "requested determinant range. An explicit value is a manual override."
        ),
    )
    parser.add_argument("--weight-geom", default=1.0, type=float)
    parser.add_argument(
        "--weight-rmsd",
        default=1.0,
        type=float,
        help="Weight for the dimensionless squared distance (RMSD / d0)^2.",
    )
    parser.add_argument(
        "--rmsd-scale-ang",
        default=None,
        type=float,
        help=(
            "RMSD normalization length d0 in Angstrom. By default, use the average "
            "of the two endpoint mean nearest-neighbor distances."
        ),
    )
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
    parser.add_argument(
        "--enumerate-stop-after-t",
        default=None,
        type=int,
        help=(
            "In enumerate mode, finish and checkpoint this t stage, export the "
            "current global top candidates, and then stop normally."
        ),
    )
    parser.add_argument(
        "--enumerate-resume",
        action="store_true",
        help=(
            "Resume enumerate mode from output-dir/enumeration_progress.json. "
            "Inputs and all search/scoring settings must match the checkpoint."
        ),
    )
    parser.add_argument(
        "--skip-intrinsic-period-audit",
        action="store_true",
        help=(
            "Do not exact-audit the exported atom correspondence for reducible "
            "lattice/atomic periods. This disables intrinsic mechanism labels."
        ),
    )
    parser.add_argument(
        "--intrinsic-period-max-points",
        default=250000,
        type=int,
        help="Hard cap for exact residue/coset enumeration in the intrinsic-period audit.",
    )
    parser.add_argument(
        "--export-lattice-mode",
        choices=("neb", "norot", "medium", "raw"),
        default="neb",
        help=(
            "POSCAR lattice normalization: neb keeps the final Cartesian frame and "
            "rotates the initial a axis/a-b plane into it; norot removes final rigid rotation, "
            "medium writes both endpoints in the half-distorted common cell, "
            "raw keeps the matched endpoint lattices."
        ),
    )
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
        diag_max=None if args.diag_max is None else max(1, args.diag_max),
        weight_geom=float(args.weight_geom),
        weight_rmsd=float(args.weight_rmsd),
        rmsd_scale_ang=None if args.rmsd_scale_ang is None else float(args.rmsd_scale_ang),
        weight_shape=float(args.weight_shape),
        max_origin_shifts=max(1, args.max_origin_shifts),
        reduce_cell=not args.no_reduce_cell,
        image_shell=max(0, args.image_shell),
        dedupe_mode=args.dedupe_mode,
        symprec=max(1e-12, float(args.symprec)),
        exhaustive_limit=max(1, args.exhaustive_limit),
        export_lattice_mode=args.export_lattice_mode,
        intrinsic_period_audit=not args.skip_intrinsic_period_audit,
        intrinsic_period_max_points=max(1, args.intrinsic_period_max_points),
        enumerate_stop_after_t=args.enumerate_stop_after_t,
        enumerate_resume=bool(args.enumerate_resume),
        ga_stop_after_generation=args.ga_stop_after_generation,
        ga_resume=bool(args.ga_resume),
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
    diag_max, diag_max_source, diag_max_no_truncation_bound = resolve_diag_max(
        cfg.diag_max,
        cfg.det_mult_max,
        d_a_base,
        d_b_base,
    )
    cfg.diag_max = diag_max
    rmsd_scale_ang, rmsd_scale_source, mean_nn_a_ang, mean_nn_b_ang = resolve_rmsd_scale(
        cfg.rmsd_scale_ang,
        struct_a,
        struct_b,
    )
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
        diag_max_source=diag_max_source,
        diag_max_no_truncation_bound=diag_max_no_truncation_bound,
        rmsd_scale_ang=rmsd_scale_ang,
        rmsd_scale_source=rmsd_scale_source,
        mean_nn_a_ang=mean_nn_a_ang,
        mean_nn_b_ang=mean_nn_b_ang,
        fitness_cache={},
        eq_cache=set(),
    )


def json_tree(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [json_tree(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_tree(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def freeze_json_tree(value: object) -> object:
    if isinstance(value, list):
        return tuple(freeze_json_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: freeze_json_tree(item) for key, item in value.items()}
    return value


def individual_to_dict(ind: Individual) -> Dict[str, object]:
    return {"H1_hnf_flat": list(ind[0]), "H2_hnf_flat": list(ind[1])}


def individual_from_dict(payload: Dict[str, object]) -> Individual:
    return (
        tuple(int(x) for x in payload["H1_hnf_flat"]),
        tuple(int(x) for x in payload["H2_hnf_flat"]),
    )


def ga_signature(problem: Problem) -> Dict[str, object]:
    signature = enumeration_signature(problem)
    signature.update(
        {
            "checkpoint_kind": "ga",
            "algorithm_version": 1,
            "seed": problem.cfg.seed,
            "pop_size": problem.cfg.pop_size,
            "init_keep": problem.cfg.init_keep,
            "generations": problem.cfg.generations,
            "substitute_number": problem.cfg.substitute_number,
            "elite_keep": problem.cfg.elite_keep,
        }
    )
    return signature


def run_ga(
    problem: Problem,
    timing: Optional[Dict[str, object]] = None,
) -> List[Individual]:
    cfg = problem.cfg
    if cfg.ga_stop_after_generation is not None and not (
        0 <= cfg.ga_stop_after_generation <= cfg.generations
    ):
        raise ValueError(
            "--ga-stop-after-generation must be between 0 and --generations."
        )

    signature = ga_signature(problem)
    checkpoint_path = cfg.output_dir / "ga_progress.json"
    stages_dir = cfg.output_dir / "ga_stages"
    population: List[Individual]
    generation_seconds: List[float] = []
    stage_records: List[Dict[str, object]] = []
    completed_generation = 0
    init_elapsed = 0.0
    resumed = False
    status = "initializing"

    def checkpoint_payload() -> Dict[str, object]:
        return {
            "schema_version": 1,
            "signature": signature,
            "status": status,
            "completed_generation": completed_generation,
            "next_generation": completed_generation + 1,
            "population": [individual_to_dict(ind) for ind in population],
            "rng_state": json_tree(problem.rng.getstate()),
            "equivalence_cache": [json_tree(key) for key in problem.eq_cache],
            "fitness_cache": [
                {"key": json_tree(key), "result": eval_result_to_dict(result)}
                for key, result in problem.fitness_cache.items()
            ],
            "initial_population_seconds": init_elapsed,
            "generation_seconds": generation_seconds,
            "stages": stage_records,
            "updated_at": local_timestamp(),
        }

    if cfg.ga_resume:
        if not checkpoint_path.exists():
            raise RuntimeError(f"Cannot resume GA: checkpoint not found at {checkpoint_path}.")
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        if checkpoint.get("signature") != signature:
            raise RuntimeError(
                "Cannot resume GA: the input files or GA/search/scoring settings "
                "do not match the checkpoint."
            )
        population = [
            individual_from_dict(dict(payload))
            for payload in checkpoint.get("population", [])
        ]
        if not population:
            raise RuntimeError("Cannot resume GA: checkpoint population is empty.")
        rng_state = freeze_json_tree(checkpoint["rng_state"])
        if not isinstance(rng_state, tuple):
            raise RuntimeError("Cannot resume GA: invalid random-number state.")
        problem.rng.setstate(rng_state)
        problem.eq_cache = {
            freeze_json_tree(key) for key in checkpoint.get("equivalence_cache", [])
        }
        problem.fitness_cache = {
            freeze_json_tree(record["key"]): eval_result_from_dict(dict(record["result"]))
            for record in checkpoint.get("fitness_cache", [])
        }
        completed_generation = int(checkpoint.get("completed_generation", 0))
        generation_seconds = [
            float(value) for value in checkpoint.get("generation_seconds", [])
        ]
        stage_records = list(checkpoint.get("stages", []))
        init_elapsed = float(checkpoint.get("initial_population_seconds", 0.0))
        resumed = True
        status = str(checkpoint.get("status", "resumed"))
        print(
            f"[GA] Resuming after generation {completed_generation}; "
            f"population={len(population)}, seen_classes={len(problem.eq_cache)}",
            flush=True,
        )
    else:
        init_started = time.perf_counter()
        print("[Info] Building initial population ...", flush=True)
        population = generate_population(problem, cfg.pop_size)
        if not population:
            raise RuntimeError(
                "Initial population is empty. Increase --det-mult-max or --diag-max."
            )

        scored: List[Tuple[float, Individual]] = []
        score_started = time.perf_counter()
        last_report = score_started
        report_step = max(1, len(population) // 10)
        next_report = report_step
        for index, ind in enumerate(population, 1):
            scored.append((evaluate(problem, ind)[0].fitness, ind))
            now = time.perf_counter()
            if index >= next_report or now - last_report >= 60.0:
                print(
                    f"[GA Init] scored={index}/{len(population)} "
                    f"elapsed={format_duration(now - score_started)}",
                    flush=True,
                )
                while next_report <= index:
                    next_report += report_step
                last_report = now
        scored.sort(key=lambda item: item[0])
        population = [ind for _, ind in scored[: min(cfg.init_keep, len(scored))]]
        init_elapsed = time.perf_counter() - init_started
        status = "complete" if cfg.generations == 0 else "initialized"
        initial_top = top_k(problem, population, cfg.top_k)
        stage_records.append(
            {
                "generation": 0,
                "population": len(population),
                "elapsed_seconds": init_elapsed,
                "elapsed_hms": format_duration(init_elapsed),
                "top": [ranked_candidate_to_dict(item) for item in initial_top],
            }
        )
        write_json_atomic(
            stages_dir / "generation_0000.json",
            {
                "schema_version": 1,
                "signature": signature,
                "stage": stage_records[-1],
                "updated_at": local_timestamp(),
            },
        )
        write_json_atomic(checkpoint_path, checkpoint_payload())
        print(
            f"[Info] Init best fitness = {scored[0][0]:.8e} | pop = {len(population)} "
            f"| elapsed={format_duration(init_elapsed)}",
            flush=True,
        )
        print(f"[GA] Checkpoint written: {checkpoint_path}", flush=True)

    if cfg.ga_stop_after_generation == completed_generation and (
        completed_generation < cfg.generations
    ):
        status = (
            "stopped_after_initialization"
            if completed_generation == 0
            else "stopped_after_generation"
        )
        write_json_atomic(checkpoint_path, checkpoint_payload())
        print(
            f"[GA] Normal stop requested after generation {completed_generation}. "
            "Use --ga-resume to continue.",
            flush=True,
        )
    else:
        try:
            for gen in range(completed_generation + 1, cfg.generations + 1):
                generation_started = time.perf_counter()
                population = evolve_population(problem, population)
                current_top = top_k(problem, population, cfg.top_k)
                best = current_top[0][0]
                generation_elapsed = time.perf_counter() - generation_started
                generation_seconds.append(generation_elapsed)
                completed_generation = gen
                status = "complete" if gen == cfg.generations else "generation_complete"
                stage_record = {
                    "generation": gen,
                    "population": len(population),
                    "seen_equivalence_classes": len(problem.eq_cache),
                    "fitness_cache_entries": len(problem.fitness_cache),
                    "elapsed_seconds": generation_elapsed,
                    "elapsed_hms": format_duration(generation_elapsed),
                    "top": [ranked_candidate_to_dict(item) for item in current_top],
                }
                stage_records.append(stage_record)
                write_json_atomic(
                    stages_dir / f"generation_{gen:04d}.json",
                    {
                        "schema_version": 1,
                        "signature": signature,
                        "stage": stage_record,
                        "updated_at": local_timestamp(),
                    },
                )
                write_json_atomic(checkpoint_path, checkpoint_payload())
                print(
                    f"[Gen {gen:03d}] fitness={best.fitness:.8e} "
                    f"rmsd={best.rmsd:.6f} A lattice={best.geom:.6e} "
                    f"shape={best.shape:.6e} elapsed={format_duration(generation_elapsed)}",
                    flush=True,
                )
                print(f"[GA] Checkpoint written: {checkpoint_path}", flush=True)
                if cfg.ga_stop_after_generation == gen and gen < cfg.generations:
                    status = "stopped_after_generation"
                    write_json_atomic(checkpoint_path, checkpoint_payload())
                    print(
                        f"[GA] Normal stop requested after generation {gen}. "
                        "Use --ga-resume to continue.",
                        flush=True,
                    )
                    break
        except KeyboardInterrupt:
            print(
                "\n[GA] Interrupted. The last completed-generation checkpoint is intact; "
                "the active incomplete generation will be recomputed.",
                flush=True,
            )
            raise

    if completed_generation == cfg.generations:
        status = "complete"
        write_json_atomic(checkpoint_path, checkpoint_payload())
    if timing is not None:
        timing.update(
            {
                "mode": "ga",
                "status": status,
                "resumed": resumed,
                "completed_generation": completed_generation,
                "initial_population_seconds": init_elapsed,
                "generation_seconds": generation_seconds,
                "generation_count": len(generation_seconds),
                "stages": stage_records,
                "checkpoint": str(checkpoint_path),
            }
        )
    return population


def main(argv: Optional[Sequence[str]] = None) -> int:
    run_started_perf = time.perf_counter()
    run_started_at = local_timestamp()
    print(f"[Timer] Started at {run_started_at}", flush=True)
    cfg = parse_args(argv)
    problem = build_problem(cfg)
    setup_elapsed = time.perf_counter() - run_started_perf

    print(f"[Info] Files: A='{cfg.file_a}', B='{cfg.file_b}'", flush=True)
    print(f"[Info] Species: {list(problem.species)}", flush=True)
    print(f"[Info] Formula units: Z_A={problem.z_a}, Z_B={problem.z_b}", flush=True)
    print(
        f"[Info] Base det pair: det(H1)=t*{problem.d_a_base}, "
        f"det(H2)=t*{problem.d_b_base}, t={cfg.det_mult_min}..{cfg.det_mult_max}",
        flush=True,
    )
    print(
        f"[Info] HNF diagonal cap: diag_max={cfg.diag_max} "
        f"(source={problem.diag_max_source}, "
        f"no-truncation bound={problem.diag_max_no_truncation_bound})",
        flush=True,
    )
    if (
        problem.diag_max_source == "manual"
        and cfg.diag_max < problem.diag_max_no_truncation_bound
    ):
        print(
            f"[Warn] Manual diag_max={cfg.diag_max} is below the no-truncation "
            f"bound {problem.diag_max_no_truncation_bound}; some requested HNFs "
            "will be excluded.",
            flush=True,
        )
    print(
        f"[Info] RMSD normalization: d0={problem.rmsd_scale_ang:.8f} A "
        f"(source={problem.rmsd_scale_source}, endpoint means="
        f"{problem.mean_nn_a_ang:.8f}/{problem.mean_nn_b_ang:.8f} A)",
        flush=True,
    )
    print(
        "[Info] Fitness = weight_rmsd*(RMSD/d0)^2 + "
        "weight_geom*strain + weight_shape*shape",
        flush=True,
    )
    print(
        f"[Info] Export lattice mode={cfg.export_lattice_mode}"
        + (
            " (final frame fixed; initial a axis and a-b plane aligned for NEB)"
            if cfg.export_lattice_mode == "neb"
            else ""
        ),
        flush=True,
    )
    print(
        f"[Info] Q reduction={cfg.reduce_cell}, dedupe={cfg.dedupe_mode}, "
        f"image_shell={cfg.image_shell}, spglib={'yes' if spglib is not None else 'no'}",
        flush=True,
    )
    print(
        "[Info] Intrinsic period audit="
        f"{cfg.intrinsic_period_audit} (exact cap={cfg.intrinsic_period_max_points})",
        flush=True,
    )
    print(f"[Timer] Setup elapsed={format_duration(setup_elapsed)}", flush=True)

    search_timing: Dict[str, object] = {}
    search_started = time.perf_counter()
    if cfg.mode == "enumerate":
        total = count_exhaustive_pairs(problem)
        print(f"[Info] Exhaustive candidate pairs inside bounds: {total}", flush=True)
        results = run_enumeration(problem, search_timing)
    else:
        population = run_ga(problem, search_timing)
        results = top_k(problem, population, cfg.top_k)
    search_elapsed = time.perf_counter() - search_started
    print(f"[Timer] Search elapsed={format_duration(search_elapsed)}", flush=True)
    print("\n[Result] Top candidates:", flush=True)
    records: List[Dict[str, object]] = []
    candidate_export_seconds: List[float] = []
    export_started = time.perf_counter()
    for rank, (result, ind) in enumerate(results, 1):
        print(
            f"  #{rank:02d}: fitness={result.fitness:.8e} "
            f"rmsd={result.rmsd:.6f} A atom_norm2={result.rmsd2_normalized:.6e} "
            f"lattice={result.geom:.6e} shape={result.shape:.6e} "
            f"det(H1)={result.det_a} det(H2)={result.det_b}",
            flush=True,
        )
        candidate_export_started = time.perf_counter()
        records.append(export_candidate(problem, rank, ind, result))
        candidate_export_elapsed = time.perf_counter() - candidate_export_started
        candidate_export_seconds.append(candidate_export_elapsed)
        print(
            f"[Timer] Candidate #{rank:02d} export="
            f"{format_duration(candidate_export_elapsed)}",
            flush=True,
        )

    export_elapsed = time.perf_counter() - export_started
    timing: Dict[str, object] = {
        "started_at": run_started_at,
        "finished_at": None,
        "setup_seconds": setup_elapsed,
        "search_seconds": search_elapsed,
        "export_seconds": export_elapsed,
        "summary_write_seconds": None,
        "total_elapsed_seconds": None,
        "total_elapsed_hms": None,
        "candidate_export_seconds": candidate_export_seconds,
        "search_detail": search_timing,
        "clock": "time.perf_counter",
        "timestamp_timezone": "local system timezone",
    }
    summary_started = time.perf_counter()
    save_summary(problem, records, timing)
    summary_elapsed = time.perf_counter() - summary_started
    if records:
        print("\n[Best] H1 HNF:")
        print(np.array(records[0]["H1_hnf"], dtype=int))
        print("[Best] H2 HNF:")
        print(np.array(records[0]["H2_hnf"], dtype=int))
        print(f"[Best] SNF topology key: {records[0]['snf_pair_key']}")
        print(
            "[Best] Intrinsic period: "
            f"{records[0].get('intrinsic_period_status')} | "
            f"atomic class={records[0].get('atomic_coupled_imt_key', 'unverified')}",
            flush=True,
        )
        print("[Best] Q-reduced display H1 = Q_A @ H1:")
        print(np.array(records[0]["H1_display_QH"], dtype=int))
        print("[Best] Q-reduced display H2 = Q_B @ H2:")
        print(np.array(records[0]["H2_display_QH"], dtype=int))
        snf_classes = build_snf_classes(records)
        print(f"[SNF] Top-{len(records)} candidates collapse to {len(snf_classes)} integer-topology class(es):")
        for entry in snf_classes:
            print(
                f"  {entry['snf_pair_key']}: count={entry['count']} "
                f"best_rank={entry['best_rank']} best_fitness={float(entry['best_fitness']):.8e}",
                flush=True,
            )
        atomic_imt_classes = build_intrinsic_period_classes(records, "atomic_coupled_imt_key")
        print(
            f"[Intrinsic] Exact atom-period audit gives {len(atomic_imt_classes)} "
            "Q-coupled class(es) among the exported candidates.",
            flush=True,
        )
    total_elapsed = time.perf_counter() - run_started_perf
    timing.update(
        {
            "finished_at": local_timestamp(),
            "summary_write_seconds": summary_elapsed,
            "total_elapsed_seconds": total_elapsed,
            "total_elapsed_hms": format_duration(total_elapsed),
        }
    )
    write_timing_receipt(problem, timing)
    print(
        f"[Timer] Export elapsed={format_duration(export_elapsed)} | "
        f"summary={format_duration(summary_elapsed)} | "
        f"total={format_duration(total_elapsed)}",
        flush=True,
    )
    print(f"\n[Done] Exported matched POSCARs and metrics to '{cfg.output_dir}'.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1)
