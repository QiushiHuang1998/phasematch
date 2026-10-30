#!/usr/bin/env python3
"""Exact lattice-period reduction for integer supercell matches.

This module deliberately sits below the geometry and atom-assignment layers.
For an actual pair of supercell bases, it removes *lattice* repetitions that
do not change the rational translation map.  It does not claim that the
resulting period is the smallest CSM period: an atom correspondence can have a
larger period than its sublattice match.

The convention is the one used by Crystmatch's integer-matrix triplet (IMT):

    P = M_B Q M_A^{-1}

where M_A and M_B are column-supercell matrices and Q is unimodular.  The
public row-HNF wrapper accepts the lower row-HNF convention used by
Crysmapping_v5.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


IntMatrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
Snf = Tuple[int, int, int]


class SaturationLimitError(RuntimeError):
    """Raised when exact residue enumeration would exceed the configured cap."""


def _as_int_matrix(matrix: Sequence[Sequence[int]], *, name: str) -> Matrix:
    value = Matrix(matrix)
    if value.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix, got {value.shape}.")
    if any(not entry.is_Integer for entry in value):
        raise ValueError(f"{name} must have integer entries.")
    if value.det() <= 0:
        raise ValueError(f"{name} must have positive determinant.")
    return value


def _tuple_matrix(matrix: Matrix) -> IntMatrix:
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {matrix.shape}.")
    rows = []
    for row in matrix.tolist():
        values = []
        for entry in row:
            if not entry.is_Integer:
                raise ValueError(f"Expected an integer matrix, got entry {entry}.")
            values.append(int(entry))
        rows.append(tuple(values))
    return tuple(rows)  # type: ignore[return-value]


def _gcd(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = gcd(result, abs(int(value)))
    return result


def smith_invariant_3x3(matrix: Sequence[Sequence[int]]) -> Snf:
    """Return the three invariant factors of a nonsingular 3x3 integer matrix."""

    value = _as_int_matrix(matrix, name="SNF matrix")
    det_abs = abs(int(value.det()))
    d1 = _gcd(int(entry) for entry in value)
    minors = []
    for r0 in range(3):
        for r1 in range(r0 + 1, 3):
            for c0 in range(3):
                for c1 in range(c0 + 1, 3):
                    minors.append(int(value.extract([r0, r1], [c0, c1]).det()))
    g2 = _gcd(minors)
    if d1 <= 0 or g2 <= 0 or g2 % d1 or det_abs % g2:
        raise ValueError(f"Cannot derive SNF invariant from {value}.")
    return int(d1), int(g2 // d1), int(det_abs // g2)


def snf_key(snf: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in snf)


def _row_hnf(generators: Sequence[Sequence[int]]) -> Matrix:
    """Return the lower row-HNF spanning the integer row lattice of generators."""

    rows = Matrix(generators)
    if rows.cols != 3 or rows.rows < 3 or rows.rank() != 3:
        raise ValueError("Generators must span a rank-three integer row lattice.")
    if any(not entry.is_Integer for entry in rows):
        raise ValueError("Generators must be integral.")
    # SymPy returns a column HNF.  Transposing turns it into this project's
    # lower row-HNF convention, L' = H @ L.
    result = hermite_normal_form(rows.T).T
    if result.shape != (3, 3) or result.det() <= 0:
        raise ValueError("Could not construct a positive lower row-HNF.")
    return result


def row_hnf_3x3(matrix: Sequence[Sequence[int]]) -> IntMatrix:
    """Return the positive lower row-HNF of a full-rank 3x3 integer basis."""

    value = _as_int_matrix(matrix, name="row-HNF matrix")
    return _tuple_matrix(_row_hnf(value.tolist()))


def _common_denominator(matrix: Matrix) -> Tuple[Matrix, int]:
    denominator = 1
    for entry in matrix:
        denominator = int(np.lcm(denominator, int(entry.q)))
    numerator = matrix.applyfunc(lambda entry: int(entry * denominator))
    common = _gcd([denominator, *(int(entry) for entry in numerator)])
    if common > 1:
        numerator = numerator.applyfunc(lambda entry: int(entry // common))
        denominator //= common
    return numerator, denominator


def _kernel_lattice_modulo(numerator: Matrix, denominator: int, max_residue_points: int) -> Matrix:
    """Find {x in Z^3 | numerator*x == 0 (mod denominator)} as a row HNF."""

    if denominator <= 0:
        raise ValueError("The common denominator must be positive.")
    if denominator == 1:
        return Matrix.eye(3)
    search_size = denominator**3
    if search_size > max_residue_points:
        raise SaturationLimitError(
            "Exact saturation needs %d residue points for denominator %d, above the cap %d. "
            "Increase max_residue_points or use a smaller-period candidate." % (
                search_size,
                denominator,
                max_residue_points,
            )
        )

    generators = [
        (denominator, 0, 0),
        (0, denominator, 0),
        (0, 0, denominator),
    ]
    for residue in product(range(denominator), repeat=3):
        vector = Matrix(residue)
        if all(int(entry) % denominator == 0 for entry in numerator * vector):
            generators.append(residue)
    return _row_hnf(generators)


def _integral_matrix(matrix: Matrix, *, name: str) -> Matrix:
    if any(not entry.is_Integer for entry in matrix):
        raise ValueError(f"{name} should be integral, got {matrix}.")
    return matrix.applyfunc(int)


def _matrix_key(matrix: IntMatrix) -> str:
    return ";".join(",".join(str(value) for value in row) for row in matrix)


@dataclass(frozen=True)
class RationalSmithSignature:
    """Smith data for a positive rational map modulo oriented endpoint bases.

    If ``P = N / d`` has a positive primitive common denominator, then ``d``
    and the integer Smith factors of ``N`` are invariant under independent
    proper-unimodular (``GL+(3,Z)``) changes of the source and target primitive
    bases.  This is a
    coarse rational-map invariant; it intentionally cannot retain a separate
    IMT Q coordinate under the full basis-change group.
    """

    denominator: int
    numerator_snf: Snf

    @property
    def key(self) -> str:
        return f"den:{self.denominator}|num_snf:{snf_key(self.numerator_snf)}"

    def as_dict(self) -> dict[str, object]:
        return {
            "rational_map_denominator": int(self.denominator),
            "rational_map_numerator_snf": list(self.numerator_snf),
            "rational_map_smith_key": self.key,
        }


def rational_smith_signature(matrix: Matrix) -> RationalSmithSignature:
    """Return the primitive rational Smith data of a positive 3x3 map.

    The input must be exact SymPy rationals.  Accepting floating-point maps
    here would silently turn a basis invariant into a tolerance-dependent
    descriptor, so that is intentionally rejected.
    """

    value = Matrix(matrix)
    if value.shape != (3, 3):
        raise ValueError(f"Rational map must be 3x3, got {value.shape}.")
    if any(not entry.is_Rational for entry in value):
        raise ValueError("Rational Smith data requires an exact rational matrix.")
    if value.det() <= 0:
        raise ValueError("Rational map must have positive determinant.")
    numerator, denominator = _common_denominator(value)
    return RationalSmithSignature(
        denominator=int(denominator),
        numerator_snf=smith_invariant_3x3(numerator.tolist()),
    )


@dataclass(frozen=True)
class SaturatedSLMSignature:
    """Minimal lattice-period IMT plus the coarse quotient signature.

    ``relative_q_row`` is kept because it reconstructs the coupling between the
    two saturated sublattices.  It is a canonical IMT coordinate only after
    the primitive-cell convention has been fixed; the two SNF quotient types
    and the saturation factors are invariant under unimodular primitive-basis
    changes.
    """

    source_hnf_row: IntMatrix
    target_hnf_row: IntMatrix
    relative_q_row: IntMatrix
    source_snf: Snf
    target_snf: Snf
    source_index: int
    target_index: int
    source_reduction_factor: int
    target_reduction_factor: int
    rational_map_smith: RationalSmithSignature
    atomic_period_verified: bool = False

    @property
    def snf_pair_key(self) -> str:
        return f"A:{snf_key(self.source_snf)}|B:{snf_key(self.target_snf)}"

    @property
    def coupled_imt_key(self) -> str:
        return (
            f"{self.snf_pair_key}|Qrow:{_matrix_key(self.relative_q_row)}"
        )

    @property
    def input_is_reported_period_minimal(self) -> bool:
        return self.source_reduction_factor == 1 and self.target_reduction_factor == 1

    def as_dict(self) -> dict[str, object]:
        return {
            "source_hnf_row": [list(row) for row in self.source_hnf_row],
            "target_hnf_row": [list(row) for row in self.target_hnf_row],
            "relative_q_row": [list(row) for row in self.relative_q_row],
            "source_snf": list(self.source_snf),
            "target_snf": list(self.target_snf),
            "slm_snf_pair_key": self.snf_pair_key,
            "coupled_imt_key": self.coupled_imt_key,
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_reduction_factor": self.source_reduction_factor,
            "target_reduction_factor": self.target_reduction_factor,
            **self.rational_map_smith.as_dict(),
            "input_is_reported_period_minimal": self.input_is_reported_period_minimal,
            "saturated_slm_is_minimal": True,
            # An atom assignment has to be checked separately before calling a
            # period the minimal period of a full CSM rather than an SLM.
            "atomic_period_verified": self.atomic_period_verified,
        }


def _signature_from_source_lattice(
    coordinate_map: Matrix,
    source_hnf_row: Matrix,
    *,
    source_index: int,
    target_index: int,
    atomic_period_verified: bool,
) -> SaturatedSLMSignature:
    """Complete a canonical IMT once the source period lattice is known."""

    source_hnf_column = source_hnf_row.T
    target_generators_column = _integral_matrix(
        coordinate_map * source_hnf_column,
        name="target generators",
    )
    target_hnf_row = _row_hnf(target_generators_column.T.tolist())
    target_hnf_column = target_hnf_row.T
    q_column = _integral_matrix(
        target_hnf_column.inv() * coordinate_map * source_hnf_column,
        name="relative Q",
    )
    if q_column.det() != 1:
        raise ValueError(f"The relative Q must have determinant +1, got {q_column.det()}.")

    source_min_index = abs(int(source_hnf_row.det()))
    target_min_index = abs(int(target_hnf_row.det()))
    if source_index % source_min_index or target_index % target_min_index:
        raise ValueError("The reported period is not a sublattice of the input supercells.")
    source_factor = source_index // source_min_index
    target_factor = target_index // target_min_index
    if source_factor != target_factor:
        raise ValueError(
            "A paired supercell representation must reduce by the same common factor; "
            f"got {source_factor} and {target_factor}."
        )

    source_tuple = _tuple_matrix(source_hnf_row)
    target_tuple = _tuple_matrix(target_hnf_row)
    q_row_tuple = _tuple_matrix(q_column.T)
    return SaturatedSLMSignature(
        source_hnf_row=source_tuple,
        target_hnf_row=target_tuple,
        relative_q_row=q_row_tuple,
        source_snf=smith_invariant_3x3(source_tuple),
        target_snf=smith_invariant_3x3(target_tuple),
        source_index=source_index,
        target_index=target_index,
        source_reduction_factor=source_factor,
        target_reduction_factor=target_factor,
        rational_map_smith=rational_smith_signature(coordinate_map),
        atomic_period_verified=atomic_period_verified,
    )


def saturate_supercell_columns(
    source_columns: Sequence[Sequence[int]],
    target_columns: Sequence[Sequence[int]],
    *,
    max_residue_points: int = 250_000,
) -> SaturatedSLMSignature:
    """Remove common supercell repetition from a pair of actual column bases.

    The exact rational translation map is ``P = M_B M_A^{-1}``.  The largest
    parent-lattice sublattice of A whose image is an integer B translation is
    ``K_A = Z^3 intersection P^{-1} Z^3``.  Its image is ``K_B = P K_A``.
    These lattices define the smallest *SLM* period for this deformation map.
    """

    source = _as_int_matrix(source_columns, name="source_columns")
    target = _as_int_matrix(target_columns, name="target_columns")
    coordinate_map = target * source.inv()
    numerator, denominator = _common_denominator(coordinate_map)
    source_hnf_row = _kernel_lattice_modulo(
        numerator,
        denominator,
        max_residue_points,
    )
    source_index = abs(int(source.det()))
    target_index = abs(int(target.det()))
    return _signature_from_source_lattice(
        coordinate_map,
        source_hnf_row,
        source_index=source_index,
        target_index=target_index,
        atomic_period_verified=False,
    )


def saturate_row_imt(
    source_hnf_row: Sequence[Sequence[int]],
    target_hnf_row: Sequence[Sequence[int]],
    relative_q_row: Optional[Sequence[Sequence[int]]] = None,
    *,
    max_residue_points: int = 250_000,
) -> SaturatedSLMSignature:
    """Saturate a row-HNF IMT in Crysmapping's lattice-row convention.

    The physical target basis for an IMT is ``Q_row @ H_B @ L_B``.  The
    existing matcher implicitly uses ``Q_row = I``; passing a nonidentity Q
    makes the coupling explicit and is intentionally not confused with the
    display-only common Q reduction used for cell-shape cleanup.
    """

    source_row = _as_int_matrix(source_hnf_row, name="source_hnf_row")
    target_row = _as_int_matrix(target_hnf_row, name="target_hnf_row")
    q_row = Matrix.eye(3) if relative_q_row is None else _as_int_matrix(relative_q_row, name="relative_q_row")
    if q_row.det() != 1:
        raise ValueError("relative_q_row must be unimodular with determinant +1.")
    return saturate_supercell_columns(
        source_row.T.tolist(),
        (q_row * target_row).T.tolist(),
        max_residue_points=max_residue_points,
    )


def rational_smith_from_row_imt(
    source_hnf_row: Sequence[Sequence[int]],
    target_hnf_row: Sequence[Sequence[int]],
    relative_q_row: Optional[Sequence[Sequence[int]]] = None,
) -> RationalSmithSignature:
    """Return the basis-invariant rational Smith data of a row-basis IMT.

    This is the double-coset invariant of the rational lattice map
    ``P = (Q_row H_B)^T (H_A^T)^-1``.  It is intentionally coarser than the
    fixed-input Q coordinate: ``P = I`` and any unimodular shear have the same
    rational Smith data after arbitrary independent endpoint basis changes.
    """

    source_row = _as_int_matrix(source_hnf_row, name="source_hnf_row")
    target_row = _as_int_matrix(target_hnf_row, name="target_hnf_row")
    q_row = Matrix.eye(3) if relative_q_row is None else _as_int_matrix(
        relative_q_row, name="relative_q_row"
    )
    if q_row.det() != 1:
        raise ValueError("relative_q_row must be unimodular with determinant +1.")
    coordinate_map = (q_row * target_row).T * source_row.T.inv()
    return rational_smith_signature(coordinate_map)


def _same_coset(left: Matrix, right: Matrix, repeat_inverse: Matrix) -> bool:
    return all(entry.is_Integer for entry in repeat_inverse * (left - right))


def _coset_representatives(repeat: Matrix, max_coset_points: int) -> list[Matrix]:
    """Represent Z^3 / repeat Z^3 from the exact column-HNF residue box."""

    factor = abs(int(repeat.det()))
    if factor <= 0:
        raise ValueError("The repeat matrix must be nonsingular.")
    if factor > max_coset_points:
        raise SaturationLimitError(
            "Atom-period verification needs %d exact quotient residues for index %d, above the cap %d."
            % (factor, factor, max_coset_points)
        )
    column_hnf = hermite_normal_form(repeat)
    diagonal = [int(column_hnf[index, index]) for index in range(3)]
    if any(value <= 0 for value in diagonal) or diagonal[0] * diagonal[1] * diagonal[2] != factor:
        raise ValueError(
            f"Could not construct a valid column-HNF residue box for index {factor}."
        )
    representatives = [Matrix(entries) for entries in product(*(range(value) for value in diagonal))]
    if len(representatives) != factor:
        raise ValueError(f"Expected {factor} quotient residues but constructed {len(representatives)}.")
    return representatives


def _fractional_array(values: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3).")
    return array - np.floor(array)


def _find_periodic_atom(
    positions: np.ndarray,
    labels: Sequence[str],
    position: np.ndarray,
    label: str,
    tolerance: float,
) -> Optional[int]:
    candidates = [index for index, value in enumerate(labels) if value == label]
    hits = []
    for index in candidates:
        delta = positions[index] - position
        delta -= np.round(delta)
        if float(np.max(np.abs(delta))) <= tolerance:
            hits.append(index)
    return hits[0] if len(hits) == 1 else None


def _translation_preserves_assignment(
    source_fractional: np.ndarray,
    target_fractional: np.ndarray,
    labels: Sequence[str],
    shift: np.ndarray,
    tolerance: float,
) -> bool:
    for index, label in enumerate(labels):
        source_partner = _find_periodic_atom(
            source_fractional,
            labels,
            (source_fractional[index] + shift) % 1.0,
            label,
            tolerance,
        )
        if source_partner is None:
            return False
        target_expected = (target_fractional[index] + shift) % 1.0
        delta = target_fractional[source_partner] - target_expected
        delta -= np.round(delta)
        if float(np.max(np.abs(delta))) > tolerance:
            return False
    return True


def resolve_atomic_period_columns(
    source_columns: Sequence[Sequence[int]],
    target_columns: Sequence[Sequence[int]],
    source_fractional: Sequence[Sequence[float]],
    target_fractional: Sequence[Sequence[float]],
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    *,
    tolerance: float = 1e-7,
    max_residue_points: int = 250_000,
    max_coset_points: int = 250_000,
) -> SaturatedSLMSignature:
    """Find the smallest period preserved by a paired atom correspondence.

    The input fractional arrays must already be ordered so row ``i`` in the
    source maps to row ``i`` in the target.  Every translation in the finite
    quotient between the input supercell and the saturated SLM is tested.  The
    accepted translations are required to form a subgroup and are then lifted
    to the atom-period lattice exactly.
    """

    source = _as_int_matrix(source_columns, name="source_columns")
    target = _as_int_matrix(target_columns, name="target_columns")
    source_positions = _fractional_array(source_fractional, name="source_fractional")
    target_positions = _fractional_array(target_fractional, name="target_fractional")
    labels_a = [str(label) for label in source_labels]
    labels_b = [str(label) for label in target_labels]
    if source_positions.shape != target_positions.shape:
        raise ValueError("Source and target fractional arrays must have the same shape.")
    if len(labels_a) != len(labels_b) or len(labels_a) != source_positions.shape[0]:
        raise ValueError("Labels must match the number of paired atoms.")
    if labels_a != labels_b:
        raise ValueError("Atom rows must be species-consistent after assignment.")

    coordinate_map = target * source.inv()
    lattice_signature = saturate_supercell_columns(
        source.tolist(),
        target.tolist(),
        max_residue_points=max_residue_points,
    )
    source_saturated_column = Matrix(lattice_signature.source_hnf_row).T
    repeat = _integral_matrix(
        source_saturated_column.inv() * source,
        name="source repeat matrix",
    )
    representatives = _coset_representatives(repeat, max_coset_points)
    repeat_inverse = repeat.inv()

    accepted: set[int] = set()
    for index, representative in enumerate(representatives):
        parent_translation = source_saturated_column * representative
        shift = source.inv() * parent_translation
        shift_float = np.array([float(value) for value in shift], dtype=float)
        if _translation_preserves_assignment(
            source_positions,
            target_positions,
            labels_a,
            shift_float,
            tolerance,
        ):
            accepted.add(index)
    if not accepted:
        raise ValueError("The identity translation did not preserve the paired atom assignment.")

    for left in accepted:
        for right in accepted:
            total = representatives[left] + representatives[right]
            target_index = next(
                (
                    index
                    for index, representative in enumerate(representatives)
                    if _same_coset(total, representative, repeat_inverse)
                ),
                None,
            )
            if target_index is None or target_index not in accepted:
                raise ValueError("Accepted atom translations did not form a closed quotient subgroup.")

    generators = source.T.tolist()
    for index in sorted(accepted):
        translation = source_saturated_column * representatives[index]
        generators.append([int(value) for value in translation])
    atomic_source_hnf_row = _row_hnf(generators)
    return _signature_from_source_lattice(
        coordinate_map,
        atomic_source_hnf_row,
        source_index=abs(int(source.det())),
        target_index=abs(int(target.det())),
        atomic_period_verified=True,
    )
