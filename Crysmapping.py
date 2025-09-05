#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

"""

import os
import sys
import math
import json
import random
from functools import reduce
import numpy as np
from scipy.optimize import linear_sum_assignment
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

# --------------------------
# User parameters
# --------------------------
# 默认用你给的文件名；可改为自己的。
FILE_A = "POSCAR1"
FILE_B = "POSCAR2"

SEED = 1000                 # 固定随机种子
POP_SIZE = 10000              # 初始规模
INIT_KEEP = 1000             # 初始化后保留的更优个体数
NUM_GENERATIONS = 60        # 进化代数
SUBSTITUTE_NUMBER = 192     # 每代替换最差的个体数
ELITE_KEEP = 32             # 保留数量
TOP_K_EXPORT = 10           # 导出前 K 个候选

DET_MULT_MAX = 12           # t 的最大值（det 只在 t ∈ [1..DET_MULT_MAX] 上采样）
DIAG_MAX = 12               # HNF 对角元上限

WEIGHT_GEOM = 1.0           # Gram 惩罚权重
WEIGHT_RMSD = 1.0           # RMSD^2 权重

# --------------------------
# Utilities
# --------------------------
rng = random.Random(SEED) if SEED is not None else random.Random()

def divisors(n: int):
    """正因子列表（升序）"""
    n = int(n)
    if n <= 0:
        return []
    small, large = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            small.append(i)
            if i*i != n:
                large.append(n//i)
        i += 1
    return small + large[::-1]

def as_row_lattice_matrix(struct: Structure):
    return np.array(struct.lattice.matrix, dtype=float)  # rows = a,b,c

def structure_species_and_frac(struct: Structure):
    labels = np.array([str(sp) for sp in struct.species], dtype=object)
    frac = np.array(struct.frac_coords, dtype=float)
    return labels, frac

def species_counts(struct: Structure, species_order=None):
    d = struct.composition.get_el_amt_dict()
    if species_order is None:
        species = tuple(sorted(d.keys()))
    else:
        species = tuple(species_order)
    counts = np.array([int(round(d.get(s, 0))) for s in species], dtype=int)
    return species, counts

def metric_tensor(T):
    return T @ T.T

def frob2(A):
    return float(np.sum(A*A))

def canonical_geom_key(T):
    """几何去重键：Gram 的 6 个独立分量 + 排序后长度，做 1e-8 舍入稳健去噪"""
    G = metric_tensor(T)
    elems = (G[0,0], G[1,1], G[2,2], G[0,1], G[1,2], G[0,2])
    lens = np.linalg.norm(T, axis=1)
    tup = tuple([round(x, 8) for x in elems] + sorted([round(x, 8) for x in lens]))
    return tup

def sort_lattice_and_matrix(T, M):
    lens = np.linalg.norm(T, axis=1)
    order = np.argsort(lens)
    return T[order], M[order], lens[order]

def det_int(M):
    return int(round(np.linalg.det(M)))

# --------------------------
# HNF parametrization
# H = [[a,0,0],[b,d,0],[c,e,f]]
# a,d,f>0; 0<=b<d; 0<=c,e<f; det(H)=a*d*f
# --------------------------
def sample_hnf_with_det(det_target: int, diag_max=DIAG_MAX):
    """随机生成 (a,d,f) 满足 a*d*f = det_target，并尽量不超过 diag_max。"""
    if det_target <= 0:
        return None
    divs = divisors(det_target)
    if not divs:
        return None
    # 优先 diag<=diag_max
    for _ in range(64):
        a = rng.choice([x for x in divs if x <= diag_max])
        rem = det_target // a
        divs2 = [x for x in divisors(rem) if x <= diag_max]
        if not divs2:
            continue
        d = rng.choice(divs2)
        f = rem // d
        if f <= diag_max and a*d*f == det_target and a>0 and d>0 and f>0:
            return a, d, f
    # fallback：不限制 diag_max
    a = rng.choice(divs)
    rem = det_target // a
    d = rng.choice(divisors(rem))
    f = rem // d
    if a*d*f == det_target and a>0 and d>0 and f>0:
        return a, d, f
    return None

def _hnf_from_diag(diag):
    a, d, f = map(int, diag)
    b = 0 if d == 1 else rng.randint(0, d-1)
    c = 0 if f == 1 else rng.randint(0, f-1)
    e = 0 if f == 1 else rng.randint(0, f-1)
    return np.array([[a,0,0],[b,d,0],[c,e,f]], dtype=int)

def hnf_flat(H):
    return tuple(int(x) for x in H.reshape(-1).tolist())

def hnf_from_flat(v):
    return np.array(v, dtype=int).reshape(3,3)

# --------------------------
# Fast supercell build (corrected)
# Row-lattice convention:
#   L' = H @ L
#   f' = (f @ H^{-1} + n @ H^{-1}) mod 1,  n in [0..a)×[0..d)×[0..f)
# --------------------------
def make_supercell_fast(L_row, frac_coords, labels, H):
    """
    Fast supercell build under row-lattice convention (CORRECTED):
      L' = H @ L,
      f' = (f @ H^{-1} + n @ H^{-1}) mod 1,
      n ∈ [0,a)×[0,d)×[0,f)
    """
    H = np.array(H, dtype=int)
    a, d, f = H[0,0], H[1,1], H[2,2]
    if a<=0 or d<=0 or f<=0:
        raise ValueError("HNF diag must be positive.")

    Lp = H @ L_row
    invH = np.linalg.inv(H.astype(float))

    # 先把“基”分数坐标变换到新胞：f_base = f @ H^{-1}
    f_base = frac_coords @ invH
    f_base -= np.floor(f_base)   # wrap to [0,1)

    # 枚举 HNF 的平移代表元：n ∈ [0,a)×[0,d)×[0,f)
    ns = np.stack(np.meshgrid(np.arange(a), np.arange(d), np.arange(f), indexing='ij'),
                  axis=-1).reshape(-1,3)
    shifts = ns @ invH                   # shape [det,3]

    # 复制全部原子： (f_base + shifts)
    detH = a*d*f
    frac_rep = (f_base[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    frac_rep -= np.floor(frac_rep)       # wrap to [0,1)

    labels_rep = np.repeat(labels, detH)
    return Lp, labels_rep, frac_rep

# --------------------------
# Cost terms
# --------------------------
def gram_penalty(T1, T2):
    G1 = metric_tensor(T1)
    G2 = metric_tensor(T2)
    scale = max(1e-12, (np.trace(G1) + np.trace(G2)) * 0.5)
    return frob2(G1 - G2) / scale

def periodic_cost_matrix(B1, B2):
    diff = B1[:, None, :] - B2[None, :, :]
    diff -= np.round(diff)  # min image
    return np.linalg.norm(diff, axis=2)

def species_hungarian_rmsd2(frac1, lab1, frac2, lab2, species_tuple):
    all_d2 = []
    for s in species_tuple:
        idx1 = np.where(lab1 == s)[0]
        idx2 = np.where(lab2 == s)[0]
        if len(idx1) != len(idx2):
            return 1e9
        if len(idx1) == 0:
            continue
        cost = periodic_cost_matrix(frac1[idx1], frac2[idx2])
        r, c = linear_sum_assignment(cost)
        d = cost[r, c]
        all_d2.append(np.mean(d**2))
    if not all_d2:
        return 1e9
    return float(np.mean(all_d2))

# --------------------------
# Export helpers (integrity check & sorting + save best matrices)
# --------------------------
def _assert_species_integrity(lab1, lab2):
    ok = True
    for s in SPECIES:
        c1 = int(np.sum(lab1 == s))
        c2 = int(np.sum(lab2 == s))
        if c1 != c2:
            print(f"[WARN] species count mismatch for {s}: {c1} vs {c2}")
            ok = False
    return ok

def _reorder_by_species(labels, frac, species_order=None):
    if species_order is None:
        species_order = SPECIES
    order_idx = np.concatenate([np.where(labels==s)[0] for s in species_order if np.any(labels==s)])
    return labels[order_idx], frac[order_idx]

def save_best_matrices(H1_flat, H2_flat, outdir="best_solutions", extra_info=None):
    """把最优 H1/H2 以 txt / npy / json 形式保存，并可带上额外信息（如 fitness、det 等）。"""
    os.makedirs(outdir, exist_ok=True)
    H1 = hnf_from_flat(H1_flat)
    H2 = hnf_from_flat(H2_flat)

    np.savetxt(os.path.join(outdir, "best_H1.txt"), H1, fmt="%d")
    np.savetxt(os.path.join(outdir, "best_H2.txt"), H2, fmt="%d")
    np.save(os.path.join(outdir, "best_H1.npy"), H1)
    np.save(os.path.join(outdir, "best_H2.npy"), H2)

    payload = {"H1": H1.tolist(), "H2": H2.tolist()}
    if extra_info is not None:
        payload.update(extra_info)
    with open(os.path.join(outdir, "best_matrices.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def export_candidate(idx, H1_flat, H2_flat, outdir="best_solutions"):
    os.makedirs(outdir, exist_ok=True)
    H1 = hnf_from_flat(H1_flat)
    H2 = hnf_from_flat(H2_flat)

    Lp1, lab1, f1 = make_supercell_fast(L1, frac1, labels1, H1)
    Lp2, lab2, f2 = make_supercell_fast(L2, frac2, labels2, H2)

    # 自检 & 按 SPECIES 顺序聚合，避免“元素混显”
    _assert_species_integrity(lab1, lab2)
    lab1, f1 = _reorder_by_species(lab1, f1, SPECIES)
    lab2, f2 = _reorder_by_species(lab2, f2, SPECIES)

    s1 = Structure(Lattice(Lp1), lab1.tolist(), f1.tolist(), coords_are_cartesian=False)
    s2 = Structure(Lattice(Lp2), lab2.tolist(), f2.tolist(), coords_are_cartesian=False)

    s1.to(filename=os.path.join(outdir, f"{idx:02d}-POSCAR-i.vasp"), fmt="poscar")
    s2.to(filename=os.path.join(outdir, f"{idx:02d}-POSCAR-f.vasp"), fmt="poscar")

# --------------------------
# Load inputs & composition alignment (by formula units)
# --------------------------
try:
    A0 = Structure.from_file(FILE_A)
    B0 = Structure.from_file(FILE_B)
except Exception as e:
    print(f"Failed to read {FILE_A}/{FILE_B}: {e}")
    sys.exit(1)

L1 = as_row_lattice_matrix(A0)
L2 = as_row_lattice_matrix(B0)
labels1, frac1 = structure_species_and_frac(A0)
labels2, frac2 = structure_species_and_frac(B0)

# 统一物种顺序
SPECIES = tuple(sorted(set([str(sp) for sp in A0.composition]) | set([str(sp) for sp in B0.composition])))
S1, N1 = species_counts(A0, species_order=SPECIES)
S2, N2 = species_counts(B0, species_order=SPECIES)
assert S1 == S2 == SPECIES

def gcd_array(arr):
    vals = [int(x) for x in arr if int(x) != 0]
    if not vals:
        return 1
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    return int(g) if g != 0 else 1

Z1 = gcd_array(N1)
Z2 = gcd_array(N2)
F1 = (N1 // Z1).astype(int)
F2 = (N2 // Z2).astype(int)

if not np.array_equal(F1, F2):
    raise ValueError(f"Incompatible stoichiometric ratios:\n  N1={N1} -> Z1={Z1}, F1={F1}\n  N2={N2} -> Z2={Z2}, F2={F2}")

# 最小基元行列式对（保证 d1*Z1 == d2*Z2 的最小正整数解）
GZ = math.gcd(int(Z1), int(Z2))
D1_BASE = Z2 // GZ
D2_BASE = Z1 // GZ

print(f"[Info] Files: A='{FILE_A}', B='{FILE_B}'")
print(f"[Info] Formula unit: F={F1.tolist()}, Z1={Z1}, Z2={Z2}")
print(f"[Info] Base det pair: d1_base={D1_BASE}, d2_base={D2_BASE}   (d1*Z1 = d2*Z2)")
print(f"[Info] Sampling determinants: det(H1)=t*d1_base, det(H2)=t*d2_base,  t=1..{DET_MULT_MAX}")
print(f"[Info] DIAG_MAX={DIAG_MAX}, DET_MULT_MAX={DET_MULT_MAX}")

# --------------------------
# GA: individual = (H1_flat, H2_flat)
# --------------------------
_fitness_cache = {}
_eqclass_cache = set()

def eq_key_pair(H1_flat, H2_flat):
    T1 = hnf_from_flat(H1_flat) @ L1
    T2 = hnf_from_flat(H2_flat) @ L2
    return (canonical_geom_key(T1), canonical_geom_key(T2))

def random_individual():
    # 只在小倍数 t 上采样（避免超大超胞）
    t = rng.randint(1, DET_MULT_MAX)
    det1 = t * D1_BASE
    det2 = t * D2_BASE

    diag1 = sample_hnf_with_det(det1, diag_max=DIAG_MAX)
    if diag1 is None:
        return None
    H1 = _hnf_from_diag(diag1)

    diag2 = sample_hnf_with_det(det2, diag_max=DIAG_MAX)
    if diag2 is None:
        return None
    H2 = _hnf_from_diag(diag2)

    return (hnf_flat(H1), hnf_flat(H2))

def mutate_individual(ind):
    H1 = hnf_from_flat(ind[0]).copy()
    H2 = hnf_from_flat(ind[1]).copy()
    for H in (H1, H2):
        if rng.random() < 0.5:
            choices = []
            if H[1,1] > 1: choices.append((1,0,H[1,1]))
            if H[2,2] > 1:
                choices.append((2,0,H[2,2]))
                choices.append((2,1,H[2,2]))
            if choices:
                i,j,mod = rng.choice(choices)
                H[i,j] = (H[i,j] + rng.choice([-1,1])) % mod
    return (hnf_flat(H1), hnf_flat(H2))

def crossover_individual(a, b):
    def cross(v1, v2):
        v1 = list(v1); v2 = list(v2)
        p1 = rng.randint(1, len(v1)-2)
        p2 = rng.randint(p1, len(v1)-1)
        return tuple(v1[:p1] + v2[p1:p2] + v1[p2:])
    return (cross(a[0], b[0]), cross(a[1], b[1]))

def eval_fitness(ind):
    key = ind
    if key in _fitness_cache:
        return _fitness_cache[key]
    H1 = hnf_from_flat(ind[0])
    H2 = hnf_from_flat(ind[1])

    d1 = abs(det_int(H1))
    d2 = abs(det_int(H2))
    # 组成硬约束：det(H1)*Z1 == det(H2)*Z2
    if d1*Z1 != d2*Z2:
        _fitness_cache[key] = 1e12
        return 1e12

    T1 = H1 @ L1
    T2 = H2 @ L2
    geom = gram_penalty(T1, T2)

    try:
        Lp1, lab1, f1 = make_supercell_fast(L1, frac1, labels1, H1)
        Lp2, lab2, f2 = make_supercell_fast(L2, frac2, labels2, H2)
    except Exception:
        _fitness_cache[key] = 1e11
        return 1e11

    if f1.shape[0] != f2.shape[0]:
        _fitness_cache[key] = 1e10
        return 1e10

    # 同物种总数一致（防御）
    for s in SPECIES:
        if np.sum(lab1==s) != np.sum(lab2==s):
            _fitness_cache[key] = 1e10
            return 1e10

    rmsd2 = species_hungarian_rmsd2(f1, lab1, f2, lab2, SPECIES)
    value = WEIGHT_RMSD * rmsd2 + WEIGHT_GEOM * geom
    _fitness_cache[key] = float(value)
    return float(value)

def select_parent(pop, k=3):
    cand = rng.sample(pop, k=min(k, len(pop)))
    fits = [eval_fitness(x) for x in cand]
    return cand[int(np.argmin(fits))]

def _all_diag_triplets(n, maxdiag):
    trips = []
    for a in range(1, maxdiag+1):
        if n % a: continue
        rem = n // a
        for d in range(1, maxdiag+1):
            if rem % d: continue
            f = rem // d
            if 1 <= f <= maxdiag:
                trips.append((a,d,f))
    return trips

def generate_population(n):
    pop, seen, tries = [], set(), 0
    max_attempts = max(20000, n*3000)

    # 随机采样
    while len(pop) < n and tries < max_attempts:
        tries += 1
        ind = random_individual()
        if ind is None or ind in seen:
            continue
        ek = eq_key_pair(ind[0], ind[1])  # 几何去重
        if ek in _eqclass_cache:
            continue
        _eqclass_cache.add(ek)
        seen.add(ind)
        pop.append(ind)

    # 兜底：最小基元 × 小 t 的对角枚举 + 随机下三角
    if len(pop) < n:
        seed = []
        for t in range(1, DET_MULT_MAX+1):
            d1 = t * D1_BASE
            d2 = t * D2_BASE
            t1 = _all_diag_triplets(d1, DIAG_MAX)
            t2 = _all_diag_triplets(d2, DIAG_MAX)
            if not t1 or not t2:
                continue
            rng.shuffle(t1); rng.shuffle(t2)
            cap1 = max(1, len(t1)//2)
            cap2 = max(1, len(t2)//2)
            for A in t1[:cap1]:
                for B in t2[:cap2]:
                    H1 = _hnf_from_diag(A)
                    H2 = _hnf_from_diag(B)
                    ind = (hnf_flat(H1), hnf_flat(H2))
                    seed.append(ind)
                    if len(seed) >= 5*n: break
                if len(seed) >= 5*n: break
            if len(seed) >= 5*n: break

        for ind in seed:
            if len(pop) >= n:
                break
            if ind in seen:
                continue
            ek = eq_key_pair(ind[0], ind[1])
            if ek in _eqclass_cache:
                continue
            _eqclass_cache.add(ek)
            seen.add(ind)
            pop.append(ind)

    if len(pop) < n:
        print(f"[Warn] Initial population truncated to {len(pop)}.")
    return pop

def evolve_population(pop):
    fits = np.array([eval_fitness(ind) for ind in pop])
    order = np.argsort(fits)
    elites = [pop[i] for i in order[:ELITE_KEEP]]

    worst_idx = order[-SUBSTITUTE_NUMBER:]
    for i in worst_idx:
        p1 = select_parent(pop)
        p2 = select_parent(pop)
        child = crossover_individual(p1, p2)
        if rng.random() < 0.8:
            child = mutate_individual(child)
        ek = eq_key_pair(child[0], child[1])
        if ek in _eqclass_cache:
            for _ in range(5):
                child = mutate_individual(child)
                ek = eq_key_pair(child[0], child[1])
                if ek not in _eqclass_cache:
                    break
        _eqclass_cache.add(ek)
        pop[i] = child

    fits2 = np.array([eval_fitness(ind) for ind in pop])
    order2 = np.argsort(fits2)
    replace_idx = order2[-ELITE_KEEP:]
    for j, tgt in enumerate(replace_idx):
        pop[tgt] = elites[j % len(elites)]
    return pop

def best_k(pop, k=TOP_K_EXPORT):
    fits = np.array([eval_fitness(ind) for ind in pop])
    order = np.argsort(fits)[:k]
    return [(fits[i], pop[i]) for i in order]

# --------------------------
# Main
# --------------------------
def main():
    print("[Info] Building initial population ...")
    population = generate_population(POP_SIZE)

    # 初始化清理
    fits = [eval_fitness(ind) for ind in population]
    if len(fits) == 0:
        print("[Error] Initial population is empty. Try increasing DET_MULT_MAX / DIAG_MAX.")
        return 1
    keep_idx = np.argsort(fits)[:min(INIT_KEEP, len(population))]
    population = [population[i] for i in keep_idx]
    print(f"[Info] Init best fitness = {min([fits[i] for i in keep_idx]):.6e}  |  pop = {len(population)}")

    for gen in range(1, NUM_GENERATIONS+1):
        population = evolve_population(population)
        top = best_k(population, k=1)[0]
        print(f"[Gen {gen:03d}] Best Fitness = {top[0]:.6e}")

    # 导出前 TOP_K，并在最后单独输出“最优”的 H1/H2
    topk = best_k(population, k=TOP_K_EXPORT)
    print("\n[Result] Top candidates:")
    for rank, (fit, ind) in enumerate(topk, 1):
        H1_flat, H2_flat = ind
        d1 = abs(det_int(hnf_from_flat(H1_flat)))
        d2 = abs(det_int(hnf_from_flat(H2_flat)))
        print(f"  #{rank:02d}: fitness={fit:.6e} | det(H1)={d1:4d}, det(H2)={d2:4d}")
        export_candidate(rank, H1_flat, H2_flat)

    # —— 打印“最终最优”的变换矩阵 —— #
    best_fit, best_ind = topk[0]
    best_H1_flat, best_H2_flat = best_ind
    best_H1 = hnf_from_flat(best_H1_flat)
    best_H2 = hnf_from_flat(best_H2_flat)
    d1 = abs(det_int(best_H1))
    d2 = abs(det_int(best_H2))

    print("\n[Best] Final selected transform matrices:")
    print("[Best] H1 (apply to A => supercell):")
    print(best_H1)
    print("[Best] H2 (apply to B => supercell):")
    print(best_H2)
    print(f"[Best] det(H1)={d1}, det(H2)={d2}  |  ratio det(H1):det(H2) = {d1}:{d2}")

    # 保存到文件
    save_best_matrices(
        best_H1_flat,
        best_H2_flat,
        outdir="best_solutions",
        extra_info={
            "fitness": float(best_fit),
            "det_H1": int(d1),
            "det_H2": int(d2),
            "det_ratio": f"{d1}:{d2}",
            "species_order": list(SPECIES),
            "file_A": FILE_A,
            "file_B": FILE_B,
            "D1_BASE": int(D1_BASE),
            "D2_BASE": int(D2_BASE),
            "DET_MULT_MAX": int(DET_MULT_MAX),
            "DIAG_MAX": int(DIAG_MAX),
        },
    )

    print("\n[Done] Exported top POSCARs and best matrices to ./best_solutions/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
