"""
build_tables.py - step 1 of the offline pipeline. Builds the card tables:

  score2x3[pair_id, triple_id]   dense rank of a 5-card hand (0 = cards overlap)
  board_canon[colex5(board)]     suit-canonical id of a 5-card board
  hs_thr[canon_id, level]        smallest score that reaches HS_LEVELS[level]
                                 against a uniformly random opponent hand

Output: training/cache/tables.npz (the runtime copy is written by export.py).
"""
import itertools
import os
import sys
import time

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from evaluator import _build_table  # noqa: E402

CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

# Fine near 1.0: "how close to the nuts" is what matters once money goes in.
HS_LEVELS = np.array(
    [0.05 * i for i in range(1, 15)]          # .05 .. .70
    + [0.70 + 0.02 * i for i in range(1, 11)]  # .72 .. .90
    + [0.90 + 0.005 * i for i in range(1, 20)]  # .905 .. .995
    + [0.9975, 0.999], dtype=np.float64)


def comb(n, k):
    from math import comb as c
    return c(n, k)


def build_score_table():
    tbl = _build_table()
    scores = np.fromiter(tbl.values(), dtype=np.int64, count=len(tbl))
    uniq = np.unique(scores)
    dense = {int(s): i + 1 for i, s in enumerate(uniq)}  # 0 reserved for "overlap"
    print("distinct 5-card scores:", len(uniq))
    t = np.zeros((630, 7140), dtype=np.uint16)
    for b in range(36):
        for a in range(b):
            pid = comb(b, 2) + a
            pm = (1 << a) | (1 << b)
            for e in range(36):
                for d in range(e):
                    for c_ in range(d):
                        tm = (1 << c_) | (1 << d) | (1 << e)
                        if pm & tm:
                            continue
                        tid = comb(e, 3) + comb(d, 2) + c_
                        t[pid, tid] = dense[tbl[pm | tm]]
    return t


def build_board_canon():
    """Map every 5-card board to the id of its suit-canonical form."""
    perms = list(itertools.permutations(range(4)))
    boards = np.array(list(itertools.combinations(range(36), 5)), dtype=np.int64)
    best = None
    for p in perms:
        pa = np.array(p)
        m = (boards & ~3) | pa[boards & 3]
        mask = (np.int64(1) << m).sum(axis=1)
        best = mask if best is None else np.minimum(best, mask)
    uniq, inv = np.unique(best, return_inverse=True)
    # colex index of each board (boards are ascending within a row)
    cn = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)
    colex = sum(cn[boards[:, i], i + 1] for i in range(5))
    canon = np.zeros(len(boards), dtype=np.uint16)
    canon[colex] = inv
    # one representative board for every canonical id
    reps = np.zeros((len(uniq), 5), dtype=np.int64)
    reps[inv] = boards
    print("canonical boards:", len(uniq))
    return canon, reps


@nb.njit(nogil=True, parallel=True)
def build_hs(t, reps, levels, n_ranks):
    """Enumerate every opponent hand on every canonical board."""
    n = reps.shape[0]
    out = np.zeros((n, len(levels)), dtype=np.uint16)
    pairs = [(a, b) for b in range(36) for a in range(b)]
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    tri_idx = [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
               (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]

    cn = np.zeros((37, 6), dtype=np.int64)
    for i in range(37):
        cn[i, 0] = 1
        for k in range(1, 6):
            cn[i, k] = 0 if i == 0 else cn[i - 1, k - 1] + cn[i - 1, k]

    for bi in nb.prange(n):
        b = reps[bi]
        tids = np.zeros(10, dtype=np.int64)
        for j in range(10):
            x, y, z = tri_idx[j]
            tids[j] = cn[b[z], 3] + cn[b[y], 2] + b[x]
        # best score of each 2-card holding on this board
        best = np.zeros((36, 36), dtype=np.int32)
        for p in range(630):
            s = 0
            for j in range(10):
                v = t[p, tids[j]]
                if v > s:
                    s = v
            best[pa[p], pb[p]] = s
            best[pb[p], pa[p]] = s
        rest = np.zeros(31, dtype=np.int64)
        m = 0
        for c in range(36):
            if c != b[0] and c != b[1] and c != b[2] and c != b[3] and c != b[4]:
                rest[m] = c
                m += 1
        hist = np.zeros(n_ranks + 2, dtype=np.int64)
        for i1 in range(31):
            c1 = rest[i1]
            for i2 in range(i1 + 1, 31):
                c2 = rest[i2]
                s12 = best[c1, c2]
                for i3 in range(i2 + 1, 31):
                    c3 = rest[i3]
                    s3 = max(s12, best[c1, c3], best[c2, c3])
                    for i4 in range(i3 + 1, 31):
                        c4 = rest[i4]
                        s4 = max(s3, best[c1, c4], best[c2, c4], best[c3, c4])
                        for i5 in range(i4 + 1, 31):
                            c5 = rest[i5]
                            s5 = max(s4, best[c1, c5], best[c2, c5], best[c3, c5], best[c4, c5])
                            hist[s5] += 1
        total = hist.sum()
        cum = np.cumsum(hist)
        # HS(r) = P(opp < r) + P(opp == r)/2
        lev = 0
        for r in range(n_ranks + 2):
            hs = ((cum[r] - hist[r]) + 0.5 * hist[r]) / total
            while lev < len(levels) and hs >= levels[lev]:
                out[bi, lev] = r
                lev += 1
        while lev < len(levels):
            out[bi, lev] = 65535  # level is unreachable on this board
            lev += 1
        # Ranks that no opponent hand reaches are still legal for us; levels
        # assigned above only ever rise, so a plain "count thr <= score" works.
    return out


if __name__ == "__main__":
    t0 = time.time()
    t = build_score_table()
    print("score table", t.shape, time.time() - t0)
    canon, reps = build_board_canon()
    print("canon", time.time() - t0)
    hs = build_hs(t, reps, HS_LEVELS, int(t.max()))
    print("hs", hs.shape, time.time() - t0)
    np.savez(os.path.join(CACHE, "tables.npz"), t=t, canon=canon, hs=hs, levels=HS_LEVELS)
    print("saved", time.time() - t0)
