"""Step 2: preflop features for every canonical hand, k-means hand clusters."""
import os
import sys
import time
from itertools import combinations

import numba as nb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402


@nb.njit(nogil=True, parallel=True)
def pre_all(t, canon, hst, lv, i1, i2, reps, n_boards):
    out = np.zeros((reps.shape[0], 3))
    for i in nb.prange(reps.shape[0]):
        rng = np.array([np.uint64(i * 2654435761 + 12345678912345)], dtype=np.uint64)
        for _ in range(4):
            E.rand_next(rng)
        x = np.zeros(3)
        E.feat_pre(t, canon, hst, lv, i1, i2, reps[i], n_boards, rng, x)
        out[i] = x
    return out


@nb.njit(nogil=True, parallel=True)
def sample_feats(t, canon, hst, lv, i1, i2, n, nb_):
    out = np.zeros((n, 3))
    for i in nb.prange(n):
        rng = np.array([np.uint64(i * 2654435761 + 998877665544 + nb_)], dtype=np.uint64)
        for _ in range(4):
            E.rand_next(rng)
        deck = np.arange(36, dtype=np.int64)
        E.draw(rng, deck)
        hole = np.sort(deck[:5])
        x = np.zeros(3)
        E.feat_runouts(t, canon, hst, lv, i1, i2, hole, deck[10:15], nb_, x)
        out[i] = x
    return out


@nb.njit(nogil=True, parallel=True)
def assign(X, C, lab):
    for i in nb.prange(X.shape[0]):
        lab[i] = E.nearest(C, X[i])


def kmeans(X, k, w=None, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    w = np.ones(len(X)) if w is None else w
    sub = rng.choice(len(X), size=min(40000, len(X)), replace=False, p=w / w.sum())
    S = X[sub]
    C = [S[rng.integers(len(S))]]
    d = ((S - C[0]) ** 2).sum(1)
    for _ in range(k - 1):                     # k-means++ seeding
        c = S[rng.choice(len(S), p=d / d.sum())]
        C.append(c)
        d = np.minimum(d, ((S - c) ** 2).sum(1))
    C = np.array(C)
    lab = np.zeros(len(X), dtype=np.int64)
    for _ in range(iters):
        assign(X, C, lab)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = np.average(X[m], axis=0, weights=w[m])
    return C, lab


if __name__ == "__main__":
    t0 = time.time()
    T = E.load()
    a = (T["t"], T["canon"], T["hst"], T["lv"], T["i1"], T["i2"])
    hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    colex = sum(E.CN[hands[:, i], i + 1] for i in range(5))
    cid = T["canon"][colex]
    n_c = int(cid.max()) + 1
    reps = np.zeros((n_c, 5), dtype=np.int64)
    reps[cid] = hands
    weights = np.bincount(cid, minlength=n_c).astype(np.float64)
    P = pre_all(*a, reps, 4000)
    print("pre features", round(time.time() - t0, 1), P.mean(0), P[:, 0].min(), P[:, 0].max())
    # Preflop values are compressed, so standardise each feature for clustering.
    sd = P.std(0)
    CP, lab = kmeans(P / sd, E.N_PRE, w=weights)
    print("pre clusters", round(time.time() - t0, 1))
    F = sample_feats(*a, 400000, 3)
    print("flop feats", round(time.time() - t0, 1), F.mean(0))
    Cf, _ = kmeans(F * E.SCALE, E.N_FLOP)
    G = sample_feats(*a, 400000, 4)
    print("turn feats", round(time.time() - t0, 1), G.mean(0))
    Ct, _ = kmeans(G * E.SCALE, E.N_TURN)
    np.savez(os.path.join(E.CACHE, "abstraction.npz"), pre_bk=lab.astype(np.int64), pre_eq=P[:, 0], pre_x=P,
             cf=Cf, ct=Ct, pre_c=CP)
    print("saved", round(time.time() - t0, 1))
    print(np.round(Cf[np.argsort(Cf[:, 0])][::8], 3))
