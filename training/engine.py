"""
engine.py - numba core used by the whole offline pipeline.

  * a compact re-implementation of game.py (checked by test_parity.py)
  * hand-strength features and the hand/betting abstraction
  * external-sampling MCCFR, optionally against a fixed opponent model
    (a "restricted Nash response")
  * rule-based opponent archetypes and a fast match simulator
"""
import os

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

N_PRE, N_FLOP, N_TURN, N_RIVER = 40, 64, 64, 14
TEX_FLOP, TEX_TURN, TEX_RIVER = 6, 4, 4
N_POTCTX = 9
N_SEQ = 62
STACK = 80

# ---- street betting sequences --------------------------------------------


def build_seq():
    """Node 0 = street start. Symbols: 0 check/call, 1 half raise, 2 pot raise."""
    ids = {(): 0}
    order = [()]
    i = 0
    nxt = []
    while i < len(order):
        h = order[i]
        row = [-1, -1, -1]
        n_raises = sum(1 for x in h if x != 0)
        for sym in (0, 1, 2):
            if sym == 0:
                if len(h) != 0:     # a call after a raise / second check ends the street
                    continue
            elif n_raises >= 4:
                continue
            h2 = h + (sym,)
            if h2 not in ids:
                ids[h2] = len(order)
                order.append(h2)
            row[sym] = ids[h2]
        nxt.append(row)
        i += 1
    return np.array(nxt, dtype=np.int32)


NEXT_SEQ = build_seq()
assert NEXT_SEQ.shape[0] == N_SEQ

OFF_PRE = 0
SIZE_PRE = 2 * N_PRE * N_SEQ
OFF_FLOP = SIZE_PRE
SIZE_FLOP = 2 * 3 * N_POTCTX * TEX_FLOP * N_FLOP * N_SEQ
OFF_TURN = OFF_FLOP + SIZE_FLOP
SIZE_TURN = 2 * 3 * N_POTCTX * TEX_TURN * N_TURN * N_SEQ
OFF_RIVER = OFF_TURN + SIZE_TURN
SIZE_RIVER = 2 * 3 * N_POTCTX * TEX_RIVER * N_RIVER * N_SEQ
N_INFOSETS = OFF_RIVER + SIZE_RIVER

_e = [2, 4, 6, 9, 14, 20, 30, 45, 80]
POTCTX = np.array([next(i for i, x in enumerate(_e) if c <= x) for c in range(81)], dtype=np.int32)

RIVER_EDGES = np.array([.30, .50, .65, .75, .82, .87, .91, .94, .96, .975, .985, .992, .997], dtype=np.float64)

# state vector fields
ST, SEQ, ACT, B0, B1, C0, C1, RS, LI, A0, A1, LA, PA, PCTX, TERM, LOSER = range(16)
NF = 16


@nb.njit(inline="always")
def init_state(S, d):
    S[d, ST] = 0; S[d, SEQ] = 0; S[d, ACT] = 0
    S[d, B0] = 1; S[d, B1] = 2; S[d, C0] = 1; S[d, C1] = 2
    S[d, RS] = 0; S[d, LI] = 2; S[d, A0] = 0; S[d, A1] = 0
    S[d, LA] = -1; S[d, PA] = -1; S[d, PCTX] = 0; S[d, TERM] = 0; S[d, LOSER] = -1


@nb.njit(inline="always")
def raise_total(S, d, a):
    actor = S[d, ACT]
    my_bet = S[d, B0 + actor]; opp_bet = S[d, B1 - actor]
    my_c = S[d, C0 + actor]
    pot = S[d, C0] + S[d, C1]
    to_call = max(0, opp_bet - my_bet)
    all_in = my_bet + (STACK - my_c)
    pot_to = opp_bet + pot + to_call
    half_to = opp_bet + (pot + to_call) // 2
    target = half_to if a == 2 else pot_to
    return min(max(target, opp_bet + max(S[d, LI], 2)), pot_to, all_in)


@nb.njit(inline="always")
def legal_mask(S, d, m):
    """m[a]=1 for legal actions. RAISE_HALF is dropped when it equals RAISE_POT."""
    actor = S[d, ACT]
    my_bet = S[d, B0 + actor]; opp_bet = S[d, B1 - actor]
    to_call = max(0, opp_bet - my_bet)
    m[0] = 1 if to_call > 0 else 0
    m[1] = 1
    m[2] = 0; m[3] = 0
    if S[d, RS] < 4 and STACK - S[d, C0 + actor] > to_call and STACK - S[d, C1 - actor] > 0:
        m[3] = 1
        if raise_total(S, d, 2) != raise_total(S, d, 3):
            m[2] = 1


@nb.njit(inline="always")
def step(S, d, a):
    """Apply action a at depth d and write the new state to depth d+1."""
    for i in range(NF):
        S[d + 1, i] = S[d, i]
    n = d + 1
    actor = S[d, ACT]; opp = 1 - actor
    if a == 0:
        S[n, TERM] = 1; S[n, LOSER] = actor
        return
    if a == 1:
        to_call = max(0, S[d, B0 + opp] - S[d, B0 + actor])
        amt = min(to_call, STACK - S[d, C0 + actor])
        S[n, B0 + actor] += amt; S[n, C0 + actor] += amt
        S[n, A0 + actor] = 1
        if S[n, A0] == 1 and S[n, A1] == 1 and S[n, B0] == S[n, B1]:
            if S[n, ST] == 3 or S[n, C0] == STACK or S[n, C1] == STACK:
                S[n, TERM] = 2
            else:
                S[n, ST] += 1; S[n, SEQ] = 0; S[n, ACT] = 1
                S[n, B0] = 0; S[n, B1] = 0; S[n, A0] = 0; S[n, A1] = 0
                S[n, RS] = 0; S[n, LI] = 2
                S[n, PA] = S[n, LA]; S[n, LA] = -1
                S[n, PCTX] = POTCTX[S[n, C0]]
        else:
            S[n, ACT] = opp
            S[n, SEQ] = NEXT_SEQ[S[d, SEQ], 0]
        return
    tot = raise_total(S, d, a)
    amt = tot - S[d, B0 + actor]
    S[n, LI] = tot - S[d, B0 + opp]
    S[n, RS] += 1
    S[n, B0 + actor] += amt; S[n, C0 + actor] += amt
    S[n, A0 + opp] = 0; S[n, A0 + actor] = 1
    S[n, LA] = actor; S[n, ACT] = opp
    S[n, SEQ] = NEXT_SEQ[S[d, SEQ], a - 1]


@nb.njit(inline="always")
def info_key(S, d, bk, tex):
    """bk[player, street] hand buckets; tex[street] board texture."""
    st = S[d, ST]; actor = S[d, ACT]; seq = S[d, SEQ]
    b = bk[actor, st]
    if st == 0:
        return OFF_PRE + (actor * N_PRE + b) * N_SEQ + seq
    pa = S[d, PA]
    ag = 0 if pa < 0 else (1 if pa == actor else 2)
    ctx = (actor * 3 + ag) * N_POTCTX + S[d, PCTX]
    if st == 1:
        return OFF_FLOP + ((ctx * TEX_FLOP + tex[1]) * N_FLOP + b) * N_SEQ + seq
    if st == 2:
        return OFF_TURN + ((ctx * TEX_TURN + tex[2]) * N_TURN + b) * N_SEQ + seq
    return OFF_RIVER + ((ctx * TEX_RIVER + tex[3]) * N_RIVER + b) * N_SEQ + seq


# ---- cards and features ----------------------------------------------------

CN = np.zeros((37, 6), dtype=np.int64)
for _i in range(37):
    CN[_i, 0] = 1
    for _k in range(1, 6):
        CN[_i, _k] = 0 if _i == 0 else CN[_i - 1, _k - 1] + CN[_i - 1, _k]

TRI = np.array([(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
                (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)], dtype=np.int64)
PAIRS5 = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)], dtype=np.int64)


@nb.njit(inline="always")
def rand_next(rng):
    x = rng[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    rng[0] = x
    return x


@nb.njit(inline="always")
def rand_int(rng, n):
    return np.int64((rand_next(rng) >> np.uint64(11)) % np.uint64(n))


@nb.njit(inline="always")
def rand_f(rng):
    return float(rand_next(rng) >> np.uint64(11)) / 9007199254740992.0


@nb.njit
def hand_pids(hole, out):
    for i in range(10):
        a = hole[PAIRS5[i, 0]]; b = hole[PAIRS5[i, 1]]
        if a > b:
            a, b = b, a
        out[i] = CN[b, 2] + a


@nb.njit(inline="always")
def best_score(t, pids, b):
    s = 0
    for j in range(10):
        x = b[TRI[j, 0]]; y = b[TRI[j, 1]]; z = b[TRI[j, 2]]
        tid = CN[z, 3] + CN[y, 2] + x
        for p in range(10):
            v = t[pids[p], tid]
            if v > s:
                s = v
    return s


@nb.njit(inline="always")
def level_of(canon, hst, b, s):
    cid = canon[CN[b[0], 1] + CN[b[1], 2] + CN[b[2], 3] + CN[b[3], 4] + CN[b[4], 5]]
    lo = 0; hi = hst.shape[1]
    while lo < hi:
        mid = (lo + hi) >> 1
        if hst[cid, mid] <= s:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(inline="always")
def hs_of(t, canon, hst, pids, b):
    """HS level index of a hand on the sorted 5-card board b."""
    return level_of(canon, hst, b, best_score(t, pids, b))


@nb.njit(inline="always")
def insert_sorted(src, n, c, dst):
    i = 0
    while i < n and src[i] < c:
        dst[i] = src[i]
        i += 1
    dst[i] = c
    while i < n:
        dst[i + 1] = src[i]
        i += 1


@nb.njit
def unseen(hole, board, nb_, out):
    used = np.zeros(36, dtype=np.uint8)
    for c in hole:
        used[c] = 1
    for i in range(nb_):
        used[board[i]] = 1
    m = 0
    for c in range(36):
        if not used[c]:
            out[m] = c
            m += 1
    return m


# The two levels that define the "nut potential" features.
HI1, HI2 = 0.90, 0.97


@nb.njit
def feat_runouts(t, canon, hst, lv, i1, i2, hole, board, nb_, out):
    """Features of (hole, board) over board completions.
    nb_=4: every river. nb_=3: every 4th (turn, river) pair, a fixed subset."""
    pids = np.zeros(10, dtype=np.int64)
    hand_pids(hole, pids)
    rest = np.zeros(36, dtype=np.int64)
    m = unseen(hole, board, nb_, rest)
    s3 = np.zeros(3, dtype=np.int64)
    for i in range(3):
        s3[i] = board[i]
    s3.sort()
    b4 = np.zeros(4, dtype=np.int64)
    b5 = np.zeros(5, dtype=np.int64)
    tot = 0.0; h1 = 0.0; h2 = 0.0; n = 0
    if nb_ == 4:
        insert_sorted(s3, 3, board[3], b4)
        for i in range(m):
            insert_sorted(b4, 4, rest[i], b5)
            l = hs_of(t, canon, hst, pids, b5)
            tot += lv[l]; h1 += l >= i1; h2 += l >= i2; n += 1
    else:
        k = 0
        for i in range(m):
            for j in range(i + 1, m):
                k += 1
                if k % 4 != 0:
                    continue
                insert_sorted(s3, 3, rest[i], b4)
                insert_sorted(b4, 4, rest[j], b5)
                l = hs_of(t, canon, hst, pids, b5)
                tot += lv[l]; h1 += l >= i1; h2 += l >= i2; n += 1
    out[0] = tot / n; out[1] = h1 / n; out[2] = h2 / n


@nb.njit
def feat_pre(t, canon, hst, lv, i1, i2, hole, n_boards, rng, out):
    pids = np.zeros(10, dtype=np.int64)
    hand_pids(hole, pids)
    rest = np.zeros(36, dtype=np.int64)
    m = unseen(hole, hole, 0, rest)
    b5 = np.zeros(5, dtype=np.int64)
    tot = 0.0; h1 = 0.0; h2 = 0.0
    for _ in range(n_boards):
        for i in range(5):
            j = i + rand_int(rng, m - i)
            rest[i], rest[j] = rest[j], rest[i]
            b5[i] = rest[i]
        b5.sort()
        l = hs_of(t, canon, hst, pids, b5)
        tot += lv[l]; h1 += l >= i1; h2 += l >= i2
    out[0] = tot / n_boards; out[1] = h1 / n_boards; out[2] = h2 / n_boards


@nb.njit(inline="always")
def nearest(cent, x):
    best = 0; bd = 1e18
    for k in range(cent.shape[0]):
        d = 0.0
        for j in range(3):
            e = cent[k, j] - x[j]
            d += e * e
        if d < bd:
            bd = d; best = k
    return best


# Feature scaling used for clustering and lookup alike.
SCALE = np.array([1.0, 0.7, 1.0])


@nb.njit(inline="always")
def scaled(x):
    x[0] *= SCALE[0]; x[1] *= SCALE[1]; x[2] *= SCALE[2]


@nb.njit
def texture(board, n):
    rc = np.zeros(9, dtype=np.int64)
    sc = np.zeros(4, dtype=np.int64)
    for i in range(n):
        rc[board[i] >> 2] += 1
        sc[board[i] & 3] += 1
    paired = 1 if rc.max() >= 2 else 0
    ms = sc.max()
    if n == 3:
        return (0 if ms == 1 else (1 if ms == 2 else 2)) * 2 + paired
    return (1 if ms >= 3 else 0) * 2 + paired


@nb.njit
def river_bucket(hs):
    b = 0
    while b < len(RIVER_EDGES) and hs >= RIVER_EDGES[b]:
        b += 1
    return b


@nb.njit
def deal_info(t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, cards, bk, tex, eq):
    """Fill bk/tex/eq for a 15-card deal; returns +1/0/-1 for player 0."""
    h = np.zeros((2, 5), dtype=np.int64)
    for i in range(5):
        h[0, i] = cards[i]; h[1, i] = cards[5 + i]
    h[0].sort(); h[1].sort()
    board = np.zeros(5, dtype=np.int64)
    for i in range(5):
        board[i] = cards[10 + i]
    sb = np.sort(board)
    x = np.zeros(3)
    tex[1] = texture(board, 3); tex[2] = texture(board, 4); tex[3] = texture(board, 5)
    pids = np.zeros(10, dtype=np.int64)
    s = np.zeros(2, dtype=np.int64)
    for p in range(2):
        cid = canon[CN[h[p, 0], 1] + CN[h[p, 1], 2] + CN[h[p, 2], 3] + CN[h[p, 3], 4] + CN[h[p, 4], 5]]
        bk[p, 0] = pre_bk[cid]; eq[p, 0] = pre_eq[cid]
        feat_runouts(t, canon, hst, lv, i1, i2, h[p], board, 3, x)
        eq[p, 1] = x[0]; scaled(x); bk[p, 1] = nearest(cf, x)
        feat_runouts(t, canon, hst, lv, i1, i2, h[p], board, 4, x)
        eq[p, 2] = x[0]; scaled(x); bk[p, 2] = nearest(ct, x)
        hand_pids(h[p], pids)
        s[p] = best_score(t, pids, sb)
        l = level_of(canon, hst, sb, s[p])
        eq[p, 3] = lv[l]; bk[p, 3] = river_bucket(lv[l])
    return 1 if s[0] > s[1] else (-1 if s[1] > s[0] else 0)


@nb.njit(inline="always")
def draw(rng, deck):
    for i in range(15):
        j = i + rand_int(rng, 36 - i)
        deck[i], deck[j] = deck[j], deck[i]


# ---- rule-based opponents --------------------------------------------------
# params row: kind, margin, raise_thr, pot_thr, bluff, sticky, cbet, pre-spread
K_RANDOM, K_POT, K_HALF, K_STATION, K_RULE = 0, 1, 2, 3, 4


@nb.njit(inline="always")
def rule_action(pr, S, d, m, e, rng):
    kind = int(pr[0])
    if kind == K_RANDOM:
        while True:
            a = rand_int(rng, 4)
            if m[a]:
                return a
    if kind == K_POT:
        return 3 if m[3] else 1
    if kind == K_HALF:
        return 2 if m[2] else (3 if m[3] else 1)
    if kind == K_STATION:
        return 1
    actor = S[d, ACT]
    st = S[d, ST]
    to_call = max(0, S[d, B1 - actor] - S[d, B0 + actor])
    pot = S[d, C0] + S[d, C1]
    if st == 0:
        e = 0.5 + (e - 0.5) * pr[7]     # preflop values cluster near .5: spread them
    can_raise = m[3] == 1
    if to_call == 0:
        if can_raise and e >= pr[3]:
            return 3
        if can_raise and e >= pr[2]:
            return 2 if m[2] else 3
        if can_raise and rand_f(rng) < pr[6]:
            return 2 if m[2] else 3
        return 1
    if can_raise and e >= pr[3]:
        return 3
    if can_raise and e >= pr[2]:
        return 2 if m[2] else 3
    if can_raise and rand_f(rng) < pr[4]:
        return 2 if m[2] else 3
    po = to_call / (pot + to_call)
    if e >= po + pr[1] or rand_f(rng) < pr[5]:
        return 1
    return 0


# ---- MCCFR -----------------------------------------------------------------

@nb.njit(inline="always")
def rm(reg, idx, m, s):
    tot = 0.0
    for a in range(4):
        r = reg[idx, a]
        v = r if (m[a] and r > 0) else 0.0
        s[a] = v
        tot += v
    if tot > 0:
        for a in range(4):
            s[a] /= tot
    else:
        n = float(m[0] + m[1] + m[2] + m[3])
        for a in range(4):
            s[a] = m[a] / n


@nb.njit
def traverse(S, d, tr, win, bk, tex, eq, reg, avg, rng, pr, fixed, M, P, V, wt, thr):
    term = S[d, TERM]
    if term == 1:
        c = S[d, C0 + S[d, LOSER]]
        return float(-c if S[d, LOSER] == tr else c)
    if term == 2:
        w = win if tr == 0 else -win
        return float(w * S[d, C0])
    m = M[d]; s = P[d]; v = V[d]
    legal_mask(S, d, m)
    actor = S[d, ACT]
    if fixed and actor != tr:
        a = rule_action(pr, S, d, m, eq[actor, S[d, ST]], rng)
        step(S, d, a)
        return traverse(S, d + 1, tr, win, bk, tex, eq, reg, avg, rng, pr, fixed, M, P, V, wt, thr)
    idx = info_key(S, d, bk, tex)
    rm(reg, idx, m, s)
    if actor == tr:
        node = 0.0
        for a in range(4):
            if m[a]:
                # Regret-based pruning: skip hopeless actions (fold is terminal, so it is free).
                if a > 0 and reg[idx, a] < thr:
                    m[a] = 0
                    continue
                step(S, d, a)
                v[a] = traverse(S, d + 1, tr, win, bk, tex, eq, reg, avg, rng, pr, fixed, M, P, V, wt, thr)
                node += s[a] * v[a]
        for a in range(4):
            if m[a]:
                reg[idx, a] += v[a] - node
        return node
    for a in range(4):
        avg[idx, a] += s[a] * wt
    x = rand_f(rng)
    a = 0; c = 0.0
    for i in range(4):
        if m[i]:
            a = i
            c += s[i]
            if x < c:
                break
    step(S, d, a)
    return traverse(S, d + 1, tr, win, bk, tex, eq, reg, avg, rng, pr, fixed, M, P, V, wt, thr)


@nb.njit(nogil=True)
def run_chunk(n, seed, t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, reg, avg, pr, fixed_p, wt, prune_thr, prune_p):
    rng = np.array([np.uint64(seed * 2654435761 + 88172645463325252)], dtype=np.uint64)
    for _ in range(8):
        rand_next(rng)
    S = np.zeros((48, NF), dtype=np.int32)
    M = np.zeros((48, 4), dtype=np.int32)
    P = np.zeros((48, 4))
    V = np.zeros((48, 4))
    deck = np.arange(36, dtype=np.int64)
    bk = np.zeros((2, 4), dtype=np.int64)
    tex = np.zeros(4, dtype=np.int64)
    eq = np.zeros((2, 4))
    for i in range(n):
        draw(rng, deck)
        win = deal_info(t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, deck[:15], bk, tex, eq)
        for tr in range(2):
            fixed = rand_f(rng) < fixed_p
            init_state(S, 0)
            thr = prune_thr if rand_f(rng) < prune_p else -1e30
            traverse(S, 0, tr, win, bk, tex, eq, reg, avg, rng, pr, fixed, M, P, V, wt, thr)


# ---- match simulation --------------------------------------------------------

@nb.njit(inline="always")
def pick_table(tab, idx, m, rng):
    tot = 0.0
    for a in range(4):
        if m[a]:
            tot += tab[idx, a]
    a0 = 1
    c = 0.0
    if tot <= 0:
        n = float(m[0] + m[1] + m[2] + m[3])
        x = rand_f(rng) * n
        for a in range(4):
            if m[a]:
                a0 = a
                c += 1
                if x < c:
                    return a
        return a0
    x = rand_f(rng) * tot
    for a in range(4):
        if m[a]:
            a0 = a
            c += tab[idx, a]
            if x < c:
                return a
    return a0


@nb.njit(nogil=True)
def sim(n, seed, t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, tab_a, pr_b, tab_b, b_is_table, out):
    """A (table) vs B (rule or table) on duplicate deals.

    Each deal is played with A holding each of the two hands, with the hand
    pairing matching the tournament's seat swap. out[i] = A's bb on deal i,
    summed over both halves (the quantity the judge averages).
    """
    rng = np.array([np.uint64(seed * 2654435761 + 1140071481932319848)], dtype=np.uint64)
    for _ in range(8):
        rand_next(rng)
    S = np.zeros((48, NF), dtype=np.int32)
    m = np.zeros(4, dtype=np.int32)
    deck = np.arange(36, dtype=np.int64)
    bk = np.zeros((2, 4), dtype=np.int64)
    tex = np.zeros(4, dtype=np.int64)
    eq = np.zeros((2, 4))
    for i in range(n):
        draw(rng, deck)
        win = deal_info(t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, deck[:15], bk, tex, eq)
        total = 0.0
        # The button stays on the same cards in both halves (seat 0's cards are
        # the button's on even deals); only who holds them changes.
        for a_seat in range(2):
            init_state(S, 0)
            d = 0
            while S[d, TERM] == 0:
                legal_mask(S, d, m)
                actor = S[d, ACT]
                if actor == a_seat:
                    a = pick_table(tab_a, info_key(S, d, bk, tex), m, rng)
                elif b_is_table:
                    a = pick_table(tab_b, info_key(S, d, bk, tex), m, rng)
                else:
                    a = rule_action(pr_b, S, d, m, eq[actor, S[d, ST]], rng)
                step(S, d, a)
                d += 1
            if S[d, TERM] == 1:
                r = S[d, C0 + S[d, LOSER]]
                net = -r if S[d, LOSER] == a_seat else r
            else:
                net = win * S[d, C0] * (1 if a_seat == 0 else -1)
            total += net
        out[i] = total / 2.0   # chips -> bb


@nb.njit(inline="always")
def opp_ctx(S, d):
    """Coarse situation for opponent profiling; matches brain.opp_context."""
    actor = S[d, ACT]
    tc = max(0, S[d, B1 - actor] - S[d, B0 + actor])
    if S[d, RS] >= 2:
        fc = 3
    elif tc == 0:
        fc = 0
    else:
        pot = S[d, C0] + S[d, C1]
        fc = 1 if tc * 5 <= (pot - tc) * 3 else 2
    return S[d, ST] * 4 + fc


@nb.njit(nogil=True)
def collect(n, seed, t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, tab_a, pr_b, tab_b, b_is_table, counts):
    """counts[ctx, action] of opponent B's actions while playing table A."""
    rng = np.array([np.uint64(seed * 2654435761 + 7046029254386353087)], dtype=np.uint64)
    for _ in range(8):
        rand_next(rng)
    S = np.zeros((48, NF), dtype=np.int32)
    m = np.zeros(4, dtype=np.int32)
    deck = np.arange(36, dtype=np.int64)
    bk = np.zeros((2, 4), dtype=np.int64)
    tex = np.zeros(4, dtype=np.int64)
    eq = np.zeros((2, 4))
    for i in range(n):
        draw(rng, deck)
        deal_info(t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, deck[:15], bk, tex, eq)
        for a_seat in range(2):
            init_state(S, 0)
            d = 0
            while S[d, TERM] == 0:
                legal_mask(S, d, m)
                actor = S[d, ACT]
                if actor == a_seat:
                    a = pick_table(tab_a, info_key(S, d, bk, tex), m, rng)
                else:
                    if b_is_table:
                        a = pick_table(tab_b, info_key(S, d, bk, tex), m, rng)
                    else:
                        a = rule_action(pr_b, S, d, m, eq[actor, S[d, ST]], rng)
                    counts[opp_ctx(S, d), a] += 1
                step(S, d, a)
                d += 1


def load():
    """Load tables + abstraction into one dict of arrays."""
    z = np.load(os.path.join(CACHE, "tables.npz"))
    lv = np.concatenate(([0.025], (z["levels"] + np.append(z["levels"][1:], 1.0)) / 2))
    i1 = int(np.searchsorted(z["levels"], HI1 - 1e-9)) + 1
    i2 = int(np.searchsorted(z["levels"], HI2 - 1e-9)) + 1
    out = dict(t=z["t"], canon=z["canon"], hst=z["hs"], lv=lv, i1=i1, i2=i2)
    p = os.path.join(CACHE, "abstraction.npz")
    if os.path.exists(p):
        a = np.load(p)
        out.update(pre_bk=a["pre_bk"], pre_eq=a["pre_eq"], cf=a["cf"], ct=a["ct"])
    return out


def args(T):
    return (T["t"], T["canon"], T["hst"], T["lv"], T["i1"], T["i2"],
            T["pre_bk"], T["pre_eq"], T["cf"], T["ct"])


ARCHETYPES = {
    #            kind  margin raise pot   bluff sticky cbet  pre-spread
    "random":   (0,    0,     0,    0,    0,    0,     0,    0),
    "pot":      (1,    0,     0,    0,    0,    0,     0,    0),
    "half":     (2,    0,     0,    0,    0,    0,     0,    0),
    "station":  (3,    0,     0,    0,    0,    0,     0,    0),
    "nit":      (4,    0.10,  0.82, 0.90, 0.00, 0.00,  0.03, 2.5),
    "tag":      (4,    0.03,  0.72, 0.85, 0.03, 0.03,  0.10, 2.5),
    "lag":      (4,   -0.05,  0.60, 0.78, 0.15, 0.10,  0.30, 2.5),
    "loose":    (4,   -0.25,  0.80, 0.90, 0.02, 0.25,  0.05, 2.5),
    "fitfold":  (4,    0.20,  0.70, 0.85, 0.05, 0.00,  0.60, 2.5),
    "potodds":  (4,    0.00,  0.65, 0.80, 0.00, 0.00,  0.00, 2.5),
}
