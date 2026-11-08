"""
brain.py - the bot's decision logic (numpy only, no compiled code).

How a decision is made:
  1. Rebuild the betting state from obs.action_history.
  2. Turn the cards into a hand bucket (see training/engine.py for the
     definition; this file must stay in step with it, and
     training/test_runtime.py checks that it does).
  3. Look the situation up in a pre-solved strategy table.

Which table? The default is the near-equilibrium "blueprint". Every action the
opponent takes is also scored against a set of opponent profiles; once one
profile clearly fits, the table trained to exploit that profile takes over.
If that goes badly, or nothing fits, the bot returns to the blueprint.
"""
from __future__ import annotations

import math
import os
import random
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "bot_data")

FOLD, CALL, HALF, POT = 0, 1, 2, 3

# Must match training/engine.py
N_PRE, N_FLOP, N_TURN, N_RIVER = 40, 64, 64, 14
TEX_FLOP, TEX_TURN, TEX_RIVER = 6, 4, 4
N_POTCTX, N_SEQ = 9, 62
OFF_FLOP = 2 * N_PRE * N_SEQ
OFF_TURN = OFF_FLOP + 2 * 3 * N_POTCTX * TEX_FLOP * N_FLOP * N_SEQ
OFF_RIVER = OFF_TURN + 2 * 3 * N_POTCTX * TEX_TURN * N_TURN * N_SEQ
STACK_DEFAULT = 80
RAISE_CAP = 4
RIVER_EDGES = (.30, .50, .65, .75, .82, .87, .91, .94, .96, .975, .985, .992, .997)
_E = [2, 4, 6, 9, 14, 20, 30, 45, 80]
POTCTX = [next(i for i, x in enumerate(_E) if c <= x) for c in range(81)]
SCALE = np.array([1.0, 0.7, 1.0])

# Opponent-model tuning
MIN_ACTIONS = 120        # opponent actions seen before any table switch
MARGIN_PER_ACTION = 0.03 # a profile must beat the blueprint profile by this many nats per action
FIT_LIMIT = 0.16         # ...and sit this close to the best possible fit
STOPLOSS_BB = 60.0       # give up on a profile after losing this much with it

# Time-bank guard. The judge charges wall time per request, so estimate that
# from what we measure here, and thin out the board sampling if it gets tight.
TIME_BANK = 8.0
OVERHEAD_PER_CALL = 0.0004   # JSON/HTTP cost above the free 1 ms (generous)
CHARGE_FACTOR = 1.3          # slack for time we cannot see from in here
SOFT_BUDGET = 6.0            # projected use above this: sample less
HARD_BUDGET = 7.0            # used already above this: sample very little


def _build_seq():
    ids = {(): 0}
    order = [()]
    nxt = []
    i = 0
    while i < len(order):
        h = order[i]
        row = [-1, -1, -1]
        raises = sum(1 for x in h if x != 0)
        for sym in (0, 1, 2):
            if sym == 0:
                if len(h) != 0:
                    continue
            elif raises >= 4:
                continue
            h2 = h + (sym,)
            if h2 not in ids:
                ids[h2] = len(order)
                order.append(h2)
            row[sym] = ids[h2]
        nxt.append(row)
        i += 1
    return nxt


NEXT_SEQ = _build_seq()

_CN = np.zeros((37, 6), dtype=np.int64)
for _n in range(37):
    _CN[_n, 0] = 1
    for _k in range(1, 6):
        _CN[_n, _k] = 0 if _n == 0 else _CN[_n - 1, _k - 1] + _CN[_n - 1, _k]
CN = _CN
CN1, CN2, CN3, CN4, CN5 = (CN[:, k] for k in range(1, 6))
_CNL = CN.tolist()
TRI = np.array([(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
                (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)])
PAIRS5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]

# The fixed subset of (turn, river) completions used on the flop.
_k = 0
_si, _sj = [], []
for _i in range(28):
    for _j in range(_i + 1, 28):
        _k += 1
        if _k % 4 == 0:
            _si.append(_i)
            _sj.append(_j)
SEL_I, SEL_J = np.array(_si), np.array(_sj)


class Tables:
    """All read-only data, loaded once when the server starts."""

    def __init__(self):
        z = np.load(os.path.join(DATA, "tables.npz"))
        self.t = z["t"]
        self.canon = z["canon"].astype(np.int64)
        self.hst = z["hst"]
        self.lv = z["lv"]
        self.i1 = int(z["i1"])
        self.i2 = int(z["i2"])
        self.pre_bk = z["pre_bk"]
        self.pre_eq = z["pre_eq"]
        self.cf = z["cf"]
        self.ct = z["ct"]
        p = np.load(os.path.join(DATA, "policies.npz"))
        self.names = [str(x) for x in p["names"]]
        self.tabs = [p["tab_" + n] for n in self.names]
        self.models = np.log(p["models"])  # [profile, context, action]
        self.resp = [int(x) for x in p["resp"]]  # table used against each profile
        self.model_names = [str(x) for x in p["model_names"]]


T: Tables | None = None


def load() -> Tables:
    global T
    if T is None:
        T = Tables()
    return T


# ---- hand features ----------------------------------------------------------

def _levels(tb, pids, boards):
    """HS level of a hand on each sorted 5-card board in `boards` (K x 5)."""
    x = boards[:, TRI[:, 0]]
    y = boards[:, TRI[:, 1]]
    z = boards[:, TRI[:, 2]]
    tid = CN3[z] + CN2[y] + x
    scores = tb.t[pids[None, :, None], tid[:, None, :]].reshape(len(boards), -1).max(axis=1)
    colex = (CN1[boards[:, 0]] + CN2[boards[:, 1]] + CN3[boards[:, 2]]
             + CN4[boards[:, 3]] + CN5[boards[:, 4]])
    thr = tb.hst[tb.canon[colex]]
    return (thr <= scores[:, None]).sum(axis=1)


def hand_pids(hole):
    out = []
    for i, j in PAIRS5:
        a, b = hole[i], hole[j]
        if a > b:
            a, b = b, a
        out.append(_CNL[b][2] + a)
    return np.array(out)


def _rest(hole, board):
    used = set(hole) | set(board)
    return np.array([c for c in range(36) if c not in used])


def features(tb, hole, board, stride=1):
    """(mean HS, P(HS >= .90), P(HS >= .97)) over board completions.

    stride > 1 uses every stride-th completion (cheaper, slightly noisier).
    """
    pids = hand_pids(hole)
    rest = _rest(hole, board)
    n = len(board)
    if n == 3:
        boards = np.empty((len(SEL_I), 5), dtype=np.int64)
        boards[:, 0:3] = board
        boards[:, 3] = rest[SEL_I]
        boards[:, 4] = rest[SEL_J]
    else:
        boards = np.empty((len(rest), 5), dtype=np.int64)
        boards[:, 0:4] = board
        boards[:, 4] = rest
    boards = boards[::stride]
    boards.sort(axis=1)
    lev = _levels(tb, pids, boards)
    return (float(tb.lv[lev].mean()), float((lev >= tb.i1).mean()), float((lev >= tb.i2).mean()))


def river_hs(tb, hole, board):
    b = np.array([sorted(board)], dtype=np.int64)
    return float(tb.lv[_levels(tb, hand_pids(hole), b)[0]])


def river_bucket(hs):
    b = 0
    for e in RIVER_EDGES:
        if hs >= e:
            b += 1
        else:
            break
    return b


def texture(board):
    rc = [0] * 9
    sc = [0] * 4
    for c in board:
        rc[c >> 2] += 1
        sc[c & 3] += 1
    paired = 1 if max(rc) >= 2 else 0
    ms = max(sc)
    if len(board) == 3:
        return (0 if ms == 1 else (1 if ms == 2 else 2)) * 2 + paired
    return (1 if ms >= 3 else 0) * 2 + paired


def hand_bucket(tb, st, hole, board, stride=1):
    """Returns (bucket, equity-like strength)."""
    if st == 0:
        h = sorted(hole)
        cid = tb.canon[_CNL[h[0]][1] + _CNL[h[1]][2] + _CNL[h[2]][3] + _CNL[h[3]][4] + _CNL[h[4]][5]]
        return int(tb.pre_bk[cid]), float(tb.pre_eq[cid])
    if st == 3:
        hs = river_hs(tb, hole, board)
        return river_bucket(hs), hs
    f = features(tb, hole, board, stride)
    x = np.array(f) * SCALE
    c = tb.cf if st == 1 else tb.ct
    return int(((c - x) ** 2).sum(axis=1).argmin()), f[0]


# ---- betting state ----------------------------------------------------------

class Replay:
    """Replays a hand's action history with the rules of game.py."""

    __slots__ = ("st", "seq", "actor", "bets", "c", "rs", "li", "acted", "la", "pa", "pctx",
                 "stack", "over")

    def __init__(self, stack=STACK_DEFAULT):
        self.st = 0
        self.seq = 0
        self.actor = 0          # 0 = button
        self.bets = [1, 2]
        self.c = [1, 2]
        self.rs = 0
        self.li = 2
        self.acted = [False, False]
        self.la = -1
        self.pa = -1
        self.pctx = 0
        self.stack = stack
        self.over = False

    def raise_total(self, a):
        me = self.actor
        my_bet, opp_bet = self.bets[me], self.bets[1 - me]
        pot = self.c[0] + self.c[1]
        to_call = max(0, opp_bet - my_bet)
        all_in = my_bet + (self.stack - self.c[me])
        pot_to = opp_bet + pot + to_call
        half_to = opp_bet + (pot + to_call) // 2
        target = half_to if a == HALF else pot_to
        return min(max(target, opp_bet + max(self.li, 2)), pot_to, all_in)

    def to_call(self):
        return max(0, self.bets[1 - self.actor] - self.bets[self.actor])

    def can_raise(self):
        me = self.actor
        return (self.rs < RAISE_CAP and self.stack - self.c[me] > self.to_call()
                and self.stack - self.c[1 - me] > 0)

    def apply(self, a):
        me = self.actor
        opp = 1 - me
        if a == FOLD:
            self.over = True
            return
        if a == CALL:
            amt = min(self.to_call(), self.stack - self.c[me])
            self.bets[me] += amt
            self.c[me] += amt
            self.acted[me] = True
            if self.acted[0] and self.acted[1] and self.bets[0] == self.bets[1]:
                if self.st == 3 or self.c[0] >= self.stack or self.c[1] >= self.stack:
                    self.over = True
                else:
                    self.st += 1
                    self.seq = 0
                    self.actor = 1
                    self.bets = [0, 0]
                    self.acted = [False, False]
                    self.rs = 0
                    self.li = 2
                    self.pa = self.la
                    self.la = -1
                    self.pctx = POTCTX[min(self.c[0], 80)]
            else:
                self.actor = opp
                self.seq = NEXT_SEQ[self.seq][0]
            return
        # A half raise that equals the pot raise is the same action in the tree.
        if a == HALF and self.raise_total(HALF) == self.raise_total(POT):
            a = POT
        tot = self.raise_total(a)
        amt = tot - self.bets[me]
        self.li = tot - self.bets[opp]
        self.rs += 1
        self.bets[me] += amt
        self.c[me] += amt
        self.acted[opp] = False
        self.acted[me] = True
        self.la = me
        self.actor = opp
        self.seq = NEXT_SEQ[self.seq][a - 1]


def replay_history(obs):
    """Replay obs.action_history; the first actor is always the button."""
    r = Replay()
    for h in obs.action_history:
        r.apply(int(h.action))
    return r


def info_key(r, bucket, tex):
    st, actor, seq = r.st, r.actor, r.seq
    if st == 0:
        return (actor * N_PRE + bucket) * N_SEQ + seq
    ag = 0 if r.pa < 0 else (1 if r.pa == actor else 2)
    ctx = (actor * 3 + ag) * N_POTCTX + r.pctx
    if st == 1:
        return OFF_FLOP + ((ctx * TEX_FLOP + tex) * N_FLOP + bucket) * N_SEQ + seq
    if st == 2:
        return OFF_TURN + ((ctx * TEX_TURN + tex) * N_TURN + bucket) * N_SEQ + seq
    return OFF_RIVER + ((ctx * TEX_RIVER + tex) * N_RIVER + bucket) * N_SEQ + seq


def opp_context(r):
    """Coarse situation used for opponent profiling (16 values).

    Must match engine.opp_ctx.
    """
    tc = r.to_call()
    if r.rs >= 2:
        fc = 3
    elif tc == 0:
        fc = 0
    else:
        pot = r.c[0] + r.c[1]
        fc = 1 if tc * 5 <= (pot - tc) * 3 else 2
    return r.st * 4 + fc


# ---- the bot ------------------------------------------------------------------

class Brain:
    """All state for one half; lives in the agent's memory dict."""

    def __init__(self, rng_seed=None):
        self.tb = load()
        self.rng = random.Random(rng_seed)
        self.n_models = len(self.tb.model_names)
        self.ll = [0.0] * self.n_models     # log-likelihood of each profile
        self.counts = [[0] * 4 for _ in range(16)]
        self.n_seen = 0
        self.active = 0                     # index into tb.tabs (0 = blueprint)
        self.banned = set()
        self.net_with = {}                  # chips won with each table
        self.cache_key = None
        self.cache_val = (0, 0.5)
        self.seen_upto = {}                 # hand_index -> opp actions already scored
        self.last_hist = []                 # actions of the current hand, mine included
        self.last_hand = -1
        self.last_button = False
        self.hands = 0
        self.think = 0.0
        self.calls = 0
        self.stride = 1

    # -- decision ---------------------------------------------------------------
    def bucket(self, obs, st):
        key = (obs.hand_index, st)
        if key != self.cache_key:
            self.cache_key = key
            self.cache_val = hand_bucket(self.tb, st, obs.hole_cards, obs.board, self.stride)
        return self.cache_val

    def score_opponent(self, obs, hist):
        """Score opponent actions we have not scored yet this hand."""
        done = self.seen_upto.get(obs.hand_index, 0)
        if done >= len(hist):
            return
        r = Replay()
        for i, a in enumerate(hist):
            if i >= done and not a[0]:
                self.observe(r, a[1])
            r.apply(a[1])
        self.seen_upto = {obs.hand_index: len(hist)}

    def observe(self, r, a):
        ctx = opp_context(r)
        if a == HALF and r.raise_total(HALF) == r.raise_total(POT):
            a = POT
        m = self.tb.models
        for i in range(self.n_models):
            self.ll[i] += float(m[i, ctx, a])
        self.counts[ctx][a] += 1
        self.n_seen += 1

    def saturated_ll(self):
        tot = 0.0
        for row in self.counts:
            s = sum(row)
            if s:
                for c in row:
                    if c:
                        tot += c * math.log(c / s)
        return tot

    def pick_table(self):
        if self.n_seen < MIN_ACTIONS or self.n_models < 2:
            return 0
        order = sorted(range(self.n_models), key=lambda i: -self.ll[i])
        best = order[0]
        if self.tb.resp[best] == 0:
            return 0
        per = 1.0 / self.n_seen
        lead = (self.ll[best] - self.ll[0]) * per       # profile 0 is the blueprint itself
        gap = (self.saturated_ll() - self.ll[best]) * per
        if lead < MARGIN_PER_ACTION or gap > FIT_LIMIT:
            return 0
        tab = self.tb.resp[best]
        return 0 if tab in self.banned else tab

    def used_estimate(self):
        return self.think * CHARGE_FACTOR + self.calls * OVERHEAD_PER_CALL

    def update_pace(self, hands_left):
        used = self.used_estimate()
        projected = used
        if self.hands >= 30:
            projected = used + used / self.hands * hands_left
        if used > HARD_BUDGET:
            self.stride = 8
        elif projected > SOFT_BUDGET:
            self.stride = 3
        else:
            self.stride = 1

    def choose(self, obs):
        start = time.perf_counter()
        try:
            return self._choose(obs)
        finally:
            self.think += time.perf_counter() - start
            self.calls += 1

    def _choose(self, obs):
        st = int(obs.street)
        hist = [(h.is_me, int(h.action)) for h in obs.action_history]
        self.score_opponent(obs, hist)
        r = Replay()
        for _, a in hist:
            r.apply(a)
        bucket, eq = self.bucket(obs, st)
        if st != r.st:              # should never happen; do not trust the replay
            raise ValueError("street mismatch")
        actor = 0 if obs.is_button else 1
        if actor != r.actor:
            raise ValueError("actor mismatch")
        tex = texture(obs.board) if st else 0
        key = info_key(r, bucket, tex)
        legal = [int(a) for a in obs.legal_actions]
        distinct_half = (HALF in legal and POT in legal
                         and obs.raise_to[_ACT[HALF]] != obs.raise_to[_ACT[POT]])
        tab = self.tb.tabs[self.active]
        row = tab[key]
        w = [0.0, 0.0, 0.0, 0.0]
        for a in legal:
            if a == HALF and not distinct_half:
                continue
            w[a] = float(row[a])
        if sum(w) <= 0 and self.active:
            row = self.tb.tabs[0][key]
            for a in legal:
                if a != HALF or distinct_half:
                    w[a] = float(row[a])
        if sum(w) <= 0:
            a = self.heuristic(obs, eq, legal)
        else:
            x = self.rng.random() * sum(w)
            a = CALL
            for i in range(4):
                if w[i] > 0:
                    a = i
                    x -= w[i]
                    if x < 0:
                        break
        self.last_hist = hist + [(True, a)]
        self.last_hand = obs.hand_index
        self.last_button = obs.is_button
        return a

    def heuristic(self, obs, eq, legal):
        """Only for situations the training never reached."""
        tc = obs.to_call
        po = tc / (obs.pot + tc) if tc else 0.0
        if eq > 0.85 and POT in legal:
            return POT
        if tc == 0 or eq >= po + 0.08:
            return CALL
        return FOLD if FOLD in legal else CALL

    # -- after the hand -----------------------------------------------------------
    def hand_over(self, res):
        start = time.perf_counter()
        try:
            self._hand_over(res)
        finally:
            self.think += time.perf_counter() - start
        self.update_pace(max(0, 1000 - self.hands))

    def _hand_over(self, res):
        self.hands += 1
        self.net_with[self.active] = self.net_with.get(self.active, 0) + res.my_net
        if self.last_hand == res.hand_index:
            # The opponent may have closed the hand after our last action.
            r = Replay()
            for _, a in self.last_hist:
                r.apply(a)
            if not r.over:
                me_button = self.last_button
                opp_seat = 1 if me_button else 0
                if r.actor == opp_seat:
                    self.observe(r, CALL if res.went_to_showdown else FOLD)
        else:
            # We never acted: the opponent, on the button, folded straight away.
            self.observe(Replay(), FOLD)
        if self.active and self.net_with[self.active] < -STOPLOSS_BB * 2:
            self.banned.add(self.active)
        self.active = self.pick_table()


_ACT = {}


def bind_actions(action_enum):
    """Let brain.py index obs.raise_to without importing game.py twice."""
    _ACT[HALF] = action_enum.RAISE_HALF
    _ACT[POT] = action_enum.RAISE_POT
