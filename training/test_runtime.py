"""brain.py must see exactly what the trainer saw: same buckets, same keys."""
import os
import random
import sys
import time

import numba as nb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import brain  # noqa: E402
import engine as E  # noqa: E402
from game import Action, Game, make_deal  # noqa: E402


@nb.njit
def _deal(t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, cards, bk, tex, eq):
    return E.deal_info(t, canon, hst, lv, i1, i2, pre_bk, pre_eq, cf, ct, cards, bk, tex, eq)


@nb.njit
def _key(S, d, bk, tex):
    return E.info_key(S, d, bk, tex)


@nb.njit
def _step(S, d, a):
    E.step(S, d, a)


@nb.njit
def _init(S):
    E.init_state(S, 0)


def main(n=3000):
    T = E.load()
    a = E.args(T)
    tb = brain.load()
    S = np.zeros((64, E.NF), dtype=np.int32)
    bk = np.zeros((2, 4), dtype=np.int64)
    tex = np.zeros(4, dtype=np.int64)
    eq = np.zeros((2, 4))
    rng = random.Random(3)
    checked = 0
    for h in range(n):
        d = make_deal("rt", h)
        cards = np.array(list(d.hole_cards[0]) + list(d.hole_cards[1]) + list(d.board), dtype=np.int64)
        win = _deal(*a, cards, bk, tex, eq)
        g = Game(d, button_seat=0)
        _init(S)
        depth = 0
        while not g.is_terminal:
            seat = g.to_act
            obs = g.observation(seat)
            st = int(obs.street)
            b, e = brain.hand_bucket(tb, st, obs.hole_cards, obs.board)
            assert b == bk[seat, st], (h, st, seat, b, bk[seat, st])
            assert abs(e - eq[seat, st]) < 1e-4, (e, eq[seat, st])
            r = brain.replay_history(obs)
            t = brain.texture(obs.board) if st else 0
            if st:
                assert t == tex[st]
            assert brain.info_key(r, b, t) == _key(S, depth, bk, tex)
            assert brain.opp_context(r) >= 0
            checked += 1
            leg = list(obs.legal_actions)
            act = rng.choice(leg + [Action.CHECK_CALL] * 2)
            sym = int(act)
            if act == Action.RAISE_HALF and g.raise_to(act) == g.raise_to(Action.RAISE_POT):
                sym = 3
            g.apply(act)
            _step(S, depth, sym)
            depth += 1
    print("runtime parity OK:", checked, "decisions on", n, "hands")
    # cost of each street's bucket
    for st, nb_ in ((1, 3), (2, 4), (3, 5)):
        hs = [make_deal("t", i) for i in range(300)]
        t0 = time.perf_counter()
        for dd in hs:
            brain.hand_bucket(tb, st, dd.hole_cards[0], dd.board[:nb_])
        print("street", st, "bucket us:", round((time.perf_counter() - t0) / 300 * 1e6))


if __name__ == "__main__":
    main()
