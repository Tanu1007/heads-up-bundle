"""Check the numba game engine against game.py over many random hands."""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import engine as E  # noqa: E402
from game import Action, Game, make_deal  # noqa: E402

import numba as nb  # noqa: E402


@nb.njit
def apply_line(S, d, a):
    E.step(S, d, a)


@nb.njit
def legal(S, d, m):
    E.legal_mask(S, d, m)


@nb.njit
def init(S):
    E.init_state(S, 0)


def main(n=20000):
    rng = random.Random(1)
    S = np.zeros((64, E.NF), dtype=np.int32)
    m = np.zeros(4, dtype=np.int32)
    streets = 0
    for h in range(n):
        g = Game(make_deal("parity", h), button_seat=0)
        init(S)
        d = 0
        weights = rng.choice([(1, 1, 1), (1, 3, 3), (3, 1, 1), (1, 1, 5), (1, 5, 1)])
        while not g.is_terminal:
            assert S[d, E.TERM] == 0
            assert int(S[d, E.ST]) == int(g.street), (h, d)
            assert int(S[d, E.ACT]) == g.to_act
            assert [int(S[d, E.B0]), int(S[d, E.B1])] == g.bets
            assert [int(S[d, E.C0]), int(S[d, E.C1])] == g.contributed
            legal(S, d, m)
            leg = g.legal_actions()
            assert (m[0] == 1) == (Action.FOLD in leg)
            assert (m[3] == 1) == (Action.RAISE_POT in leg)
            if m[2]:
                assert Action.RAISE_HALF in leg
                assert g.raise_to(Action.RAISE_HALF) != g.raise_to(Action.RAISE_POT)
            elif m[3]:
                assert g.raise_to(Action.RAISE_HALF) == g.raise_to(Action.RAISE_POT)
            opts = [1]
            if m[0]:
                opts.append(0)
            if m[2]:
                opts.append(2)
            if m[3]:
                opts.append(3)
            w = {0: 1, 1: weights[0], 2: weights[1], 3: weights[2]}
            a = rng.choices(opts, [w[o] for o in opts])[0]
            g.apply(Action(a))
            apply_line(S, d, a)
            d += 1
        r = g.result()
        if r.went_to_showdown:
            assert S[d, E.TERM] == 2
            streets += 1
        else:
            assert S[d, E.TERM] == 1
            assert S[d, E.LOSER] == 1 - r.winner_seat
        assert [int(S[d, E.C0]), int(S[d, E.C1])] == g.contributed
    print("parity OK on", n, "hands;", streets, "showdowns")


if __name__ == "__main__":
    main()
