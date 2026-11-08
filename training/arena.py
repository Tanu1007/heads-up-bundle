"""
arena.py - plays brain.py against opponents through game.py, in-process.

Same engine as the judge minus HTTP, so it is fast enough for many matches.
Every opponent here is hand-written, and the "mc_*" bots share no code with
the training pipeline: they estimate equity by Monte Carlo with omaha_score,
the way the guide suggests. They stand in for the other teams.

    python training/arena.py --deals 1000 --seeds 4 --opps mc_tight,mc_loose
"""
import argparse
import concurrent.futures as cf
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import brain  # noqa: E402
from evaluator import get_table, omaha_score  # noqa: E402
from game import Action, Game, Street, make_deal  # noqa: E402

A = Action


class Bot:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, obs):
        raise NotImplementedError

    def end(self, res):
        pass


class RandomBot(Bot):
    def act(self, obs):
        return self.rng.choice(obs.legal_actions)


class AlwaysRaise(Bot):
    kind = A.RAISE_POT

    def act(self, obs):
        return self.kind if self.kind in obs.legal_actions else A.CHECK_CALL


class AlwaysHalf(AlwaysRaise):
    kind = A.RAISE_HALF


class Station(Bot):
    def act(self, obs):
        return A.CHECK_CALL


class CheckFold(Bot):
    def act(self, obs):
        return A.CHECK_CALL if obs.to_call == 0 else A.FOLD


def mc_equity(obs, rng, n):
    """Monte Carlo equity against one uniformly random hand."""
    seen = set(obs.hole_cards) | set(obs.board)
    deck = [c for c in range(36) if c not in seen]
    need = 5 - len(obs.board)
    wins = 0.0
    for _ in range(n):
        s = rng.sample(deck, 5 + need)
        opp, extra = s[:5], s[5:]
        b = list(obs.board) + extra
        me, them = omaha_score(obs.hole_cards, b), omaha_score(opp, b)
        wins += 1.0 if me > them else (0.5 if me == them else 0.0)
    return wins / n


class MCBot(Bot):
    """A typical contest bot: Monte Carlo equity + pot odds + raise thresholds."""
    n = 40
    raise_at = 0.70
    pot_at = 0.82
    margin = 0.03
    bluff = 0.05

    def act(self, obs):
        rng = self.rng
        eq = mc_equity(obs, rng, self.n)
        if obs.street == Street.PREFLOP:
            eq = 0.5 + (eq - 0.5) * 2.5
        tc = obs.to_call
        po = tc / (obs.pot + tc) if tc else 0.0
        legal = obs.legal_actions
        if A.RAISE_POT in legal and eq >= self.pot_at:
            return A.RAISE_POT
        if A.RAISE_HALF in legal and eq >= self.raise_at:
            return A.RAISE_HALF
        if tc == 0:
            if A.RAISE_HALF in legal and rng.random() < self.bluff:
                return A.RAISE_HALF
            return A.CHECK_CALL
        if eq >= po + self.margin:
            return A.CHECK_CALL
        return A.FOLD


class MCTight(MCBot):
    raise_at, pot_at, margin, bluff = 0.75, 0.88, 0.10, 0.02


class MCLoose(MCBot):
    raise_at, pot_at, margin, bluff = 0.62, 0.78, -0.08, 0.20


class MCManiac(MCBot):
    raise_at, pot_at, margin, bluff = 0.50, 0.65, -0.15, 0.40


class MCPassive(MCBot):
    raise_at, pot_at, margin, bluff = 0.92, 0.97, -0.05, 0.0


class MCAdaptive(MCBot):
    """Tightens up against aggression, bluffs more when the opponent folds."""

    def __init__(self, seed):
        super().__init__(seed)
        self.fold_wins = 1
        self.hands = 2

    def act(self, obs):
        self.bluff = min(0.35, 0.05 + 0.5 * max(0.0, self.fold_wins / self.hands - 0.3))
        return super().act(obs)

    def end(self, res):
        self.hands += 1
        if res.my_net > 0 and not res.went_to_showdown:
            self.fold_wins += 1


class Mirror(Bot):
    """The blueprint alone: a strong opponent with no exploitable tendencies."""

    def __init__(self, seed):
        super().__init__(seed)
        self.inner = BrainBot(seed)
        self.inner.b.pick_table = lambda: 0

    def act(self, obs):
        return self.inner.act(obs)

    def end(self, res):
        self.inner.end(res)


class SelfCopy(Mirror):
    """This exact bot, table switching included."""

    def __init__(self, seed):
        Bot.__init__(self, seed)
        self.inner = BrainBot(seed)


class Trap(MCBot):
    """Plays tight to invite an exploit, then turns into a maniac."""
    switch_at = 250

    def __init__(self, seed):
        super().__init__(seed)
        self.tight = MCTight(seed)
        self.wild = MCManiac(seed + 1)
        self.hands = 0

    def act(self, obs):
        return (self.tight if self.hands < self.switch_at else self.wild).act(obs)

    def end(self, res):
        self.hands += 1


class TrapPot(Trap):
    """Same idea, ending with a pot raise at every opportunity."""

    def __init__(self, seed):
        super().__init__(seed)
        self.wild = AlwaysRaise(seed)


OPPONENTS = {
    "mirror": Mirror, "selfcopy": SelfCopy, "trap": Trap, "trap_pot": TrapPot,
    "random": RandomBot, "pot": AlwaysRaise, "half": AlwaysHalf, "station": Station,
    "checkfold": CheckFold, "mc": MCBot, "mc_tight": MCTight, "mc_loose": MCLoose,
    "mc_maniac": MCManiac, "mc_passive": MCPassive, "mc_adaptive": MCAdaptive,
}


class BrainBot:
    def __init__(self, seed):
        brain.bind_actions(Action)
        self.b = brain.Brain(seed)
        self.elapsed = 0.0
        self.calls = 0

    def act(self, obs):
        t = time.perf_counter()
        a = Action(self.b.choose(obs))
        self.elapsed += time.perf_counter() - t
        self.calls += 1
        assert a in obs.legal_actions, (a, obs)
        return a

    def end(self, res):
        t = time.perf_counter()
        self.b.hand_over(res)
        self.elapsed += time.perf_counter() - t


def half(mine, theirs, seed, deals, my_seat):
    nets = []
    for i in range(deals):
        game = Game(make_deal(seed, i), button_seat=i % 2, hand_index=i)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            game.apply((mine if seat == my_seat else theirs).act(obs))
        mine.end(game.hand_result(my_seat))
        theirs.end(game.hand_result(1 - my_seat))
        nets.append(game.result().nets[my_seat] / 2.0)
    return nets


def match(args):
    name, seed, deals = args
    get_table()
    cls = OPPONENTS[name]
    m1, m2 = BrainBot(seed * 7 + 1), BrainBot(seed * 7 + 2)
    n1 = half(m1, cls(seed * 11 + 1), seed, deals, 0)
    n2 = half(m2, cls(seed * 11 + 2), seed, deals, 1)
    per = [a + b for a, b in zip(n1, n2)]
    mean = sum(per) / deals
    sd = math.sqrt(sum((x - mean) ** 2 for x in per) / (deals - 1))
    se = sd / math.sqrt(deals)
    t = max(m1.elapsed, m2.elapsed)
    tabs = (m1.b.active, m2.b.active, dict(m1.b.net_with), dict(m2.b.net_with))
    return name, seed, mean, se, t, max(m1.calls, m2.calls), tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--opps", default=",".join(OPPONENTS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    jobs = [(o, a.first_seed + s, a.deals) for o in a.opps.split(",") for s in range(a.seeds)]
    results = {}
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(match, jobs):
            results.setdefault(r[0], []).append(r)
            if a.verbose:
                print(r[:6], "tables", r[6], flush=True)
    print(f"{'opponent':14}{'bb/deal':>9}{'+-1.96se':>10}{'wins':>7}{'chops':>7}{'losses':>7}{'think s':>9}")
    for name, rs in results.items():
        mean = sum(r[2] for r in rs) / len(rs)
        ci = sum(1.96 * r[3] for r in rs) / len(rs)
        w = sum(r[2] - 1.96 * r[3] > 0 for r in rs)
        l = sum(r[2] + 1.96 * r[3] < 0 for r in rs)
        print(f"{name:14}{mean:+9.3f}{ci:10.3f}{w:7d}{len(rs) - w - l:7d}{l:7d}{max(r[4] for r in rs):9.2f}", flush=True)


if __name__ == "__main__":
    main()
