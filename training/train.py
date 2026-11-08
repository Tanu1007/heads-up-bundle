"""
train.py - step 3: run MCCFR.

    python training/train.py blueprint --minutes 40
    python training/train.py resp_station --opp station --fixed 0.8 --minutes 6 --warm blueprint

Regrets are shared between threads without locks; the occasional lost update
does no harm. Between chunks both tables are discounted so that the average
weights iteration t in proportion to t (linear CFR).
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402


def evaluate(T, tab, n=40000, seed=5, names=None, other=None):
    out = np.zeros(n)
    res = {}
    a = E.args(T)
    dummy = np.zeros((1, 4), dtype=np.float32)
    for name in (names or E.ARCHETYPES):
        pr = np.array(E.ARCHETYPES[name], dtype=np.float64)
        E.sim(n, seed, *a, tab, pr, dummy, False, out)
        res[name] = (out.mean(), out.std() / np.sqrt(n))
    if other is not None:
        pr = np.zeros(8)
        E.sim(n, seed, *a, tab, pr, other, True, out)
        res["other"] = (out.mean(), out.std() / np.sqrt(n))
    return res


def fmt(res):
    return "  ".join(f"{k} {v[0]:+.2f}" for k, v in res.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--chunk", type=int, default=20000, help="deals per thread per chunk")
    ap.add_argument("--opp", default=None)
    ap.add_argument("--fixed", type=float, default=0.0)
    ap.add_argument("--warm", default=None)
    ap.add_argument("--keep", type=float, default=0.02, help="fraction of warm-start regret kept")
    ap.add_argument("--prune-after", type=float, default=4.0, help="minutes before pruning starts")
    ap.add_argument("--eval-every", type=float, default=300)
    args = ap.parse_args()

    T = E.load()
    a = E.args(T)
    n = E.N_INFOSETS
    print("infosets", n, "memory MB", n * 16 // 2 ** 20, flush=True)
    start_t = 1
    if args.warm:
        z = np.load(os.path.join(E.CACHE, args.warm + "_raw.npz"))
        reg = z["reg"].copy()
        avg = np.zeros((n, 4), dtype=np.float32)
        reg *= args.keep     # keep the shape of the solution, stay able to move
    else:
        reg = np.zeros((n, 4), dtype=np.float32)
        avg = np.zeros((n, 4), dtype=np.float32)
    pr = np.array(E.ARCHETYPES[args.opp] if args.opp else np.zeros(8), dtype=np.float64)

    t0 = time.time()
    last = t0
    it = start_t
    deals = 0
    while time.time() - t0 < args.minutes * 60:
        pruning = (time.time() - t0) / 60 >= args.prune_after
        ths = [threading.Thread(target=E.run_chunk, args=(
            args.chunk, it * 1000 + i, *a, reg, avg, pr, args.fixed, 1.0, -300.0, 0.95 if pruning else 0.0))
            for i in range(args.threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        deals += args.chunk * args.threads
        d = it / (it + 1.0)
        # avg and reg decay together: late iterations count the most.
        reg *= d
        avg *= d
        it += 1
        now = time.time()
        if it % 10 == 0:
            print(f"  t={(now - t0) / 60:.1f}m chunks={it} deals={deals / 1e6:.2f}M rate={deals / (now - t0):.0f}/s", flush=True)
        if now - last >= args.eval_every:
            last = now
            res = evaluate(T, avg)
            print(f"[{(now - t0) / 60:6.1f}m] chunks {it} deals {deals / 1e6:.1f}M  {fmt(res)}", flush=True)
            np.savez(os.path.join(E.CACHE, args.name + "_raw.npz"), reg=reg, avg=avg)
    np.savez(os.path.join(E.CACHE, args.name + "_raw.npz"), reg=reg, avg=avg)
    res = evaluate(T, avg, n=40000)
    touched = (avg.sum(1) > 0).sum()
    print(f"DONE {args.name}: {deals / 1e6:.1f}M deals, {(time.time() - t0) / 60:.1f}m, "
          f"{deals / (time.time() - t0):.0f} deals/s, infosets touched {touched}", flush=True)
    print({k: (round(v[0], 3), round(v[1], 3)) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
