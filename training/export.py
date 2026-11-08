"""
export.py - step 4: write the runtime data files into bot_data/.

    python training/export.py blueprint resp_station resp_nit ...

Table 0 is the blueprint. A table named resp_<x> answers archetype <x>.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "bot_data")
PURIFY = 0.04     # actions below this probability are dropped (mostly noise)


def to_table(avg):
    s = avg.sum(1, keepdims=True)
    p = np.divide(avg, s, out=np.zeros_like(avg), where=s > 0)
    p[p < PURIFY] = 0
    s = p.sum(1, keepdims=True)
    p = np.divide(p, s, out=np.zeros_like(p), where=s > 0)
    return np.round(p * 255).astype(np.uint8)


def fingerprint(T, a, tab, name, n=60000):
    c = np.zeros((16, 4), dtype=np.int64)
    dummy = np.zeros((1, 4), dtype=np.float32)
    if name == "blueprint":
        E.collect(n, 3, *a, tab, np.zeros(8), tab, True, c)
    else:
        E.collect(n, 3, *a, tab, np.array(E.ARCHETYPES[name], dtype=np.float64), dummy, False, c)
    return c


def main(names):
    T = E.load()
    a = E.args(T)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT, "tables.npz"), t=T["t"], canon=T["canon"], hst=T["hst"],
        lv=T["lv"], i1=T["i1"], i2=T["i2"], pre_bk=T["pre_bk"].astype(np.uint8),
        pre_eq=T["pre_eq"].astype(np.float32), cf=T["cf"], ct=T["ct"])
    tabs = {n: to_table(np.load(os.path.join(E.CACHE, n + "_raw.npz"))["avg"]) for n in names}
    bp_raw = tabs["blueprint"]
    # Profiles: the blueprint first, then every archetype (with or without a table).
    model_names = ["blueprint"] + [k for k in E.ARCHETYPES]
    models = []
    resp = []
    for m in model_names:
        c = fingerprint(T, a, np.asarray(bp_raw, dtype=np.float32), m).astype(np.float64)
        # Smooth: a profile must not rule an action out completely.
        row = c.sum(1, keepdims=True)
        global_p = (c.sum(0, keepdims=True) + 1) / (c.sum() + 4)
        p = (c + 8 * global_p) / (row + 8)
        p = 0.985 * p + 0.015 / 4
        models.append(p)
        key = "resp_" + m
        resp.append(list(names).index(key) if key in tabs else 0)
    np.savez_compressed(
        os.path.join(OUT, "policies.npz"), names=np.array(names), models=np.array(models),
        resp=np.array(resp), model_names=np.array(model_names),
        **{"tab_" + n: t for n, t in tabs.items()})
    for f in os.listdir(OUT):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 2 ** 20, 2), "MB")


if __name__ == "__main__":
    main(sys.argv[1:] or ["blueprint"])
