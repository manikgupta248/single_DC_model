"""
tune_noise_levels.py - tune the Kalman filter for several noise levels of the measured current.

Noise level = standard deviation sigma [A] of Gaussian white noise that is added to the measured current.

Two ways are compared:
  A) one tuning PER noise level    (the noise settings are adapted to one sensor quality)
  B) one ROBUST tuning for all noise levels together (the tuner minimises the average error over all levels)

Every tuned set is then tested at every noise level on test runs that were not used for the tuning.
The table shows the camshaft torque error (RMSE in % of the rms torque, without the first 20 ms).

    python tune_noise_levels.py                      levels 0, 0.02, 0.05, 0.1 A, 50 evaluations (about 15 min)
    python tune_noise_levels.py --noise 0.05 0.2     other levels
    python tune_noise_levels.py --variant interp     Hall variant (default no_interp)
    python tune_noise_levels.py --quick              small data set, few evaluations (a few minutes)

Needs numpy, emotor_cam_move.py and torque_kf.py from the repository.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torque_kf as k


def noisy_copies(runs, sigma, seed0):
    """Copies of the runs in which Gaussian noise (std sigma [A]) is added to the measured current."""
    out = []
    for idx, d in enumerate(runs):
        e = dict(d)
        if sigma > 0:
            rng = np.random.default_rng(seed0 + idx)
            e["i_meas"] = d["Li"] + sigma * rng.standard_normal(len(d["Li"]))
        else:
            e["i_meas"] = d["Li"].copy()
        out.append(e)
    return out


def score(params, variant, runs):
    """Mean camshaft torque error (RMSE in % of the rms torque) over the runs, exact model parameters."""
    vals = []
    for d in runs:
        e = k.estimates(d, variant, params, "nominal")
        vals.append(k.metrics(e["true_cam"], e["est_cam"])["RMSE_pct"])
    return float(np.mean(vals))


def tune_multistart(variant, runs, evals, starts):
    """Run the tuner of torque_kf from several starting points and keep the best result.
    (A single start can get stuck on a poor solution, for example one that ignores the current.)"""
    saved = k.START_GUESS[variant]
    best_v, best = None, float("inf")
    try:
        for i, st in enumerate(starts):
            k.START_GUESS[variant] = tuple(float(x) for x in st)
            v, val = k.tune(variant, runs, evals)
            print("    start %d of %d: objective %.2f %%" % (i + 1, len(starts), val))
            if val < best:
                best_v, best = v, val
    finally:
        k.START_GUESS[variant] = saved
    return best_v, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1], help="noise levels sigma [A]")
    ap.add_argument("--variant", default="no_interp", choices=["interp", "no_interp"])
    ap.add_argument("--evals", type=int, default=50, help="tuning evaluations per tuning (default 50)")
    ap.add_argument("--quick", action="store_true", help="small data set and few evaluations")
    ap.add_argument("--out", default="noise_tuning", help="prefix of the output file / data caches")
    a = ap.parse_args()
    levels = list(a.noise)
    if a.quick:
        train_pwm, train_w0, test_pwm, test_w0 = (0.06, 0.2), (0.0,), (0.1,), (30.0,)
        evals = min(a.evals, 20)
    else:
        train_pwm, train_w0, test_pwm, test_w0 = (0.06, 0.1, 0.15, 0.2), (0.0, 30.0), (0.08, 0.12, 0.18), (15.0, 45.0)
        evals = a.evals

    t0 = time.time()
    print("simulating the runs (without noise) ...")
    train = k.build_dataset(train_pwm, train_w0, 0.0, False, a.out + "_train.pkl", "train")
    test = k.build_dataset(test_pwm, test_w0, 0.0, False, a.out + "_test.pkl", "test")
    print("%d training runs, %d test runs (%.0f s)\n" % (len(train), len(test), time.time() - t0))

    train_sets = {s: noisy_copies(train, s, 100) for s in levels}
    test_sets = {s: noisy_copies(test, s, 5000) for s in levels}

    tuned, vectors = {}, {}
    original = list(k.START_GUESS[a.variant])
    previous = None
    for s in sorted(levels):
        t1 = time.time()
        print("=== tuning A: noise level %g A ===" % s)
        starts = [original]
        with_noise = list(original)
        with_noise[3] = float(np.log10(max(s * s, 1e-8)))            # r_i = sigma^2 as a starting point
        starts.append(with_noise)
        if previous is not None:
            starts.append(previous)                                  # best result of the next lower noise level
        v, best = tune_multistart(a.variant, train_sets[s], evals, starts)
        vectors[s] = v
        previous = list(v)
        tuned["tuned for sigma = %g A" % s] = k.vec_to_params(v)
        print("    done in %.0f s\n" % (time.time() - t1))
    t1 = time.time()
    print("=== tuning B: robust, all noise levels together %s ===" % levels)
    all_runs = [d for s in levels for d in train_sets[s]]
    starts = [vectors[s] for s in sorted(levels)]                    # start from the separate tunings
    v, best = tune_multistart(a.variant, all_runs, evals, starts[:2] + starts[-1:] if len(starts) > 3 else starts)
    tuned["robust (all levels together)"] = k.vec_to_params(v)
    print("    done in %.0f s\n" % (time.time() - t1))

    print("=" * 100)
    print(" Camshaft torque error on the TEST runs, in %% of the rms torque (variant %s)" % a.variant)
    print(" rows: how the filter was tuned     columns: noise level of the current in the test")
    print("=" * 100)
    print("%-34s" % "tuned for" + "".join("%14s" % ("test %g A" % s) for s in levels))
    table = {}
    for name, params in tuned.items():
        row = [score(params, a.variant, test_sets[s]) for s in levels]
        table[name] = row
        print("%-34s" % name + "".join("%13.2f%%" % r for r in row))

    print()
    print(" The tuned noise setting r_i of the current (it should be close to sigma^2 of the tuning noise):")
    print("%-34s %12s %14s" % ("tuned for", "r_i tuned", "sigma^2 of tuning"))
    for s in levels:
        p = tuned["tuned for sigma = %g A" % s]
        print("%-34s %12.3g %14.3g" % ("sigma = %g A" % s, p["r_i"], s * s))
    p = tuned["robust (all levels together)"]
    print("%-34s %12.3g" % ("robust", p["r_i"]))

    with open(a.out + ".json", "w") as f:
        json.dump({"variant": a.variant, "levels": levels, "tuned": tuned, "test_error_pct": table}, f, indent=2)
    print("\nsaved %s.json" % a.out)


if __name__ == "__main__":
    main()
