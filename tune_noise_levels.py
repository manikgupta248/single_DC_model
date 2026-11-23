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

Graphs of the ROBUST tuning (separate figures, saved as png):
    python tune_noise_levels.py --plot               tune, then plot
    python tune_noise_levels.py --plot-only          do not tune again: read <out>.json from an earlier run and plot
    options: --pwm 0.12 --angle 20 (the movement), --seeds 5, --out-dir plots, --no-show
    noise_torque_sigma<s>.png : true torque, estimate without noise and with noise (one figure per noise level)
    noise_difference.png      : estimate with noise minus estimate without noise, against the cam angle
    noise_summary.png         : torque error without and with noise (bars); the table of the differences is printed
    RMSE [N*m, camshaft side], RMSE in % of the rms torque and R^2 are printed for every tuning/noise level and are
    written in the titles of the graphs (R^2 = 1 - sum(error^2) / sum((true - mean(true))^2), 1 = perfect).

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


def score_all(params, variant, runs):
    """Mean over the runs of RMSE [N*m], RMSE in % of the rms torque and R^2 (camshaft torque, without the start-up)."""
    ms = []
    for d in runs:
        e = k.estimates(d, variant, params, "nominal")
        ms.append(k.metrics(e["true_cam"], e["est_cam"]))
    return {key: float(np.mean([m[key] for m in ms])) for key in ("RMSE", "RMSE_pct", "R2")}


def estimate_torque(data, variant, params, sigma, seed):
    """Camshaft torque estimate (decimated) with Gaussian noise (std sigma [A]) added to the measured current."""
    d = dict(data)
    if sigma > 0:
        rng = np.random.default_rng(seed)
        d["i_meas"] = data["Li"] + sigma * rng.standard_normal(len(data["Li"]))
    else:
        d["i_meas"] = data["Li"].copy()
    x = k.run_filter(d, variant, params, {})
    return x[:, 3] / k.base.Gload


def plot_results(variant, params, levels, a):
    """Figures of the robust tuning: estimate without and with noise, difference, summary bars, printed table."""
    import matplotlib
    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = [s for s in levels if s > 0]
    os.makedirs(a.out_dir, exist_ok=True)
    print("simulating the movement (PWM %.2f, start %g deg) for the graphs ..." % (a.pwm, a.angle))
    data = k.generate_run(a.pwm, a.angle, +1)
    if data is None:
        raise SystemExit("the cam does not move 40 deg at this PWM")
    sl = slice(None, None, k.KF_DECIMATION)
    cam = data["Gdeg"][sl]
    true = data["Tab"][sl]
    t_ms = (data["t"][sl] - data["t"][0]) * 1000.0
    keep = t_ms >= a.skip_ms
    clean = estimate_torque(data, variant, params, 0.0, 0)

    def err(est):
        m = k.metrics(true[keep], est[keep])
        return m["RMSE"], m["RMSE_pct"], m["R2"]

    figs, diffs = [], {}
    for sigma in sigmas:
        noisy = estimate_torque(data, variant, params, sigma, 1)
        diffs[sigma] = noisy - clean
        r0, p0, q0 = err(clean)
        r1, p1, q1 = err(noisy)
        fig, ax = plt.subplots(figsize=(10, 6.0))
        ax.plot(cam, true, color="#555555", lw=4.0, label="true torque")
        ax.plot(cam[1:], clean[1:], color="#2e7d32", lw=1.8, ls="--", label="estimate without noise")
        ax.plot(cam[1:], noisy[1:], color="#c0392b", lw=1.0, label="estimate with noise (sigma = %g A)" % sigma)
        ax.set_xlabel("cam angle [deg]", fontsize=11)
        ax.set_ylabel("camshaft torque [N*m]", fontsize=11)
        ax.set_title("Robust tuning, noise on the current sigma = %g A: true and estimated torque\n"
                     "without noise: RMSE %.3f N*m (%.2f %%), R$^2$ = %.4f      with noise: RMSE %.3f N*m (%.2f %%), "
                     "R$^2$ = %.4f\n(first %g ms left out)"
                     % (sigma, r0, p0, q0, r1, p1, q1, a.skip_ms), fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
        fig.tight_layout()
        path = os.path.join(a.out_dir, "noise_torque_sigma%g.png" % sigma)
        fig.savefig(path, dpi=150)
        print("saved %s" % path)
        figs.append(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for sigma, col in zip(sigmas, ("#1f77b4", "#e08a00", "#c0392b", "#6a3d9a", "#2e7d32")):
        ax.plot(cam[1:], diffs[sigma][1:], lw=1.0, color=col, label="sigma = %g A" % sigma)
    ax.axhline(0, color="gray", lw=0.8)
    lim = 1.3 * max(np.abs(diffs[sg][keep]).max() for sg in sigmas)          # range of the part after the start-up
    startup = max(np.abs(diffs[sg][1:]).max() for sg in sigmas)
    ax.set_ylim(-lim, lim)
    ax.text(0.98, 0.04, "start-up (first %g ms) differences up to %.1f N*m are outside the range shown" % (a.skip_ms, startup),
            transform=ax.transAxes, ha="right", fontsize=9, color="#555555")
    ax.set_xlabel("cam angle [deg]", fontsize=11)
    ax.set_ylabel("estimate with noise - estimate without noise [N*m]", fontsize=11)
    ax.set_title("How much the torque prediction changes when noise is added to the current (robust tuning)", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(a.out_dir, "noise_difference.png")
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    figs.append(fig)

    r0, p0, q0 = err(clean)
    rms_true = float(np.sqrt(np.mean(true[keep] ** 2)))
    rows = []
    for sigma in sigmas:
        dif, mx, e1, e2, e3 = [], [], [], [], []
        for seed in range(1, a.seeds + 1):
            noisy = estimate_torque(data, variant, params, sigma, seed)
            dd = (noisy - clean)[keep]
            dif.append(np.sqrt(np.mean(dd ** 2)))
            mx.append(np.abs(dd).max())
            m1 = err(noisy)
            e1.append(m1[1])
            e2.append(m1[0])
            e3.append(m1[2])
        rows.append((sigma, float(np.mean(dif)), float(np.mean(mx)), float(np.mean(e1)),
                     float(np.mean(e2)), float(np.mean(e3))))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    xs = np.arange(len(sigmas))
    ax.bar(xs - 0.2, [p0] * len(xs), width=0.38, color="#2e7d32", label="without noise")
    ax.bar(xs + 0.2, [r[3] for r in rows], width=0.38, color="#c0392b", label="with noise")
    for x, r in zip(xs, rows):
        ax.text(x + 0.2, r[3] + 0.05, "%.2f %%\nR$^2$ = %.4f" % (r[3], r[5]), ha="center", fontsize=9)
        ax.text(x - 0.2, p0 + 0.05, "%.2f %%\nR$^2$ = %.4f" % (p0, q0), ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(["sigma = %g A" % s for s in sigmas])
    ax.set_ylabel("torque error (RMSE in % of the rms torque)", fontsize=11)
    ax.set_title("Robust tuning: torque error without and with noise on the current\n(mean of %d noise realisations)" % a.seeds,
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylim(0, 1.15 * max([p0] + [r[3] for r in rows]))
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    path = os.path.join(a.out_dir, "noise_summary.png")
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    figs.append(fig)

    print()
    print("Difference between the torque prediction with noise and without noise (PWM %.2f, start %g deg, first %g ms left out)"
          % (a.pwm, a.angle, a.skip_ms))
    print("rms torque of the movement: %.3f N*m" % rms_true)
    print("without noise: RMSE %.3f N*m = %.2f %% of the rms torque, R2 = %.5f" % (r0, p0, q0))
    print("%-10s %-32s %-18s %-14s %-12s %-10s %-10s" % ("noise", "rms difference of prediction", "largest diff.",
                                                         "RMSE [N*m]", "RMSE [%]", "R2", "(with noise)"))
    for sigma, dif, mx, e1, e2, e3 in rows:
        print("%-10s %-32s %-18s %-14s %-12s %-10s" % ("%g A" % sigma, "%.3f N*m (%.2f %% of rms)" % (dif, 100 * dif / rms_true),
                                                       "%.3f N*m" % mx, "%.3f" % e2, "%.2f %%" % e1, "%.5f" % e3))
    if not a.no_show:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


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
    ap.add_argument("--plot", action="store_true", help="after the tuning, make the graphs of the robust tuning")
    ap.add_argument("--plot-only", action="store_true", help="no tuning: read <out>.json of an earlier run and make the graphs")
    ap.add_argument("--pwm", type=float, default=0.12, help="PWM of the movement in the graphs (default 0.12)")
    ap.add_argument("--angle", type=float, default=20.0, help="start angle of the movement in the graphs [deg] (default 20)")
    ap.add_argument("--seeds", type=int, default=5, help="noise realisations for the table of the graphs (default 5)")
    ap.add_argument("--skip-ms", type=float, default=20.0, help="start-up time left out of the errors in the graphs [ms]")
    ap.add_argument("--out-dir", default=".", help="folder for the png files (default: current folder)")
    ap.add_argument("--no-show", action="store_true", help="save the png files but do not open the figures")
    a = ap.parse_args()
    if a.plot_only:
        with open(a.out + ".json") as f:
            saved = json.load(f)
        plot_results(saved["variant"], saved["tuned"]["robust (all levels together)"], saved["levels"], a)
        return
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
    table, full = {}, {}
    for name, params in tuned.items():
        res = [score_all(params, a.variant, test_sets[s]) for s in levels]
        full[name] = res
        table[name] = [r["RMSE_pct"] for r in res]
        print("%-34s" % name + "".join("%13.2f%%" % r["RMSE_pct"] for r in res))
    print()
    print(" RMSE of the camshaft torque in N*m (same rows and columns):")
    print("%-34s" % "tuned for" + "".join("%14s" % ("test %g A" % s) for s in levels))
    for name, res in full.items():
        print("%-34s" % name + "".join("%14.4f" % r["RMSE"] for r in res))
    print()
    print(" R^2 of the camshaft torque (1 = perfect; same rows and columns):")
    print("%-34s" % "tuned for" + "".join("%14s" % ("test %g A" % s) for s in levels))
    for name, res in full.items():
        print("%-34s" % name + "".join("%14.5f" % r["R2"] for r in res))

    print()
    print(" The tuned noise setting r_i of the current (it should be close to sigma^2 of the tuning noise):")
    print("%-34s %12s %14s" % ("tuned for", "r_i tuned", "sigma^2 of tuning"))
    for s in levels:
        p = tuned["tuned for sigma = %g A" % s]
        print("%-34s %12.3g %14.3g" % ("sigma = %g A" % s, p["r_i"], s * s))
    p = tuned["robust (all levels together)"]
    print("%-34s %12.3g" % ("robust", p["r_i"]))

    with open(a.out + ".json", "w") as f:
        json.dump({"variant": a.variant, "levels": levels, "tuned": tuned, "test_error_pct": table,
                   "test_rmse_Nm": {n: [r["RMSE"] for r in v] for n, v in full.items()},
                   "test_r2": {n: [r["R2"] for r in v] for n, v in full.items()}}, f, indent=2)
    print("\nsaved %s.json" % a.out)
    if a.plot:
        plot_results(a.variant, tuned["robust (all levels together)"], levels, a)


if __name__ == "__main__":
    main()
