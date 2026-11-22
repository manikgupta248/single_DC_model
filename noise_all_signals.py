"""
Noise on ALL measured signals of the torque Kalman filter (not only the current).

What is made noisy (every source has its own standard deviation, 0 = off):
  --sigma-i      current i                     Gaussian white noise [A]                  -> filter setting r_i
  --sigma-th     interpolated Hall angle       Gaussian white noise [deg, motor side]    -> r_th   (variant interp)
  --sigma-edge   Hall EDGE TIMING              Gaussian jitter of every edge time [us]   -> r_w     (variant no_interp)
                 (the speed of the variant no_interp is q / time between two edges, so jitter of the
                  edge times is the noise of that speed measurement)
  --sigma-pos    Hall edge POSITION            fixed error of each of the 36 edge positions per motor
                 turn [deg, motor side] (magnet / sensor placement). Same error every turn.   -> r_th (no_interp)
  --sigma-v      applied voltage               Gaussian white noise [V] on the voltage input of the model

Edge jitter and position error act on the times at which the quantised angle jumps, so the edges that the filter
detects move; no noise is put on the angle values themselves (an edge is always an exact multiple of 10 deg).
The filter works on a 0.2 ms grid, so edge-time jitter far below 0.1 ms only changes the edge sample with a
probability (the model of a sensor that is sampled every 0.1 ms).

What the script does
  1) default: sensitivity table. Torque error (RMSE in % of the rms camshaft torque) on the test runs for
     no noise / each source alone / all sources together, with a given filter tuning.
  2) --tune: tunes the filter again for the combined noise and shows the table with the old and the new tuning.

Examples
    python noise_all_signals.py --quick
    python noise_all_signals.py --sigma-i 0.05 --sigma-edge 20 --sigma-pos 0.3 --sigma-v 0.02
    python noise_all_signals.py --variant interp --sigma-th 0.5
    python noise_all_signals.py --tune --params noise_tuning.json      (starts from the robust tuning in that file)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torque_kf as k

# tuning of the variant no_interp without noise (same values as in plot_kalman_results.py)
DEFAULT_PARAMS = dict(q_w=1.8558420108369016, q_i=0.6002396839788338, q_T=0.11826244543066099,
                      r_i=1.4195516188691276e-08, r_th=1.931481126190632e-05, r_w=152.87458228211878)
EDGES_PER_TURN = 36                     # Hall edges per motor turn (10 deg)


# =============================================================================
# Noise on the signals
# =============================================================================
def shift_edges(Q, t, e_t_shift):
    """Move the edges of the staircase Q by the times e_t_shift [s] (one value per edge, in the order of the
    edges). The edge j happened (by assumption) in the middle of the sample before the sample where Q changed."""
    dt = float(t[1] - t[0])
    edge = np.flatnonzero(Q[1:] != Q[:-1]) + 1                     # first sample after each edge
    if len(edge) == 0:
        return Q.copy()
    t_edge = t[edge] - dt / 2
    new = np.floor((t_edge + e_t_shift - t[0]) / dt).astype(int) + 1   # first sample after the shifted edge
    new = np.clip(new, 1, len(Q) - 1)
    for j in range(1, len(new)):                                   # keep the order of the edges
        new[j] = max(new[j], new[j - 1] + 1)
    new = np.minimum(new, len(Q) - 1)
    Qn = np.full(len(Q), Q[0])
    for j in range(len(new)):
        Qn[new[j]:] = Q[edge[j]]
    return Qn


def add_noise(d, s_i=0.0, s_th_deg=0.0, s_edge_us=0.0, s_pos_deg=0.0, s_v=0.0, seed=0, pos_error=None):
    """Copy of the run d with noisy measurements.
    Keys written: i_meas, Hall_meas, Q_meas, V_meas (the filter uses them when they are present)."""
    rng = np.random.default_rng(seed)
    e = dict(d)
    n = len(d["Li"])
    e["i_meas"] = d["Li"] + (s_i * rng.standard_normal(n) if s_i > 0 else 0.0)
    e["Hall_meas"] = d["Hall_out"] + (np.deg2rad(s_th_deg) * rng.standard_normal(n) if s_th_deg > 0 else 0.0)
    if s_v > 0:
        e["V_meas"] = d["V"] + s_v * rng.standard_normal(n)
    Q = d["Camshaft_quantanizer"]
    if s_edge_us > 0 or s_pos_deg > 0:
        t = d["t"]
        edge = np.flatnonzero(Q[1:] != Q[:-1]) + 1
        shift = np.zeros(len(edge))
        if s_edge_us > 0:
            shift += 1e-6 * s_edge_us * rng.standard_normal(len(edge))
        if s_pos_deg > 0:
            # fixed position error per Hall edge (the same in every motor turn), seeded separately
            if pos_error is None:
                pos_error = np.deg2rad(s_pos_deg) * np.random.default_rng(12345 + seed).standard_normal(EDGES_PER_TURN)
            # edge between two levels: id = the higher level (same sensor edge for forward and reverse crossing)
            level = np.round(np.maximum(Q[edge], Q[edge - 1]) / k.base.QuantizationInterval).astype(int) % EDGES_PER_TURN
            w = d["Wi1"][edge]
            w = np.where(np.abs(w) < 1.0, np.where(w < 0, -1.0, 1.0), w)        # signed motor speed [rad/s]
            # the sensor edge sits e rad away from its nominal angle: it is seen e/w later (w > 0) or earlier
            shift += pos_error[level] / w
        e["Q_meas"] = shift_edges(Q, t, shift)
    return e


def noisy_set(runs, cfg, seed0):
    return [add_noise(d, seed=seed0 + i, **cfg) for i, d in enumerate(runs)]


# =============================================================================
# Evaluation
# =============================================================================
def error_pct(params, variant, runs):
    vals = []
    for d in runs:
        e = k.estimates(d, variant, params, "nominal")
        vals.append(k.metrics(e["true_cam"], e["est_cam"])["RMSE_pct"])
    return float(np.mean(vals))


def sensitivity(params, variant, runs, cfg, seeds):
    """Rows: no noise, each active source alone, all together. Mean error over the seeds."""
    names = {"s_i": "current", "s_th_deg": "Hall angle (interp)", "s_edge_us": "edge timing (jitter)",
             "s_pos_deg": "edge position error", "s_v": "voltage"}
    rows = [("no noise", {})]
    for key, name in names.items():
        if cfg[key] > 0:
            rows.append(("%s only" % name, {key: cfg[key]}))
    active = {key: v for key, v in cfg.items() if v > 0}
    if len(active) > 1:
        rows.append(("all together", active))
    out = []
    for label, c in rows:
        full = dict(s_i=0.0, s_th_deg=0.0, s_edge_us=0.0, s_pos_deg=0.0, s_v=0.0)
        full.update(c)
        errs = [error_pct(params, variant, noisy_set(runs, full, 7000 + 100 * s)) for s in range(seeds if c else 1)]
        out.append((label, float(np.mean(errs)), float(np.std(errs))))
    return out


def load_params(path, variant):
    if not path:
        return dict(DEFAULT_PARAMS), "no-noise tuning of plot_kalman_results.py"
    with open(path) as f:
        data = json.load(f)
    if "tuned" in data:
        for key in data["tuned"]:
            if key.startswith("robust"):
                return data["tuned"][key], "%s : %s" % (path, key)
        key = list(data["tuned"])[-1]
        return data["tuned"][key], "%s : %s" % (path, key)
    if variant in data:
        return data[variant], "%s : %s" % (path, variant)
    return data, path


def theory_hint(cfg, dt=2e-4):
    print("  Rough values of the filter settings that match these noise levels:")
    print("    r_i  = sigma_i^2           = %.3g A^2" % cfg["s_i"] ** 2)
    print("    r_th = sigma_th^2 (interp) = %.3g rad^2,   edge position error^2 (no_interp) = %.3g rad^2"
          % (np.deg2rad(cfg["s_th_deg"]) ** 2, np.deg2rad(cfg["s_pos_deg"]) ** 2))
    print("    edge jitter -> speed noise: var(w) ~ 2 * (w*sigma_t)^2 / (Dt)^2 * ... (depends on speed),"
          " so r_w must be tuned (--tune)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="no_interp", choices=["interp", "no_interp"])
    ap.add_argument("--sigma-i", type=float, default=0.05, help="current noise std [A] (default 0.05)")
    ap.add_argument("--sigma-th", type=float, default=0.0, help="noise std of the interpolated Hall angle [deg] (variant interp)")
    ap.add_argument("--sigma-edge", type=float, default=20.0, help="jitter std of the Hall edge times [us] (default 20)")
    ap.add_argument("--sigma-pos", type=float, default=0.3, help="std of the fixed edge-position errors [deg, motor side] (default 0.3)")
    ap.add_argument("--sigma-v", type=float, default=0.02, help="voltage noise std [V] (default 0.02)")
    ap.add_argument("--params", help="json with a tuning (noise_tuning.json or a flat dict); default: no-noise tuning")
    ap.add_argument("--tune", action="store_true", help="tune the filter again for the combined noise")
    ap.add_argument("--evals", type=int, default=50, help="evaluations per start when tuning (default 50)")
    ap.add_argument("--seeds", type=int, default=3, help="noise realisations averaged in the table (default 3)")
    ap.add_argument("--skip-ms", type=float, default=20.0, help="start-up time left out of the errors [ms] (default 20)")
    ap.add_argument("--quick", action="store_true", help="small data set and few evaluations")
    ap.add_argument("--out", default="noise_all", help="prefix of the output files / data caches")
    a = ap.parse_args()

    k.SKIP_S = a.skip_ms * 1e-3
    cfg = dict(s_i=a.sigma_i, s_th_deg=a.sigma_th, s_edge_us=a.sigma_edge, s_pos_deg=a.sigma_pos, s_v=a.sigma_v)
    if a.variant == "interp":                        # edge noise does not exist in this variant
        cfg["s_edge_us"] = cfg["s_pos_deg"] = 0.0
    else:
        cfg["s_th_deg"] = 0.0                        # no_interp does not use the interpolated angle
    if a.quick:
        train_pwm, train_w0, test_pwm, test_w0 = (0.06, 0.2), (0.0,), (0.1,), (30.0,)
        evals = min(a.evals, 20)
    else:
        train_pwm, train_w0, test_pwm, test_w0 = (0.06, 0.1, 0.15, 0.2), (0.0, 30.0), (0.08, 0.12, 0.18), (15.0, 45.0)
        evals = a.evals

    t0 = time.time()
    print("simulating the runs (without noise) ...")
    test = k.build_dataset(test_pwm, test_w0, 0.0, False, a.out + "_test.pkl", "test")
    train = k.build_dataset(train_pwm, train_w0, 0.0, False, a.out + "_train.pkl", "train") if a.tune else None
    print("%d test runs (%.0f s)\n" % (len(test), time.time() - t0))
    print("variant %s, noise: %s\n" % (a.variant, {key: v for key, v in cfg.items() if v > 0}))

    params, source = load_params(a.params, a.variant)
    print("filter tuning used: %s\n" % source)
    theory_hint(cfg)

    def show(title, p):
        rows = sensitivity(p, a.variant, test, cfg, a.seeds)
        print("\n" + title)
        print("%-26s %12s %12s" % ("noise on", "error [%]", "(std seeds)"))
        for label, m, s in rows:
            print("%-26s %11.2f%% %11.2f" % (label, m, s))
        return rows

    result = {"variant": a.variant, "noise": cfg, "tuning_used": source}
    result["table_given_tuning"] = show("Torque error on the test runs, given tuning (errors in %% of rms torque, "
                                        "first %g ms left out)" % a.skip_ms, params)

    if a.tune:
        all_cfg = {key: v for key, v in cfg.items() if v > 0}
        full = dict(s_i=0.0, s_th_deg=0.0, s_edge_us=0.0, s_pos_deg=0.0, s_v=0.0)
        full.update(all_cfg)
        train_noisy = noisy_set(train, full, 300)
        start0 = [float(np.log10(params[n])) for n in k.PARAM_NAMES]
        start1 = list(start0)
        start1[3] = float(np.log10(max(cfg["s_i"] ** 2, 1e-8)))             # r_i = sigma_i^2
        if cfg["s_th_deg"] > 0:
            start1[4] = float(np.log10(np.deg2rad(cfg["s_th_deg"]) ** 2))   # r_th = sigma_th^2
        elif cfg["s_pos_deg"] > 0:
            start1[4] = float(np.log10(np.deg2rad(cfg["s_pos_deg"]) ** 2))
        import tune_noise_levels as tn
        print("\n=== tuning for the combined noise (%d evaluations per start) ===" % evals)
        v, best = tn.tune_multistart(a.variant, train_noisy, evals, [start0, start1])
        new = k.vec_to_params(v)
        print("\nnew tuning:")
        for n in k.PARAM_NAMES:
            print("  %-5s old %-12.5g new %-12.5g" % (n, params[n], new[n]))
        result["tuned_for_this_noise"] = new
        result["table_new_tuning"] = show("Torque error on the test runs, NEW tuning", new)

    with open(a.out + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved %s.json  (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
