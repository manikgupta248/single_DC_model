"""
torque_kf.py - Kalman filter that estimates the load torque acting on the camshaft from the motor
current (DC-equivalent model) and the Hall sensor signal, at CONSTANT PWM, inside a 40 deg window.

The main model emotor_cam_move.py is used unchanged as the "plant" (truth). One simulation per run
provides both Hall variants:
    variant "interp"    : angle measurement = Hall_out of HALL_MODE = INTERPOLATION (every step)
    variant "no_interp" : only Hall EDGES are used: the exact edge angle (quantized angle Q -> the edge sits at
                          Q - sign*q/2) and the speed from the time between two edges (q / edge interval);
                          nothing in between the edges
    variant "naive"     : (extra) quantized angle treated as noisy angle measurement at every step

Model of the filter (linear at constant voltage V = Vdc * PWM):
    states x = [theta (motor angle), omega, i, T_L]      T_L = load torque at the motor side
    d theta/dt = omega
    J d omega/dt = Kt*i - B*omega - T_L
    L d i/dt     = V - R*i - Ke*omega
    d T_L/dt     = white noise (random walk)             camshaft torque = T_L / Gload   (Gload = 1/(73*0.8))
Measurements: current i (every step), angle (variant dependent).

Q and R (tuned automatically on simulated training runs, truth = Tab of the model):
    q_w, q_i, q_T : process noise of omega-dot, i-dot, T_L (how much the model may be wrong / how fast T_L moves)
    r_i, r_th     : measurement noise of the current and of the angle
    r_w           : extra noise of the edge-timing speed measurement (variant no_interp only)

Usage:
    python torque_kf.py                    # generate data (cached), tune, evaluate, save results + plot
    python torque_kf.py --quick            # small data set / few evaluations (about 10 min)
    python torque_kf.py --noise-i 0.1      # add 0.1 A noise to the measured current
    python torque_kf.py --regen            # ignore the data cache
Outputs: torque_kf_params.json, torque_kf_results.csv, torque_kf.png
"""
import argparse
import csv
import json
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotor_cam_move as base          # main model, used unchanged

WINDOW_DEG = 40.0                       # the phaser can move 40 deg (camshaft) from the start
KF_DECIMATION = 2                       # Kalman filter step = LOG step (1e-4 s) * this
VARIANTS = ("interp", "no_interp", "naive")
MISMATCH_CASES = {"nominal": {}, "mismatch": {"R": 1.2, "Kt": 0.95, "Ke": 0.95}}   # model errors of the filter
SKIP_S = 0.02                           # ignore the first 20 ms (filter start-up) in the metrics
PARAM_NAMES = ("q_w", "q_i", "q_T", "r_i", "r_th", "r_w")
START_GUESS = {"interp": (3.0, 0.0, -1.5, -4.0, -2.6, 0.0), "no_interp": (3.0, 0.0, -1.5, -4.0, -6.0, 0.0),
               "naive": (3.0, 0.0, -1.5, -4.0, -2.6, 0.0)}                  # log10 of (q_w, q_i, q_T, r_i, r_th, r_w)
BOUNDS = ((-1, 8), (-4, 6), (-4, 5), (-8, 0), (-8, 0), (-4, 4))             # log10 bounds (r_w only used by no_interp)


# =============================================================================
# Data: runs of the main model at constant PWM
# =============================================================================
def generate_run(pwm, w0, direction, i_noise=0.0, seed=0):
    """One sweep through the cam window [w0, w0 + 40 deg]. direction +1: from w0 up, -1: from w0 + 40 down.
    pwm > 0. Returns a dict of numpy arrays (truncated when 40 deg of cam travel are reached) or None."""
    saved = (base.CONTROL_MODE, base.CONSTANT_PWM, base.START_ANGLE_DEG, base.StopTime, base.HALL_MODE,
             base.WRAP_LOAD_TABLE)
    start = w0 if direction > 0 else w0 + WINDOW_DEG
    try:
        base.CONTROL_MODE = "CONSTANT"
        base.CONSTANT_PWM = direction * pwm
        base.START_ANGLE_DEG = start
        base.HALL_MODE = "INTERPOLATION"            # log has BOTH the quantized and the interpolated angle
        base.WRAP_LOAD_TABLE = True
        log = None
        for stop in (0.7, 2.0, 5.0):
            base.StopTime = stop
            log = base.simulate()
            g = np.asarray(log["Gdeg"])
            if np.any(np.abs(g - g[0]) >= WINDOW_DEG):
                break
        else:
            return None
    finally:
        (base.CONTROL_MODE, base.CONSTANT_PWM, base.START_ANGLE_DEG, base.StopTime, base.HALL_MODE,
         base.WRAP_LOAD_TABLE) = saved
    n = int(np.argmax(np.abs(g - g[0]) >= WINDOW_DEG)) + 1
    d = {k: np.asarray(log[k][:n], dtype=float) for k in
         ("t", "Li", "Wi", "Wi1", "Gdeg", "Tab", "Tm", "Camshaft_quantanizer", "Hall_out")}
    rng = np.random.default_rng(seed)
    d["i_meas"] = d["Li"] + (i_noise * rng.standard_normal(n) if i_noise > 0 else 0.0)
    d["V"] = base.Vdc * direction * pwm
    d["meta"] = dict(pwm=pwm, w0=w0, direction=direction, t_end=float(d["t"][-1]))
    return d


def build_dataset(pwms, w0s, i_noise, regen, cache_file, tag):
    key = repr((tag, pwms, w0s, i_noise, base.LOG_DECIMATION))
    if os.path.exists(cache_file) and not regen:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        if cache.get("key") == key:
            print("  data loaded from %s (%d runs)" % (cache_file, len(cache["runs"])))
            return cache["runs"]
    runs = []
    for pwm in pwms:
        for w0 in w0s:
            for direction in (1, -1):
                t0 = time.time()
                r = generate_run(pwm, w0, direction, i_noise, seed=len(runs))
                if r is None:
                    print("  PWM %.2f w0 %g dir %+d: 40 deg not reached -> skipped" % (pwm, w0, direction))
                    continue
                runs.append(r)
                print("  PWM %.2f w0 %2g dir %+d: %.3f s window, %d samples (%.0f s)"
                      % (pwm, w0, direction, r["meta"]["t_end"], len(r["t"]), time.time() - t0))
    with open(cache_file, "wb") as f:
        pickle.dump({"key": key, "runs": runs}, f)
    return runs


# =============================================================================
# Kalman filter
# =============================================================================
def expm(M):
    """Matrix exponential (scaling and squaring, Taylor series) - no scipy needed."""
    s = max(0, int(math.ceil(math.log2(max(np.linalg.norm(M, 1), 1e-12)))) + 1)
    A = M / (2.0 ** s)
    E = np.eye(M.shape[0])
    T = np.eye(M.shape[0])
    for k in range(1, 20):
        T = T @ A / k
        E = E + T
    for _ in range(s):
        E = E @ E
    return E


def build_model(dt, q_w, q_i, q_T, mm):
    R = base.Rc * mm.get("R", 1.0)
    Kt = base.Ktc * mm.get("Kt", 1.0)
    Ke = base.Kec * mm.get("Ke", 1.0)
    L, B, J = base.Lc, base.Bc, base.J
    A = np.array([[0, 1, 0, 0],
                  [0, -B / J, Kt / J, -1 / J],
                  [0, -Ke / L, -R / L, 0],
                  [0, 0, 0, 0]], dtype=float)
    Aug = np.zeros((5, 5))
    Aug[:4, :4] = A
    Aug[2, 4] = 1.0 / L                           # input V enters di/dt
    E = expm(Aug * dt)
    F, Bd = E[:4, :4], E[:4, 4]
    M = np.zeros((8, 8))                          # Van Loan: discrete process noise
    M[:4, :4] = -A
    M[:4, 4:] = np.diag([0.0, q_w, q_i, q_T])
    M[4:, 4:] = A.T
    E2 = expm(M * dt)
    Qd = E2[4:, 4:].T @ E2[:4, 4:]
    return F, Bd, 0.5 * (Qd + Qd.T)


def run_filter(data, variant, p, mm, dec=KF_DECIMATION):
    """p = dict(q_w, q_i, q_T, r_i, r_th) in linear units. Returns the estimates (decimated)."""
    sl = slice(None, None, dec)
    t = data["t"][sl]
    z_i = data["i_meas"][sl]
    Q = data["Camshaft_quantanizer"][sl]
    Hall = data["Hall_out"][sl]
    dt = float(t[1] - t[0])
    F, Bd, Qd = build_model(dt, p["q_w"], p["q_i"], p["q_T"], mm)
    Bu = Bd * data["V"]
    n = len(t)
    q = base.QuantizationInterval
    x = np.array([data["Wi1"][0], 0.0, 0.0, 0.0])
    P = np.diag([1e-6, 1e-4, 1e-4, 0.05 ** 2])
    r_i, r_th = p["r_i"], p["r_th"]
    out = np.zeros((n, 4))
    out[0] = x
    Kt_m = base.Ktc * mm.get("Kt", 1.0)
    B_m, J_m = base.Bc, base.J
    t_edge_prev, s_prev = None, 0.0
    for k in range(1, n):
        x = F @ x + Bu                                  # predict
        P = F @ P @ F.T + Qd
        # --- current update (every step) ---
        S = P[2, 2] + r_i
        K = P[:, 2] / S
        x = x + K * (z_i[k] - x[2])
        P = P - np.outer(K, P[2])
        # --- angle update ---
        if variant == "interp":
            z, r = Hall[k], r_th
        elif variant == "naive":
            z, r = Q[k], r_th
        else:                                           # no_interp: only at a Hall edge
            if Q[k] == Q[k - 1]:
                out[k] = x
                continue
            s = 1.0 if Q[k] > Q[k - 1] else -1.0
            w_hat = x[1]
            z = Q[k] - s * q / 2 + w_hat * dt / 2       # the edge happened somewhere inside the last step
            r = r_th + (w_hat * dt) ** 2 / 12
        S = P[0, 0] + r
        K = P[:, 0] / S
        x = x + K * (z - x[0])
        P = P - np.outer(K, P[0])
        if variant == "no_interp":                      # speed from the time between two edges
            t_edge = t[k] - dt / 2
            if t_edge_prev is not None and s == s_prev:
                dte = t_edge - t_edge_prev
                w_meas = s * q / dte
                wdot = (Kt_m * x[2] - B_m * x[1] - x[3]) / J_m              # the average speed belongs to mid-interval
                r_w_tot = p["r_w"] + (wdot * dte / 2) ** 2 + (w_meas * dt / dte) ** 2 / 6
                S = P[1, 1] + r_w_tot
                K = P[:, 1] / S
                x = x + K * (w_meas - x[1])
                P = P - np.outer(K, P[1])
            t_edge_prev, s_prev = t_edge, s
        P = 0.5 * (P + P.T)
        out[k] = x
    return out


def vec_to_params(v):
    return {n: 10.0 ** float(x) for n, x in zip(PARAM_NAMES, v)}


# =============================================================================
# Metrics, tuning, evaluation
# =============================================================================
def metrics(true, est):
    e = est - true
    sst = float(np.sum((true - true.mean()) ** 2))
    rms_true = float(np.sqrt(np.mean(true ** 2)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return dict(RMSE=rmse, RMSE_pct=100 * rmse / rms_true if rms_true > 0 else float("nan"),
                R2=1 - float(np.sum(e ** 2)) / sst if sst > 0 else float("nan"),
                Bias=float(e.mean()), MaxAbs=float(np.abs(e).max()))


def estimates(data, variant, p, mm_name):
    """Returns time, truth and estimates (camshaft and motor side) for one run."""
    mm = MISMATCH_CASES[mm_name]
    x = run_filter(data, variant, p, mm)
    sl = slice(None, None, KF_DECIMATION)
    t = data["t"][sl]
    keep = t - t[0] >= SKIP_S
    Gl = base.Gload
    return dict(t=t[keep], cam=data["Gdeg"][sl][keep],
                true_cam=data["Tab"][sl][keep], est_cam=x[keep, 3] / Gl,
                true_mot=data["Tab"][sl][keep] * Gl, est_mot=x[keep, 3],
                true_Tm=data["Tm"][sl][keep], est_Tm=base.Ktc * mm.get("Kt", 1.0) * x[keep, 2],
                naive_mot=(base.Ktc * mm.get("Kt", 1.0)) * data["i_meas"][sl][keep])


def objective(v, variant, runs):
    p = vec_to_params(v)
    tot, cnt = 0.0, 0
    for d in runs:
        for name in MISMATCH_CASES:
            e = estimates(d, variant, p, name)
            tot += metrics(e["true_cam"], e["est_cam"])["RMSE_pct"]
            cnt += 1
    val = tot / cnt
    return val if math.isfinite(val) else 1e6


def tune(variant, runs, evals, seed=0):
    """Random search (first third of the evaluations), then a (1+4) evolution strategy with step-size adaptation
    around the best point (log10 space)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS], dtype=float)
    hi = np.array([b[1] for b in BOUNDS], dtype=float)
    best_v = np.array(START_GUESS[variant], dtype=float)
    best = objective(best_v, variant, runs)
    print("  [%s] start: objective %.2f %%" % (variant, best))
    used = 1
    n_rand = max(5, evals // 3)
    while used < n_rand:
        cand = lo + (hi - lo) * rng.random(len(lo))
        val = objective(cand, variant, runs)
        used += 1
        if val < best:
            best, best_v = val, cand
    print("  [%s] after random search (%d evals): objective %.2f %%" % (variant, used, best))
    sigma = 0.5
    while used < evals:
        improved = False
        for _ in range(4):
            if used >= evals:
                break
            cand = np.clip(best_v + sigma * rng.standard_normal(len(lo)), lo, hi)
            val = objective(cand, variant, runs)
            used += 1
            if val < best - 1e-9:
                best, best_v, improved = val, cand, True
        sigma = min(1.0, sigma * 1.4) if improved else max(0.02, sigma * 0.8)
    print("  [%s] final (%d evals): objective %.2f %%  %s" % (variant, used, best,
          {n: round(float(x), 2) for n, x in zip(PARAM_NAMES, best_v)}))
    return best_v, best


def evaluate(runs, params):
    rows = []
    for ri, d in enumerate(runs):
        m = d["meta"]
        for case in MISMATCH_CASES:
            base_row = dict(run=ri, pwm=m["pwm"], w0=m["w0"], direction=m["direction"], case=case)
            naive_done = False
            for variant in VARIANTS:
                e = estimates(d, variant, params[variant], case)
                row = dict(base_row, variant=variant)
                for tag, tr, es in (("cam", "true_cam", "est_cam"), ("mot", "true_mot", "est_mot"),
                                    ("Tm", "true_Tm", "est_Tm")):
                    for k, v in metrics(e[tr], e[es]).items():
                        row["%s_%s" % (tag, k)] = v
                rows.append(row)
                if not naive_done:                      # baseline: Kt * i only (no dynamics), motor side / camshaft side
                    naive_done = True
                    nrow = dict(base_row, variant="current_only")
                    for tag, tr, es in (("cam", "true_cam", None), ("mot", "true_mot", "naive_mot")):
                        est = e["naive_mot"] / base.Gload if tag == "cam" else e["naive_mot"]
                        for k, v in metrics(e[tr], est).items():
                            nrow["%s_%s" % (tag, k)] = v
                    rows.append(nrow)
    return rows


def summarize(rows):
    keys = ("cam_RMSE", "cam_RMSE_pct", "cam_R2", "cam_Bias", "cam_MaxAbs", "mot_RMSE", "mot_R2")
    print("\n%-10s %-13s | %9s %8s %7s %9s %9s | %10s %7s" %
          ("case", "method", "RMSE cam", "RMSE %", "R2 cam", "bias cam", "max cam", "RMSE mot", "R2 mot"))
    print("%-10s %-13s | %9s %8s %7s %9s %9s | %10s %7s" %
          ("", "", "[N*m]", "of rms", "", "[N*m]", "[N*m]", "[N*m]", ""))
    for case in MISMATCH_CASES:
        for variant in ("current_only",) + VARIANTS:
            sel = [r for r in rows if r["case"] == case and r["variant"] == variant]
            m = {k: float(np.mean([r[k] for r in sel])) for k in keys}
            print("%-10s %-13s | %9.3f %8.1f %7.3f %9.3f %9.3f | %10.5f %7.3f" %
                  (case, variant, m["cam_RMSE"], m["cam_RMSE_pct"], m["cam_R2"], m["cam_Bias"], m["cam_MaxAbs"],
                   m["mot_RMSE"], m["mot_R2"]))
        print()


def plot_examples(runs, params, filename):
    import matplotlib.pyplot as plt
    pick = []
    for direction in (1, -1):
        for pwm in sorted({r["meta"]["pwm"] for r in runs}):
            c = [i for i, r in enumerate(runs) if r["meta"]["direction"] == direction and r["meta"]["pwm"] == pwm]
            if c:
                pick.append(c[0])
    pick = pick[:6]
    fig, axes = plt.subplots(len(pick), 2, figsize=(14, 2.8 * len(pick)), squeeze=False)
    for row, ri in enumerate(pick):
        d = runs[ri]
        for col, case in enumerate(MISMATCH_CASES):
            ax = axes[row, col]
            for variant, style in (("interp", "-"), ("no_interp", "--")):
                e = estimates(d, variant, params[variant], case)
                if variant == "interp":
                    ax.plot(e["cam"], e["true_cam"], "k", linewidth=2.0, label="true Tab")
                    ax.plot(e["cam"], e["naive_mot"] / base.Gload, ":", color="gray", label="Kt*i only")
                ax.plot(e["cam"], e["est_cam"], style, linewidth=1.1, label="KF " + variant)
            m = d["meta"]
            ax.set_title("PWM %.2f  dir %+d  w0 %g  [%s]" % (m["pwm"], m["direction"], m["w0"], case), fontsize=9)
            ax.set_ylabel("camshaft torque [N*m]")
            ax.grid(True, alpha=0.4)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("cam angle [deg]")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print("saved %s" % filename)


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small data set and few evaluations")
    ap.add_argument("--evals", type=int, help="tuning evaluations per variant (default 120, quick 30)")
    ap.add_argument("--noise-i", type=float, default=0.0, help="std of noise added to the measured current [A]")
    ap.add_argument("--regen", action="store_true", help="ignore the data cache")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="torque_kf", help="prefix of the output files")
    a = ap.parse_args()

    if a.quick:
        train_pwm, train_w0, test_pwm, test_w0, evals = (0.06, 0.2), (0.0,), (0.1,), (30.0,), 30
    else:
        train_pwm, train_w0, test_pwm, test_w0, evals = (0.06, 0.1, 0.15, 0.2), (0.0, 30.0), (0.08, 0.12, 0.18), (15.0, 45.0), 120
    if a.evals:
        evals = a.evals
    t0 = time.time()
    print("training runs:")
    train = build_dataset(train_pwm, train_w0, a.noise_i, a.regen, a.out + "_train.pkl", "train")
    print("test runs:")
    test = build_dataset(test_pwm, test_w0, a.noise_i, a.regen, a.out + "_test.pkl", "test")
    print("data ready (%.0f s)\n" % (time.time() - t0))

    params, tuned = {}, {}
    for variant in VARIANTS:
        t1 = time.time()
        v, best = tune(variant, train, evals)
        params[variant] = vec_to_params(v)
        tuned[variant] = {"objective_pct_of_rms_load": best, "log10": dict(zip(PARAM_NAMES, map(float, v))),
                          "values": params[variant]}
        print("  [%s] tuned in %.0f s: %s\n" % (variant, time.time() - t1, {k: "%.3g" % x for k, x in params[variant].items()}))
    with open(a.out + "_params.json", "w") as f:
        json.dump(tuned, f, indent=2)
    print("saved %s_params.json" % a.out)

    rows = evaluate(test, params)
    summarize(rows)
    with open(a.out + "_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved %s_results.csv" % a.out)
    if not a.no_plot:
        try:
            plot_examples(test, params, a.out + ".png")
        except ImportError:
            print("matplotlib not installed -> no plot")
