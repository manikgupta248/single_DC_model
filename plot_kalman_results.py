"""
plot_kalman_results.py - plots the results of the Kalman filter (variant no_interp = Hall edges only)
for a constant PWM of 0.12 and four start angles of the cam.

For every start angle the cam moves 40 deg forward (e.g. 20 -> 60 deg) and the script makes
two SEPARATE figures, so 8 figures in total:
    kalman_torque_start<angle>.png : true camshaft torque and Kalman filter estimate against the cam angle
    kalman_gain_start<angle>.png   : gain of the torque state for the current update against time

The gain K is the number by which the current error (measured - predicted, in A) is multiplied to correct
the torque estimate:   torque_new = torque_pred + K * current_error      [K in N*m (motor side) per A]

Needs: numpy, matplotlib, and emotor_cam_move.py + torque_kf.py from the repository (it runs the
simulation of the motor model itself).

    python plot_kalman_results.py                  make the 8 figures, save them and show them
    python plot_kalman_results.py --no-show        only save the png files
    python plot_kalman_results.py --out-dir plots  save into the folder "plots"
    python plot_kalman_results.py --skip-ms 0      metrics without leaving out the first 20 ms
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torque_kf as k            # Kalman filter + simulation helpers (uses emotor_cam_move.py unchanged)

import matplotlib
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

PWM = 0.12
START_ANGLES = (20.0, 110.0, 200.0, 290.0)      # deg of cam angle, forward sweep of 40 deg each
VARIANT = "no_interp"                           # Hall edges: exact edge angle + speed from the edge timing
# tuned noise settings of the variant no_interp (from the tuning run of torque_kf.py)
PARAMS = dict(q_w=1.8558420108369016, q_i=0.6002396839788338, q_T=0.11826244543066099,
              r_i=1.4195516188691276e-08, r_th=1.931481126190632e-05, r_w=152.87458228211878)


def run_filter_with_gain(data, p):
    """The filter of torque_kf.run_filter (variant no_interp), written out here so that this script also works with
    an older torque_kf.py. It additionally records the torque gain K of the current update at every step."""
    b = k.base
    sl = slice(None, None, k.KF_DECIMATION)
    t = data["t"][sl]
    z_i = data["i_meas"][sl]
    Q = data["Camshaft_quantanizer"][sl]
    dt = float(t[1] - t[0])
    F, Bd, Qd = k.build_model(dt, p["q_w"], p["q_i"], p["q_T"], {})
    Bu = Bd * data["V"]
    q = b.QuantizationInterval
    n = len(t)
    x = np.array([data["Wi1"][0], 0.0, 0.0, 0.0])
    P = np.diag([1e-6, 1e-4, 1e-4, 0.05 ** 2])
    out = np.zeros((n, 4))
    out[0] = x
    gains = []
    t_edge_prev, s_prev = None, 0.0
    for i in range(1, n):
        x = F @ x + Bu                                   # predict
        P = F @ P @ F.T + Qd
        S = P[2, 2] + p["r_i"]                           # current update, every step
        K = P[:, 2] / S
        gains.append(float(K[3]))                        # gain of the torque state
        x = x + K * (z_i[i] - x[2])
        P = P - np.outer(K, P[2])
        if Q[i] == Q[i - 1]:                             # no Hall edge in this step
            out[i] = x
            continue
        s = 1.0 if Q[i] > Q[i - 1] else -1.0             # Hall edge: angle update
        w_hat = x[1]
        z = Q[i] - s * q / 2 + w_hat * dt / 2
        r = p["r_th"] + (w_hat * dt) ** 2 / 12
        K = P[:, 0] / (P[0, 0] + r)
        x = x + K * (z - x[0])
        P = P - np.outer(K, P[0])
        t_edge = t[i] - dt / 2                           # speed from the time between two edges
        if t_edge_prev is not None and s == s_prev:
            dte = t_edge - t_edge_prev
            w_meas = s * q / dte
            wdot = (b.Ktc * x[2] - b.Bc * x[1] - x[3]) / b.J
            r_w = p["r_w"] + (wdot * dte / 2) ** 2 + (w_meas * dt / dte) ** 2 / 6
            K = P[:, 1] / (P[1, 1] + r_w)
            x = x + K * (w_meas - x[1])
            P = P - np.outer(K, P[1])
        t_edge_prev, s_prev = t_edge, s
        P = 0.5 * (P + P.T)
        out[i] = x
    return out, gains


def run_case(angle):
    """Simulate one sweep and run the Kalman filter on it. Returns a dict with everything to plot."""
    data = k.generate_run(PWM, angle, +1)
    if data is None:
        raise RuntimeError("the cam did not move 40 deg for start angle %g" % angle)
    x, gains = run_filter_with_gain(data, PARAMS)                    # estimates (n x 4), gains (n-1)
    ref = k.run_filter(data, VARIANT, PARAMS, {})                    # check against the filter of torque_kf.py
    if np.abs(ref - x).max() > 1e-9:
        print("WARNING: the filter in this script differs from torque_kf.run_filter by %.2e" % np.abs(ref - x).max())
    sl = slice(None, None, k.KF_DECIMATION)
    t_ms = (data["t"][sl] - data["t"][0]) * 1000.0
    return dict(angle=angle,
                t_ms=t_ms,
                cam=data["Gdeg"][sl],                                 # actual cam angle [deg]
                true_cam=data["Tab"][sl],                             # true camshaft torque [N*m]
                est_cam=x[:, 3] / k.base.Gload,                       # filter estimate at the camshaft [N*m]
                gain=np.array(gains))                                 # torque gain, steps 1 ... n-1


def torque_figure(c, skip_ms, out_dir, show):
    keep = c["t_ms"] >= skip_ms
    m = k.metrics(c["true_cam"][keep], c["est_cam"][keep])
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.plot(c["cam"], c["true_cam"], color="#555555", lw=4.0, label="true torque (simulation)")
    ax.plot(c["cam"][1:], c["est_cam"][1:], color="#2e7d32", lw=1.8, ls="--",
            label="Kalman filter estimate (%s)" % VARIANT)
    up = c["est_cam"][1] < 0                                      # put the text where the curve is not
    ax.annotate("start-up of the filter:\nthe estimate needs about 1 ms\nto reach the true torque", xy=(c["cam"][1], c["est_cam"][1]),
                xytext=(10, 175 if up else -110), textcoords="offset points", fontsize=9.5, color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32"), va="center")
    ax.set_xlabel("cam angle [deg]", fontsize=11)
    ax.set_ylabel("camshaft torque [N*m]", fontsize=11)
    note = "RMSE %.3f N*m = %.2f %% of the rms torque, R2 = %.4f" % (m["RMSE"], m["RMSE_pct"], m["R2"])
    if skip_ms > 0:
        note += "  (first %g ms left out)" % skip_ms
    ax.set_title("Start angle %g deg, PWM %.2f: true and estimated camshaft torque\n%s" % (c["angle"], PWM, note), fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "kalman_torque_start%d.png" % round(c["angle"]))
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    return m, fig


def gain_figure(c, out_dir):
    t = c["t_ms"][1:]
    g = c["gain"]
    imin = int(np.argmin(g[1:20])) + 1                          # undershoot after the start-up
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.semilogx(t, g, color="#c0392b", lw=1.8)
    ax.semilogx(t[:15], g[:15], "o", color="#c0392b", ms=5)
    ax.set_xlabel("time since the start of the movement [ms], log scale (one filter step = 0.2 ms)", fontsize=11)
    ax.set_ylabel("gain of the torque for the current error\n[N*m (motor side) per A]", fontsize=11)
    ax.set_title("Start angle %g deg, PWM %.2f: how the torque gain varies" % (c["angle"], PWM), fontsize=12.5)
    ax.grid(True, alpha=0.4, which="both")
    ax.set_ylim(0, g.max() * 1.15)
    ax.annotate("step 1 (0.2 ms): %.2f" % g[0], xy=(t[0], g[0]), xytext=(t[0] * 1.6, g[0] * 0.97),
                fontsize=10.5, color="#a13030", arrowprops=dict(arrowstyle="->", color="#a13030"))
    ax.annotate("step 3 (0.6 ms): %.2f" % g[2], xy=(t[2], g[2]), xytext=(t[2] * 2.6, g[2] * 1.5),
                fontsize=10.5, color="#a13030", arrowprops=dict(arrowstyle="->", color="#a13030"))
    ax.annotate("small undershoot %.3f at %.1f ms" % (g[imin], t[imin]), xy=(t[imin], g[imin]),
                xytext=(t[imin] * 2.2, g[imin] - 0.16), fontsize=10.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.axhline(g[-1], color="gray", ls=":", lw=1.2)
    ax.text(t[-1], g[-1] + 0.05, "settled value %.3f (constant from about 2 ms on)" % g[-1], ha="right", fontsize=10,
            color="#555555")
    fig.tight_layout()
    path = os.path.join(out_dir, "kalman_gain_start%d.png" % round(c["angle"]))
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-show", action="store_true", help="save the png files but do not open the figures")
    ap.add_argument("--out-dir", default=".", help="folder for the png files (default: current folder)")
    ap.add_argument("--skip-ms", type=float, default=20.0, help="start-up time left out of the metrics [ms] (default 20)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cases = []
    for angle in START_ANGLES:
        print("start angle %g deg: simulating the movement and running the filter ..." % angle)
        cases.append(run_case(angle))

    print()
    print("%-12s %-12s %-26s %-24s" % ("start angle", "duration", "torque error (rms)", "settled torque gain"))
    for c in cases:
        keep = c["t_ms"] >= a.skip_ms
        m = k.metrics(c["true_cam"][keep], c["est_cam"][keep])
        print("%-12s %-12s %-26s %-24s" % ("%g deg" % c["angle"], "%.3f s" % (c["t_ms"][-1] / 1000),
                                           "%.3f N*m (%.2f %%)" % (m["RMSE"], m["RMSE_pct"]), "%.3f" % c["gain"][-1]))
    print()
    for i, j in ((0, 2), (1, 3)):
        n = min(len(cases[i]["est_cam"]), len(cases[j]["est_cam"]))
        d = np.abs(cases[i]["est_cam"][:n] - cases[j]["est_cam"][:n]).max()
        print("estimates for %g deg and %g deg differ by at most %.2e N*m (the cam torque repeats every 60 deg)"
              % (cases[i]["angle"], cases[j]["angle"], d))
    print()

    for c in cases:
        torque_figure(c, a.skip_ms, a.out_dir, not a.no_show)
    for c in cases:
        gain_figure(c, a.out_dir)
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
