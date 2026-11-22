"""
plot_kalman_results.py - plots the results of the Kalman filter (variant no_interp = Hall edges only)
for a constant PWM (default 0.12, any value with --pwm) and four start angles of the cam.

For every start angle the cam moves 40 deg forward (e.g. 20 -> 60 deg) and the script makes
four SEPARATE figures (16 figures for the four start angles at one PWM):
    kalman_torque_pwm<PWM>_start<angle>.png       : true camshaft torque and Kalman filter estimate against the cam angle
    kalman_gain_pwm<PWM>_start<angle>.png         : gain of the TORQUE for the current update against time
    kalman_gain_current_pwm<PWM>_start<angle>.png : gain of the CURRENT for the current update against time
    kalman_gain_angle_pwm<PWM>_start<angle>.png   : gain of the ANGLE for the Hall-angle update, at every Hall edge

Every measurement error (measured - predicted) corrects each state with its own gain K:
    state_new = state_pred + K * error
    K of the torque : [N*m (motor side) per A], can be larger than 1 (unit conversion)
    K of the current: fraction between 0 and 1 (how much of the current error the current estimate accepts)
    K of the angle  : fraction between 0 and 1 (how much of the angle error the angle estimate accepts);
                      in the variant no_interp the angle is measured only when a Hall edge occurs

Needs: numpy, matplotlib, and emotor_cam_move.py + torque_kf.py from the repository (it runs the
simulation of the motor model itself).

    python plot_kalman_results.py                  make the figures, save them and show them
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

PWM_DEFAULT = (0.12,)                           # constant PWM duty; change with --pwm, e.g. --pwm 0.12 0.18
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
    gains = []                                           # torque gain of the current update
    gains_i = []                                         # gain of the current itself in the current update
    edge_t, gains_a, edge_r = [], [], []                 # Hall edges: time [ms], gain of the angle, noise R of the angle
    t_edge_prev, s_prev = None, 0.0
    for i in range(1, n):
        x = F @ x + Bu                                   # predict
        P = F @ P @ F.T + Qd
        S = P[2, 2] + p["r_i"]                           # current update, every step
        K = P[:, 2] / S
        gains.append(float(K[3]))                        # gain of the torque state
        gains_i.append(float(K[2]))                      # gain of the current state
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
        edge_t.append(1000.0 * (t[i] - t[0]))
        gains_a.append(float(K[0]))                      # gain of the angle state in the angle update
        edge_r.append(float(r))
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
    return out, dict(torque=gains, current=gains_i, edge_t=edge_t, angle=gains_a, edge_r=edge_r)


def run_case(angle, pwm):
    """Simulate one sweep and run the Kalman filter on it. Returns a dict with everything to plot."""
    data = k.generate_run(pwm, angle, +1)
    if data is None:
        return None                                                  # the cam did not move 40 deg (PWM too low)
    x, glog = run_filter_with_gain(data, PARAMS)                     # estimates (n x 4) and the logged gains
    ref = k.run_filter(data, VARIANT, PARAMS, {})                    # check against the filter of torque_kf.py
    if np.abs(ref - x).max() > 1e-9:
        print("WARNING: the filter in this script differs from torque_kf.run_filter by %.2e" % np.abs(ref - x).max())
    sl = slice(None, None, k.KF_DECIMATION)
    t_ms = (data["t"][sl] - data["t"][0]) * 1000.0
    return dict(angle=angle, pwm=pwm,
                t_ms=t_ms,
                cam=data["Gdeg"][sl],                                 # actual cam angle [deg]
                true_cam=data["Tab"][sl],                             # true camshaft torque [N*m]
                est_cam=x[:, 3] / k.base.Gload,                       # filter estimate at the camshaft [N*m]
                gain=np.array(glog["torque"]),                        # torque gain of the current update, steps 1 ... n-1
                gain_i=np.array(glog["current"]),                     # gain of the current in the current update
                edge_t=np.array(glog["edge_t"]),                      # time of every Hall edge [ms]
                gain_a=np.array(glog["angle"]),                       # gain of the angle in the angle update at the edges
                edge_r=np.array(glog["edge_r"]))                      # noise R used for the angle at the edges


def both_metrics(c, skip_ms):
    """Metrics of the camshaft torque estimate: (all steps from the first step on, steps after the first skip_ms)."""
    first = np.arange(len(c["t_ms"])) >= 1                 # point 0 is only the starting guess, not an estimate
    late = c["t_ms"] >= skip_ms
    m_all = k.metrics(c["true_cam"][first], c["est_cam"][first])
    m_ex = k.metrics(c["true_cam"][late], c["est_cam"][late])
    return m_all, m_ex


def torque_figure(c, skip_ms, out_dir, show):
    m_all, m_ex = both_metrics(c, skip_ms)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.plot(c["cam"], c["true_cam"], color="#555555", lw=4.0, label="true torque (simulation)")
    ax.plot(c["cam"][1:], c["est_cam"][1:], color="#2e7d32", lw=1.8, ls="--",
            label="Kalman filter estimate (%s)" % VARIANT)
    up = c["est_cam"][1] < 0                                      # put the text where the curve is not
    ax.annotate("start-up of the filter:\nthe estimate needs about 1 ms\nto reach the true torque", xy=(c["cam"][1], c["est_cam"][1]),
                xytext=(10, 175 if up else -110), textcoords="offset points", fontsize=9.5, color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32"), va="center")
    ax.set_xlabel("cam angle [deg]", fontsize=11)
    ax.set_ylabel("camshaft torque [N*m]", fontsize=11)
    note = ("including the first %g ms:    RMSE %.3f N*m = %.2f %% of the rms torque,  R2 = %.4f\n"
            "without the first %g ms:   RMSE %.3f N*m = %.2f %% of the rms torque,  R2 = %.4f"
            % (skip_ms, m_all["RMSE"], m_all["RMSE_pct"], m_all["R2"], skip_ms, m_ex["RMSE"], m_ex["RMSE_pct"], m_ex["R2"]))
    ax.set_title("Start angle %g deg, PWM %.2f: true and estimated camshaft torque\n%s" % (c["angle"], c["pwm"], note), fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "kalman_torque_pwm%.2f_start%d.png" % (c["pwm"], round(c["angle"])))
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    return (m_all, m_ex), fig


def gain_figure(c, out_dir):
    t = c["t_ms"][1:]
    g = c["gain"]
    imin = int(np.argmin(g[1:20])) + 1                          # undershoot after the start-up
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.semilogx(t, g, color="#c0392b", lw=1.8)
    ax.semilogx(t[:15], g[:15], "o", color="#c0392b", ms=5)
    ax.set_xlabel("time since the start of the movement [ms], log scale (one filter step = 0.2 ms)", fontsize=11)
    ax.set_ylabel("gain of the torque for the current error\n[N*m (motor side) per A]", fontsize=11)
    ax.set_title("Start angle %g deg, PWM %.2f: how the torque gain varies" % (c["angle"], c["pwm"]), fontsize=12.5)
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
    path = os.path.join(out_dir, "kalman_gain_pwm%.2f_start%d.png" % (c["pwm"], round(c["angle"])))
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    return fig


def gain_current_figure(c, out_dir):
    t = c["t_ms"][1:]
    g = c["gain_i"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.semilogx(t, g, color="#1f4e79", lw=1.8)
    ax.semilogx(t[:15], g[:15], "o", color="#1f4e79", ms=5)
    ax.set_xlabel("time since the start of the movement [ms], log scale (one filter step = 0.2 ms)", fontsize=11)
    ax.set_ylabel("gain of the current for the current error\n(fraction of the error that is accepted, 0 to 1)", fontsize=11)
    ax.set_title("Start angle %g deg, PWM %.2f: how the gain of the current varies" % (c["angle"], c["pwm"]), fontsize=12.5)
    ax.grid(True, alpha=0.4, which="both")
    ax.set_ylim(0, 1.15)
    ax.text(0.5, 0.45, "gain between %.5f and %.5f over the whole movement:\nthe current measurement is trusted almost completely,\n"
            "the estimate takes over %.3f %% of every current error" % (g.min(), g.max(), 100 * g.min()),
            transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#1f4e79")
    ins = ax.inset_axes([0.08, 0.08, 0.5, 0.25])
    ins.semilogx(t, (1 - g) * 1e5, color="#c0392b", lw=1.4)
    ins.set_title("zoom: the part of the error that is NOT accepted, (1 - gain) x 1e5", fontsize=8.5)
    ins.set_ylim(0, (1 - g).max() * 1e5 * 1.3)
    ins.grid(True, alpha=0.4, which="both")
    ins.tick_params(labelsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "kalman_gain_current_pwm%.2f_start%d.png" % (c["pwm"], round(c["angle"])))
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    return fig


def gain_angle_figure(c, out_dir):
    t, g = c["edge_t"], c["gain_a"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(t, g, "o-", color="#6a3d9a", ms=3.5, lw=1.0)
    ax.set_xlabel("time since the start of the movement [ms]", fontsize=11)
    ax.set_ylabel("gain of the angle for the angle error\n(fraction of the error that is accepted, 0 to 1)", fontsize=11)
    ax.set_title("Start angle %g deg, PWM %.2f: how the gain of the angle varies\n(the angle is measured only at the %d Hall edges)"
                 % (c["angle"], c["pwm"], len(g)), fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, g.max() * 1.6)
    ax.text(0.97, 0.93, "gain between %.3f and %.3f, i.e. %.1f to %.1f %% of the angle error is accepted:\n"
            "the Hall angle is trusted only weakly, because its noise R (%.1e to %.1e rad^2)\n"
            "is much larger than the filter's own angle uncertainty (about 1e-6 rad^2)"
            % (g.min(), g.max(), 100 * g.min(), 100 * g.max(), c["edge_r"].min(), c["edge_r"].max()),
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#4a2a73")
    fig.tight_layout()
    path = os.path.join(out_dir, "kalman_gain_angle_pwm%.2f_start%d.png" % (c["pwm"], round(c["angle"])))
    fig.savefig(path, dpi=150)
    print("saved %s" % path)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-show", action="store_true", help="save the png files but do not open the figures")
    ap.add_argument("--out-dir", default=".", help="folder for the png files (default: current folder)")
    ap.add_argument("--skip-ms", type=float, default=20.0, help="start-up time [ms] that is left out in the second metric (default 20)")
    ap.add_argument("--pwm", type=float, nargs="+", default=list(PWM_DEFAULT),
                    help="constant PWM duty (one or more values, default 0.12). Below about 0.05 the cam cannot move 40 deg.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cases = []
    for pwm in a.pwm:
        for angle in START_ANGLES:
            print("PWM %.2f, start angle %g deg: simulating the movement and running the filter ..." % (pwm, angle))
            c = run_case(angle, pwm)
            if c is None:
                print("   the cam does not move 40 deg at PWM %.2f (too low) -> skipped" % pwm)
            else:
                cases.append(c)
    if not cases:
        raise SystemExit("no case could be run")

    print()
    print("Torque error (RMSE of the camshaft torque, in N*m and in % of the rms torque):")
    print("%-6s %-12s %-10s %-30s %-30s %-10s" % ("PWM", "start angle", "duration", "including the first %g ms" % a.skip_ms,
                                              "without the first %g ms" % a.skip_ms, "gains: torque / current (settled) / angle (range)"))
    for c in cases:
        m_all, m_ex = both_metrics(c, a.skip_ms)
        print("%-6s %-12s %-10s %-30s %-30s %-10s" % (
            "%.2f" % c["pwm"], "%g deg" % c["angle"], "%.3f s" % (c["t_ms"][-1] / 1000),
            "%.3f N*m (%.2f %%)" % (m_all["RMSE"], m_all["RMSE_pct"]),
            "%.3f N*m (%.2f %%)" % (m_ex["RMSE"], m_ex["RMSE_pct"]), "%.3f / %.5f / %.3f-%.3f" % (c["gain"][-1], c["gain_i"][-1], c["gain_a"].min(), c["gain_a"].max())))
    print()
    for pwm in a.pwm:
        cp = {c["angle"]: c for c in cases if c["pwm"] == pwm}
        for u, v in ((20.0, 200.0), (110.0, 290.0)):
            if u in cp and v in cp:
                n = min(len(cp[u]["est_cam"]), len(cp[v]["est_cam"]))
                d = np.abs(cp[u]["est_cam"][:n] - cp[v]["est_cam"][:n]).max()
                print("PWM %.2f: estimates for %g deg and %g deg differ by at most %.2e N*m (the cam torque repeats every 60 deg)"
                      % (pwm, u, v, d))
    print()

    for c in cases:
        _, fig = torque_figure(c, a.skip_ms, a.out_dir, not a.no_show)
        if a.no_show:
            plt.close(fig)                                  # do not keep the figure in memory
    for c in cases:
        fig = gain_figure(c, a.out_dir)
        if a.no_show:
            plt.close(fig)
    for c in cases:
        fig = gain_current_figure(c, a.out_dir)
        if a.no_show:
            plt.close(fig)
    for c in cases:
        fig = gain_angle_figure(c, a.out_dir)
        if a.no_show:
            plt.close(fig)
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
