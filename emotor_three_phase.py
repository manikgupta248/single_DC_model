# =============================================================================
# emotor_three_phase.py  -  SECONDARY script: three-phase part of
#                           emotor_cam_move_3_phase.slx
#
# The main script  emotor_cam_move.py  is NOT changed. This script imports it and re-uses all its
# blocks and parameters (controller, DC-equivalent plant, cam load table, quantizer, Hall option, ...),
# runs the same simulation (same fixed step h, same RK4) and ADDITIONALLY calculates, exactly like
# the three-phase branch of the Simulink model:
#
#   Wi1 (motor angle)  -> Mod 2*pi -> x Pole_pairs (6) -> Mod 2*pi -> deg      = elec_deg
#   elec_deg           -> MATLAB Function "hall_sensors"                         = H1, H2, H3
#   H1,H2,H3, Vapp2    -> MATLAB Function "inverter_commutation" (six-step)      = Va, Vb, Vc
#                         Vapp2 = Vdc1 * PWM, V_half = Vapp2 / 2
#   elec_deg           -> lookups Ke_a / Ke_b / Ke_c [V/(rad/s)]  x  Wi          = e_a, e_b, e_c
#   per phase x:       L_ph * d(i_x)/dt = V_x - e_x - R_ph * i_x      (R_ph = 0.16/2, L_ph = 0.000066/2)
#
# The three-phase branch does not act back on the DC-equivalent model (Wi, PWM come from it), exactly
# as in the Simulink model. The phase currents are integrated together with the other states in the same RK4
# step; the Hall/commutation voltages are held constant during one step (like the controller output).
#
# Results (log keys): see LOG_NAMES. Extra (not in Simulink): line-to-line back-EMFs, sum of the phase
# currents, electrical power and the torque calculated from it.
#
# Usage:
#   python emotor_three_phase.py                 run 8 s (StopTime of the main script), plot, save csv + png
#   python emotor_three_phase.py --stop-time 2   shorter run
#   python emotor_three_phase.py --check         also verify that Li, Wi, PWM equal the main script's result
#   from emotor_three_phase import simulate_three_phase;  log = simulate_three_phase()
# =============================================================================
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotor_cam_move as base          # main model, used unchanged

# ----------------------------------------------------------------------------
# Three-phase parameters (top level of emotor_cam_move_3_phase.slx)
# ----------------------------------------------------------------------------
Rc_ph = 0.1600 / 2          # Constant Rc1/Rc2/Rc3  [Ohm]
Lc_ph = 0.000066 / 2        # Constant Lc1/Lc2/Lc3  [H]
Vdc1 = 13.5                 # Constant Vdc1
Pole_pairs = 6              # Gain Pole_pairs (subsystem "E-motor velocity and angle")
TWO_PI = 2 * math.pi        # Constant "Constant" (2*pi) used by Mod / Mod1

# Lookup_n-D  Ke_a / Ke_b / Ke_c : breakpoints -6:12:354 [electrical deg], linear, extrapolation Clip
Ke_bp_start = -6.0
Ke_bp_step = 12.0
Ke_a_table = [0.010610, 0.007597, 0.006207, 0.005337, 0.004402, 0.002985, -0.003013, -0.004403, -0.005273, -0.006208,
              -0.007625, -0.010610, -0.010610, -0.010610, -0.010610, -0.010610, -0.007597, -0.006207, -0.005337,
              -0.004402, -0.002985, 0.003013, 0.004403, 0.005273, 0.006208, 0.007625, 0.010610, 0.010610, 0.010610,
              0.010610, 0.010610]
Ke_b_table = [-0.002985, 0.003013, 0.004403, 0.005273, 0.006208, 0.007625, 0.010610, 0.010610, 0.010610, 0.010610,
              0.010610, 0.007597, 0.006208, 0.005337, 0.004402, 0.002985, -0.003013, -0.004403, -0.005273, -0.006208,
              -0.007625, -0.010610, -0.010610, -0.010610, -0.010610, -0.010610, -0.007597, -0.006207, -0.005337,
              -0.004402, -0.002985]
Ke_c_table = [-0.007625, -0.010610, -0.010610, -0.010610, -0.010610, -0.010610, -0.007597, -0.006207, -0.005337,
              -0.004402, -0.002985, 0.003013, 0.004403, 0.005273, 0.006208, 0.007625, 0.010610, 0.010610, 0.010610,
              0.010610, 0.010610, 0.007597, 0.006207, 0.005337, 0.004402, 0.002985, -0.003013, -0.004403, -0.005273,
              -0.006208, -0.007625]
_KE_N = len(Ke_a_table)     # 31 points


def Ke_lookup(table, deg):
    """Lookup_n-D, 1-D, linear interpolation, extrapolation Clip."""
    pos = (deg - Ke_bp_start) / Ke_bp_step
    if pos <= 0.0:
        return table[0]
    if pos >= _KE_N - 1:
        return table[_KE_N - 1]
    k = int(pos)
    return table[k] + (pos - k) * (table[k + 1] - table[k])


# ----------------------------------------------------------------------------
# Subsystem "E-motor velocity and angle": electrical angle output 2
#   Wi1 -> Mod(.,2*pi) -> Pole_pairs -> Mod1(.,2*pi) -> Grpm2 (180/pi)
# ----------------------------------------------------------------------------
def electrical_angle_deg(Wi1):
    return (((Wi1 % TWO_PI) * Pole_pairs) % TWO_PI) * (180.0 / math.pi)


# ----------------------------------------------------------------------------
# MATLAB Function "hall_sensors"  (input: elec_deg)
# ----------------------------------------------------------------------------
def hall_sensors(elec_deg):
    theta = elec_deg % 360.0
    if 0 <= theta < 60:
        return 1, 0, 1          # sector 1 (101)
    elif 60 <= theta < 120:
        return 1, 0, 0          # sector 2 (100)
    elif 120 <= theta < 180:
        return 1, 1, 0          # sector 3 (110)
    elif 180 <= theta < 240:
        return 0, 1, 0          # sector 4 (010)
    elif 240 <= theta < 300:
        return 0, 1, 1          # sector 5 (011)
    return 0, 0, 1              # sector 6 (001)


# ----------------------------------------------------------------------------
# MATLAB Function "inverter_commutation"  (inputs: H1, H2, H3, V_in)
# ----------------------------------------------------------------------------
def inverter_commutation(H1, H2, H3, V_in):
    state = H1 * 4 + H2 * 2 + H3 * 1
    V_half = V_in / 2
    if state == 5:      # 101: A+, B-, C open
        return V_half, -V_half, 0.0
    if state == 4:      # 100: A+, C-, B open
        return V_half, 0.0, -V_half
    if state == 6:      # 110: B+, C-, A open
        return 0.0, V_half, -V_half
    if state == 2:      # 010: B+, A-, C open
        return -V_half, V_half, 0.0
    if state == 3:      # 011: C+, A-, B open
        return -V_half, 0.0, V_half
    if state == 1:      # 001: C+, B-, A open
        return 0.0, -V_half, V_half
    return 0.0, 0.0, 0.0


# ----------------------------------------------------------------------------
# State derivatives: main model (Li, Wi, Aint, Wi1) + three phase currents (ia, ib, ic)
# ----------------------------------------------------------------------------
def derivatives(x, Vapp, Vph):
    Li, Wi, Aint, Wi1, ia, ib, ic = x
    d = base.plant_derivatives(Li, Wi, Aint, Wi1, Vapp)       # main model, unchanged
    deg = electrical_angle_deg(Wi1)
    ea = Ke_lookup(Ke_a_table, deg) * Wi                      # Product e_a
    eb = Ke_lookup(Ke_b_table, deg) * Wi                      # Product e_b
    ec = Ke_lookup(Ke_c_table, deg) * Wi                      # Product e_c
    return d + ((Vph[0] - ea - Rc_ph * ia) / Lc_ph,           # Vsum1 / Lc1  -> Li1 (i_a)
                (Vph[1] - eb - Rc_ph * ib) / Lc_ph,           # Vsum2 / Lc2  -> Li2 (i_b)
                (Vph[2] - ec - Rc_ph * ic) / Lc_ph)           # Vsum3 / Lc3  -> Li3 (i_c)


def rk4_step(x, Vapp, Vph, dt):
    k1 = derivatives(x, Vapp, Vph)
    k2 = derivatives(tuple(xi + 0.5 * dt * k for xi, k in zip(x, k1)), Vapp, Vph)
    k3 = derivatives(tuple(xi + 0.5 * dt * k for xi, k in zip(x, k2)), Vapp, Vph)
    k4 = derivatives(tuple(xi + dt * k for xi, k in zip(x, k3)), Vapp, Vph)
    return tuple(xi + dt / 6.0 * (a + 2 * b + 2 * c + d)
                 for xi, a, b, c, d in zip(x, k1, k2, k3, k4))


# ----------------------------------------------------------------------------
# Simulation (same structure as emotor_cam_move.simulate(), plus the three-phase branch)
# ----------------------------------------------------------------------------
LOG_NAMES = ["t",
             # main (DC-equivalent) model
             "Li", "Wi", "Aint", "Wi1", "Gdeg", "PWM", "Vapp", "Tm",
             # three-phase branch (as in the Simulink model)
             "elec_deg", "H1", "H2", "H3", "Vapp2", "Va", "Vb", "Vc", "ea", "eb", "ec", "ia", "ib", "ic",
             # extras (not in Simulink)
             "eab", "ebc", "eca", "i_sum", "P_elec", "Tm_3ph"]


def simulate_three_phase():
    h = base.h
    n_steps = int(round((base.StopTime - base.StartTime) / h))
    rt_every = int(round(base.RT_Ts / h))
    ctrl = base.Controller()
    hall = base.HallInterpolator()
    mode = base.CONTROL_MODE.strip().upper()
    if mode not in ("CONTROLLER", "CONSTANT"):
        raise ValueError('CONTROL_MODE must be "CONTROLLER" or "CONSTANT", got %r' % (base.CONTROL_MODE,))

    Aint0 = math.radians(base.START_ANGLE_DEG)
    x = (0.0, 0.0, Aint0, Aint0 * 73, 0.0, 0.0, 0.0)           # Li, Wi, Aint, Wi1, ia, ib, ic
    log = {k: [] for k in LOG_NAMES}

    for k in range(n_steps + 1):
        t = base.StartTime + k * h
        Li, Wi, Aint, Wi1, ia, ib, ic = x

        # ---- main model signals (identical to emotor_cam_move.simulate) ----
        Gdeg_out = Aint * base.Gdeg
        Gload_out = base.Tab(Gdeg_out) * base.Gload
        Quant_out = base.Camshaft_quantanizer(Wi1)
        Hall_out = hall.output(Quant_out, t)
        Gcam1_out = Hall_out * base.Gcam1
        Gdeg1_out = base.Target_Cam_angle(t) * base.Gdeg1
        if k % rt_every == 0:
            ctrl.sample_rate_transitions(base.ThetaTC1, Gdeg1_out, Gcam1_out)
        PWM = ctrl.output_and_update()
        if mode == "CONSTANT":
            PWM = base.CONSTANT_PWM
        Vapp = base.Vdc * PWM

        # ---- three-phase branch ----
        deg = electrical_angle_deg(Wi1)
        H1, H2, H3 = hall_sensors(deg)
        Vapp2 = Vdc1 * PWM
        Va, Vb, Vc = inverter_commutation(H1, H2, H3, Vapp2)

        if k % base.LOG_DECIMATION == 0:
            ea = Ke_lookup(Ke_a_table, deg) * Wi
            eb = Ke_lookup(Ke_b_table, deg) * Wi
            ec = Ke_lookup(Ke_c_table, deg) * Wi
            P = ea * ia + eb * ib + ec * ic
            vals = dict(t=t, Li=Li, Wi=Wi, Aint=Aint, Wi1=Wi1, Gdeg=Gdeg_out, PWM=PWM, Vapp=Vapp,
                        Tm=base.Ktc * Li, elec_deg=deg, H1=H1, H2=H2, H3=H3, Vapp2=Vapp2, Va=Va, Vb=Vb, Vc=Vc,
                        ea=ea, eb=eb, ec=ec, ia=ia, ib=ib, ic=ic,
                        eab=ea - eb, ebc=eb - ec, eca=ec - ea, i_sum=ia + ib + ic, P_elec=P,
                        Tm_3ph=P / Wi if abs(Wi) > 1e-3 else 0.0)
            for key in LOG_NAMES:
                log[key].append(vals[key])

        if k < n_steps:
            x = rk4_step(x, Vapp, (Va, Vb, Vc), h)

    return log


# ----------------------------------------------------------------------------
# Helpers: check against the main script, csv export, plots
# ----------------------------------------------------------------------------
def check_against_main(stop_time=0.2):
    """Run the main script and this script with the same settings and compare Li, Wi, PWM."""
    saved = base.StopTime
    base.StopTime = stop_time
    try:
        ref = base.simulate()
        new = simulate_three_phase()
    finally:
        base.StopTime = saved
    ok = True
    for key in ("Li", "Wi", "Aint", "Wi1", "PWM"):
        d = max(abs(a - b) for a, b in zip(ref[key], new[key]))
        print("  check %-5s max |main - three_phase| = %.3e" % (key, d))
        ok = ok and d < 1e-9
    print("check against emotor_cam_move.py (StopTime %g s): %s" % (stop_time, "OK" if ok else "MISMATCH"))
    return ok


def save_csv(log, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_NAMES)
        for row in zip(*[log[k] for k in LOG_NAMES]):
            w.writerow(["%.9g" % v for v in row])
    print("saved %s" % filename)


def plot(log, filename="three_phase.png", show=True):
    import matplotlib.pyplot as plt
    t = log["t"]
    fig, ax = plt.subplots(4, 2, figsize=(15, 11), sharex=True)
    ax[0, 0].plot(t, log["PWM"]); ax[0, 0].set_ylabel("PWM [-]")
    ax[0, 1].plot(t, log["elec_deg"]); ax[0, 1].set_ylabel("electrical angle [deg]")
    for key, c in (("Va", "tab:red"), ("Vb", "tab:green"), ("Vc", "tab:blue")):
        ax[1, 0].plot(t, log[key], color=c, label=key)
    ax[1, 0].set_ylabel("phase voltages [V]"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, log["H1"], label="H1"); ax[1, 1].plot(t, log["H2"], label="H2")
    ax[1, 1].plot(t, log["H3"], label="H3"); ax[1, 1].set_ylabel("Hall signals"); ax[1, 1].legend(fontsize=8)
    for key, c in (("ea", "tab:red"), ("eb", "tab:green"), ("ec", "tab:blue")):
        ax[2, 0].plot(t, log[key], color=c, label=key)
    ax[2, 0].set_ylabel("back-EMF [V]"); ax[2, 0].legend(fontsize=8)
    for key, c in (("ia", "tab:red"), ("ib", "tab:green"), ("ic", "tab:blue")):
        ax[2, 1].plot(t, log[key], color=c, label=key)
    ax[2, 1].set_ylabel("phase currents [A]"); ax[2, 1].legend(fontsize=8)
    ax[3, 0].plot(t, log["Li"], label="Li (DC-equivalent model)")
    ax[3, 0].plot(t, log["i_sum"], label="ia+ib+ic", linewidth=0.8)
    ax[3, 0].set_ylabel("current [A]"); ax[3, 0].legend(fontsize=8)
    ax[3, 1].plot(t, log["Tm"], label="Tm = Ktc*Li (DC-equivalent)")
    ax[3, 1].plot(t, log["Tm_3ph"], label="Tm from 3-phase power", linewidth=0.8)
    ax[3, 1].set_ylabel("motor torque [N*m]"); ax[3, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(True, alpha=0.4)
    ax[3, 0].set_xlabel("Time [s]"); ax[3, 1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    print("saved %s" % filename)
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Three-phase currents / back-EMFs of emotor_cam_move_3_phase.slx")
    ap.add_argument("--stop-time", type=float, help="override StopTime of the main script [s]")
    ap.add_argument("--hall-mode", choices=["INTERPOLATION", "NOINTERPOLATION"], help="override HALL_MODE")
    ap.add_argument("--check", action="store_true", help="verify against emotor_cam_move.simulate() first")
    ap.add_argument("--out", default="three_phase", help="prefix of the output files")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()
    if a.stop_time is not None:
        base.StopTime = a.stop_time
    if a.hall_mode:
        base.HALL_MODE = a.hall_mode
    print("CONTROL_MODE = %s  HALL_MODE = %s  StopTime = %g s  h = %g s"
          % (base.CONTROL_MODE, base.HALL_MODE, base.StopTime, base.h))
    if a.check:
        check_against_main()
    log = simulate_three_phase()
    n = len(log["t"])
    print("simulated %d logged samples" % n)
    for key in ("ia", "ib", "ic", "ea", "eb", "ec"):
        print("  %-3s  min %9.4f   max %9.4f   rms %9.4f"
              % (key, min(log[key]), max(log[key]), (sum(v * v for v in log[key]) / n) ** 0.5))
    if not a.no_csv:
        save_csv(log, a.out + ".csv")
    if not a.no_plot:
        try:
            plot(log, a.out + ".png")
        except ImportError:
            print("matplotlib not installed -> no plot")
