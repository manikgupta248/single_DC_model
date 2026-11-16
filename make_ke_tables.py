"""
make_ke_tables.py - back-EMF tables (Ke_a, Ke_b, Ke_c) for the three-phase model, built from the
measured line-to-line back-EMF in  back_emf_1200rpm_20C.csv  (Ph1-Ph2, Ph2-Ph3, Ph3-Ph1 [V] over the
electrical position, measured at 1200 rpm and 20 degC).

Method
------
1) line-to-line -> phase (star equivalent, no zero-sequence component):
       e1 = (V12 - V31)/3     e2 = (V23 - V12)/3     e3 = (V31 - V23)/3
2) divide by the angular speed of the measurement (1200 rpm = 125.66 rad/s)  ->  Ke [V/(rad/s)]
3) optional shift of the electrical angle:  Ke_new(phi) = Ke_old(phi - shift)
   (a shift that is a multiple of 12 deg only re-orders the table values, any other shift interpolates
   linearly on the periodic table)

With shift = 0 the result equals the tables in emotor_cam_move_3_phase.slx (differences < 1e-6, the slx
stores 6 decimals). Those tables are about 90 deg out of phase with the Hall sectors used in the
model, therefore the default shift for the three-phase script is 84 deg.

Usage
-----
    python make_ke_tables.py                 # print MATLAB arrays for shift 84 and write ke_tables.csv
    python make_ke_tables.py --shift 0       # tables exactly as in the slx
    python make_ke_tables.py --shift 90      # exact 90 deg (linear interpolation between breakpoints)
    python make_ke_tables.py --check         # consistency checks of the measured data
"""
import argparse
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "back_emf_1200rpm_20C.csv")
SPEED_RPM = 1200.0
BP_START = -6.0          # first breakpoint [electrical deg]
BP_STEP = 12.0
N = 31                   # -6 ... 354 deg; the last point (354 = -6 + 360) repeats the first one


def load_line_to_line(path=DATA_FILE):
    pos, v12, v23, v31 = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pos.append(float(row["position_el_deg"]))
            v12.append(float(row["V12"]))
            v23.append(float(row["V23"]))
            v31.append(float(row["V31"]))
    return pos, v12, v23, v31


def _shift_periodic(values, shift_deg):
    """new(phi) = old(phi - shift_deg) on the periodic table (30 unique points, 12 deg spacing)."""
    u = values[:N - 1]
    m = N - 1
    s = shift_deg / BP_STEP
    k0 = math.floor(s)
    frac = s - k0
    out = []
    for k in range(m):
        a = u[(k - k0) % m]
        b = u[(k - k0 - 1) % m]
        out.append(a + frac * (b - a))
    return out + [out[0]]


def phase_ke(shift_deg=0.0, path=DATA_FILE):
    """Return (Ke_a, Ke_b, Ke_c), each a list of N values [V/(rad/s)] at BP_START + BP_STEP*k."""
    pos, v12, v23, v31 = load_line_to_line(path)
    w = SPEED_RPM * 2 * math.pi / 60.0
    ea = [(a - c) / 3.0 / w for a, c in zip(v12, v31)]
    eb = [(b - a) / 3.0 / w for a, b in zip(v12, v23)]
    ec = [(c - b) / 3.0 / w for b, c in zip(v23, v31)]
    if shift_deg % 360.0 == 0.0:
        return ea, eb, ec
    return (_shift_periodic(ea, shift_deg), _shift_periodic(eb, shift_deg), _shift_periodic(ec, shift_deg))


def matlab_arrays(tables):
    bp = "[" + ", ".join("%g" % (BP_START + BP_STEP * k) for k in range(N)) + "]"
    out = ["% breakpoints (electrical deg), identical for Ke_a / Ke_b / Ke_c", bp]
    for name, t in zip("abc", tables):
        out.append("%% Table of Ke_%s [V/(rad/s)]" % name)
        out.append("[" + ", ".join("%.6f" % v for v in t) + "]")
    return "\n".join(out)


def check_data(path=DATA_FILE):
    pos, v12, v23, v31 = load_line_to_line(path)
    u12, u23, u31 = v12[:N - 1], v23[:N - 1], v31[:N - 1]
    sym23 = max(abs(u23[k] - u12[(k - 10) % 30]) for k in range(30))
    sym31 = max(abs(u31[k] - u12[(k - 20) % 30]) for k in range(30))
    print("120 deg symmetry:  max |V23 - V12 shifted 120 deg| = %.2e V,  max |V31 - V12 shifted 240 deg| = %.2e V"
          % (sym23, sym31))
    ksum = [a + b + c for a, b, c in zip(v12, v23, v31)]
    print("Kirchhoff check:   V12 + V23 + V31 must be 0 for a symmetric machine")
    print("                   max |sum| = %.4f V   (data points with |sum| > 0.05 V: %d of %d)"
          % (max(abs(s) for s in ksum), sum(1 for s in ksum if abs(s) > 0.05), len(ksum)))
    print("                   plateau values: %d of %d samples are exactly +/-2 V -> probably the measurement range"
          % (sum(1 for a in v12 + v23 + v31 if abs(abs(a) - 2.0) < 1e-9), 3 * len(v12)))
    print("                   If the +/-2 V flat tops are clipped, the real peaks are higher and Ke is underestimated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shift", type=float, default=84.0, help="electrical angle shift [deg] (default 84)")
    ap.add_argument("--check", action="store_true", help="consistency checks of the measured data")
    ap.add_argument("--out", default="ke_tables.csv", help="csv file for the generated tables")
    a = ap.parse_args()
    if a.check:
        check_data()
    tabs = phase_ke(a.shift)
    print("\nshift = %g deg   (Ke_new(phi) = Ke_old(phi - shift))\n" % a.shift)
    print(matlab_arrays(tabs))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position_el_deg", "Ke_a", "Ke_b", "Ke_c"])
        for k in range(N):
            w.writerow(["%g" % (BP_START + BP_STEP * k)] + ["%.6f" % t[k] for t in tabs])
    print("\nsaved %s" % a.out)
