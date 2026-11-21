"""
kalman_walkthrough.py - the first 5 steps of the Kalman filter, one small calculation at a time.

Run it in a command prompt:
    python kalman_walkthrough.py            every calculation, press Enter for the next one
    python kalman_walkthrough.py --brief    skip the entry-by-entry matrix products of P
    python kalman_walkthrough.py --auto     no pauses (e.g. to save everything in a file:
                                            python kalman_walkthrough.py --auto > steps.txt)

Needs only Python 3 (no packages). Numbers: the PWM 0.12 run, forward sweep, start angle 15 deg,
variant "interp" with the tuned noise settings. The measured current and Hall angle of the first
5 steps are stored in this file (taken from the simulation of the Simulink model).

Each filter step has two parts:
    PREDICT : x_pred = F*x + Bd*V          and   P_pred = F*P*F^T + Qd
    CORRECT : with the current measurement, then with the Hall angle measurement
                  error = measured - predicted
                  S     = P(j,j) + R
                  K(r)  = P(r,j) / S
                  x(r)  = x_pred(r) + K(r)*error
                  P(r,c)= P_pred(r,c) - K(r)*P_pred(j,c)
"""
import math
import sys

AUTO = "--auto" in sys.argv
BRIEF = "--brief" in sys.argv

# ----------------------------------------------------------------------------------------------
# Constants (from the Simulink model) and settings
# ----------------------------------------------------------------------------------------------
R = 0.16                  # winding resistance [ohm]
L = 0.000066              # winding inductance [H]
KT = 0.01709              # torque constant [N*m/A]
KE = 0.01709              # back-EMF constant [V/(rad/s)]
B = 1e-6                  # friction [N*m/(rad/s)]
J = (0.0011081 + 0.000226854) * (1.0 / (73 * 73)) + 0.00001857   # inertia at the motor shaft [kg*m^2]
GLOAD = 1.0 / (73 * 0.8)  # camshaft torque -> motor torque
VDC = 13.5
PWM = 0.12
V = VDC * PWM             # voltage [V]
DT = 2e-4                 # filter step [s]

# tuned noise settings (variant interp)
Q_W = 16497.19144978792
Q_I = 83.16987286304315
Q_T = 4758.316383815639
R_I = 3.30601629199119e-08
R_TH = 0.03178552435484826

THETA0 = 19.111355309337906     # start angle 15 deg * 73 in rad

# measured current [A] and Hall angle [rad] at steps 1 to 5, and the true values (not seen by the filter)
Z_I = [3.9003522794227146, 6.303280441731733, 7.766214139570873, 8.63941368337653, 9.142957917119944]
HALL = [18.9368305, 18.9368305, 18.9368305, 19.125514824324323, 19.172685905405405]
TRUTH = [  # angle, speed, current, torque at the motor
    (19.11130820257087, -0.3534469938792353, 3.9003522794227146, 0.06924661206208911),
    (19.11125110881863, -0.1450329725997654, 6.303280441731733, 0.06924665656654594),
    (19.11127297613864, 0.40781322425473326, 7.766214139570873, 0.06924663952101712),
    (19.111427989056804, 1.1686367507593418, 8.63941368337653, 0.06924647375654919),
    (19.111748521790965, 2.051863135089692, 9.142957917119944, 0.06924602574144838),
]

NAMES = ["angle", "speed", "current", "torque"]
UNITS = ["rad", "rad/s", "A", "N*m"]


# ----------------------------------------------------------------------------------------------
# Small helpers (plain lists, no numpy)
# ----------------------------------------------------------------------------------------------
def f(v):
    v = v + 0.0                      # turns -0.0 into 0.0
    return ("%.8g" if abs(v) >= 10 else "%.6g") % v


def fp(v):
    s = f(v)
    return "(" + s + ")" if v < 0 else s


def pause():
    if AUTO:
        print()
        return
    try:
        input("      [press Enter for the next calculation] ")
    except EOFError:
        pass
    print()


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(r) for r in zip(*a)]


def scale(a, s):
    return [[v * s for v in r] for r in a]


def add(a, b):
    return [[x + y for x, y in zip(r, q)] for r, q in zip(a, b)]


def expm(m):
    """Matrix exponential: scaling and squaring with a Taylor series (the same method as the main script)."""
    n = len(m)
    norm1 = max(sum(abs(m[i][j]) for i in range(n)) for j in range(n))
    s = max(0, int(math.ceil(math.log2(max(norm1, 1e-12)))) + 1)
    a = scale(m, 1.0 / (2.0 ** s))
    e = eye(n)
    t = eye(n)
    for k in range(1, 20):
        t = scale(matmul(t, a), 1.0 / k)
        e = add(e, t)
    for _ in range(s):
        e = matmul(e, e)
    return e


def show_matrix(name, m):
    print("  %s =" % name)
    for r in m:
        print("      " + "  ".join("%13.6g" % v for v in r))


def show_vector(name, v):
    print("  %s = [%s]" % (name, ",  ".join("%.7g" % q for q in v)))


# ----------------------------------------------------------------------------------------------
# Setup: A, F, Bd, Qd (calculated once)
# ----------------------------------------------------------------------------------------------
A = [[0, 1, 0, 0],
     [0, -B / J, KT / J, -1.0 / J],
     [0, -KE / L, -R / L, 0],
     [0, 0, 0, 0]]
BC = [0, 0, 1.0 / L, 0]

aug = [list(A[r]) + [BC[r]] for r in range(4)] + [[0, 0, 0, 0, 0]]
e5 = expm(scale(aug, DT))
F = [[e5[r][c] for c in range(4)] for r in range(4)]
BD = [e5[r][4] for r in range(4)]

qc = [[0, 0, 0, 0], [0, Q_W, 0, 0], [0, 0, Q_I, 0], [0, 0, 0, Q_T]]
m8 = [[-A[r][c] for c in range(4)] + [qc[r][c] for c in range(4)] for r in range(4)] + \
     [[0, 0, 0, 0] + [A[c][r] for c in range(4)] for r in range(4)]
e8 = expm(scale(m8, DT))
phi12 = [[e8[r][4 + c] for c in range(4)] for r in range(4)]
phi22 = [[e8[4 + r][4 + c] for c in range(4)] for r in range(4)]
QD = matmul(transpose(phi22), phi12)
QD = [[0.5 * (QD[r][c] + QD[c][r]) for c in range(4)] for r in range(4)]


# ----------------------------------------------------------------------------------------------
# The walkthrough
# ----------------------------------------------------------------------------------------------
def intro():
    print("=" * 100)
    print(" KALMAN FILTER WALKTHROUGH - the first 5 steps, one small calculation at a time")
    print("=" * 100)
    print(" Press Enter to see the next calculation.   (python kalman_walkthrough.py --brief  = fewer details)")
    print()
    print(" Notation:  states are numbered   1 = angle [rad]   2 = speed [rad/s]")
    print("                                  3 = current [A]   4 = torque at the motor [N*m]")
    print("            F(r,c) = entry in row r and column c of the matrix F.  P(r,c), M(r,c), K(r) likewise.")
    print("            x = the filter's estimate of the 4 states,  P = its uncertainty (4 x 4 table).")
    print()
    print(" The torque in the estimate is the load torque at the MOTOR side. Camshaft torque = torque / Gload.")
    pause()


def setup():
    print("-" * 100)
    print(" SETUP A - the constants")
    print("-" * 100)
    print("  R = %s ohm      L = %s H      Kt = %s      Ke = %s      B = %s" % (f(R), f(L), f(KT), f(KE), f(B)))
    print("  J = (0.0011081 + 0.000226854) * (1/(73*73)) + 0.00001857 = %s kg*m^2" % f(J))
    print("  Gload = 1/(73*0.8) = %s" % f(GLOAD))
    print("  V = Vdc * PWM = %s * %s = %s volt" % (f(VDC), f(PWM), f(V)))
    print("  dt = %s s (one filter step)" % f(DT))
    pause()
    print("-" * 100)
    print(" SETUP B - the matrix A of the motor model  (dx/dt = A*x + Bc*V)")
    print("-" * 100)
    for (r, c, txt, val) in ((2, 2, "-B/J", -B / J), (2, 3, "Kt/J", KT / J), (2, 4, "-1/J", -1.0 / J),
                             (3, 2, "-Ke/L", -KE / L), (3, 3, "-R/L", -R / L)):
        print("  A(%d,%d) = %s = %s" % (r, c, txt, f(val)))
    print("  A(1,2) = 1 (angle changes with the speed),  Bc(3) = 1/L = %s,  all other entries are 0" % f(1.0 / L))
    show_matrix("A", A)
    pause()
    print("-" * 100)
    print(" SETUP C - F, Bd and Qd, calculated once from A with the matrix exponential (Steps 4 and 5 of the document)")
    print("-" * 100)
    print("  F = exp(A*dt)    Bd = (integral of exp(A*s) ds) * Bc    Qd = process noise per step (from Q_w, Q_i, Q_T)")
    print("  Noise settings: q_w = %s   q_i = %s   q_T = %s   r_i = %s   r_th = %s" % (f(Q_W), f(Q_I), f(Q_T), f(R_I), f(R_TH)))
    show_matrix("F", F)
    show_vector("Bd", BD)
    show_matrix("Qd", QD)
    pause()
    print("-" * 100)
    print(" SETUP D - the start of the filter")
    print("-" * 100)
    print("  x(1) = start angle = 15 deg * 73 = %s rad     x(2) = speed = 0     x(3) = current = 0     x(4) = torque = 0 (unknown)" % f(THETA0))
    print("  P0 = diagonal(1e-6, 1e-4, 1e-4, 0.0025):  angle known, speed and current known (rest), torque unknown to about 0.05 N*m")
    print("  Measurements the sensors will give in steps 1 to 5:")
    for k in range(5):
        print("     step %d (t = %.1f ms):  current %.6f A   Hall angle %.6f rad" % (k + 1, 0.2 * (k + 1), Z_I[k], HALL[k]))
    pause()


def predict_state(x, k):
    print("  1) PREDICT THE STATE:   x_pred(r) = F(r,1)*x(1) + F(r,2)*x(2) + F(r,3)*x(3) + F(r,4)*x(4) + Bd(r)*V")
    print()
    xp = []
    for r in range(4):
        terms = [F[r][c] * x[c] for c in range(4)]
        val = sum(terms) + BD[r] * V
        print("   x_pred(%d)  (%s)" % (r + 1, NAMES[r]))
        print("      = F(%d,1)*x(1) + F(%d,2)*x(2) + F(%d,3)*x(3) + F(%d,4)*x(4) + Bd(%d)*V" % ((r + 1,) * 5))
        print("      = %s*%s + %s*%s + %s*%s + %s*%s + %s*%s" % (
            fp(F[r][0]), fp(x[0]), fp(F[r][1]), fp(x[1]), fp(F[r][2]), fp(x[2]), fp(F[r][3]), fp(x[3]), fp(BD[r]), fp(V)))
        print("      = %s + %s + %s + %s + %s" % (f(terms[0]), f(terms[1]), f(terms[2]), f(terms[3]), f(BD[r] * V)))
        print("      = %s %s" % (f(val), UNITS[r]))
        xp.append(val)
        pause()
    return xp


def predict_P(P):
    print("  2) PREDICT THE UNCERTAINTY:   P_pred = F * P * F^T + Qd      (done in two parts: M = F*P, then M*F^T + Qd)")
    print()
    M = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            terms = [F[r][j] * P[j][c] for j in range(4)]
            M[r][c] = sum(terms)
            if not BRIEF:
                print("   M(%d,%d) = F(%d,1)*P(1,%d) + F(%d,2)*P(2,%d) + F(%d,3)*P(3,%d) + F(%d,4)*P(4,%d)" % (
                    (r + 1, c + 1) + (r + 1, c + 1) * 4))
                print("         = %s*%s + %s*%s + %s*%s + %s*%s" % (
                    fp(F[r][0]), fp(P[0][c]), fp(F[r][1]), fp(P[1][c]), fp(F[r][2]), fp(P[2][c]), fp(F[r][3]), fp(P[3][c])))
                print("         = %s" % f(M[r][c]))
        if not BRIEF:
            pause()
    if BRIEF:
        show_matrix("M = F*P", M)
        pause()
    Pp = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            terms = [M[r][j] * F[c][j] for j in range(4)]
            Pp[r][c] = sum(terms) + QD[r][c]
            if not BRIEF:
                print("   P_pred(%d,%d) = M(%d,1)*F(%d,1) + M(%d,2)*F(%d,2) + M(%d,3)*F(%d,3) + M(%d,4)*F(%d,4) + Qd(%d,%d)" % (
                    (r + 1, c + 1) + (r + 1, c + 1) * 4 + (r + 1, c + 1)))
                print("            = %s*%s + %s*%s + %s*%s + %s*%s + %s" % (
                    fp(M[r][0]), fp(F[c][0]), fp(M[r][1]), fp(F[c][1]), fp(M[r][2]), fp(F[c][2]), fp(M[r][3]), fp(F[c][3]), fp(QD[r][c])))
                print("            = %s + %s + %s + %s + %s = %s" % (f(terms[0]), f(terms[1]), f(terms[2]), f(terms[3]), f(QD[r][c]), f(Pp[r][c])))
        if not BRIEF:
            pause()
    show_matrix("P_pred", Pp)
    print("  (P_pred is symmetric: P_pred(r,c) = P_pred(c,r). The diagonal is the uncertainty of each state, squared.)")
    pause()
    return Pp


def correct(x, P, j, z, Rn, label, what):
    """One scalar measurement of state j (0-based)."""
    jj = j + 1
    print("  %s" % label)
    print()
    err = z - x[j]
    print("   a) the error:   error = measured - predicted")
    print("         measured %s = %s %s" % (what, f(z), UNITS[j]))
    print("         predicted %s = x_pred(%d) = %s %s" % (what, jj, f(x[j]), UNITS[j]))
    print("         error = %s - %s = %s %s" % (f(z), fp(x[j]), f(err), UNITS[j]))
    pause()
    S = P[j][j] + Rn
    print("   b) S = P(%d,%d) + R = %s + %s = %s" % (jj, jj, f(P[j][j]), f(Rn), f(S)))
    print("         (P(%d,%d) = uncertainty of the predicted %s, R = noise of this measurement)" % (jj, jj, what))
    pause()
    K = []
    for r in range(4):
        kv = P[r][j] / S
        K.append(kv)
        print("   c) K(%d) = P(%d,%d) / S = %s / %s = %s      (gain for the %s)" % (r + 1, r + 1, jj, fp(P[r][j]), f(S), f(kv), NAMES[r]))
        pause()
    xn = []
    for r in range(4):
        v = x[r] + K[r] * err
        xn.append(v)
        print("   d) x_new(%d) = x_pred(%d) + K(%d)*error = %s + %s*%s = %s + %s = %s %s   (%s)" % (
            r + 1, r + 1, r + 1, fp(x[r]), fp(K[r]), fp(err), f(x[r]), f(K[r] * err), f(v), UNITS[r], NAMES[r]))
        pause()
    Pn = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            Pn[r][c] = P[r][c] - K[r] * P[j][c]
            if not BRIEF:
                print("   e) P_new(%d,%d) = P(%d,%d) - K(%d)*P(%d,%d) = %s - %s*%s = %s" % (
                    r + 1, c + 1, r + 1, c + 1, r + 1, jj, c + 1, fp(P[r][c]), fp(K[r]), fp(P[j][c]), f(Pn[r][c])))
        if not BRIEF:
            pause()
    if BRIEF:
        show_matrix("P_new", Pn)
        pause()
    print("   The uncertainty of the %s went from %s to %s (squared)." % (what, f(P[j][j]), f(Pn[j][j])))
    pause()
    return xn, Pn


def run_steps():
    x = [THETA0, 0.0, 0.0, 0.0]
    P = [[1e-6, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 1e-4, 0], [0, 0, 0, 0.0025]]
    summary = []
    for k in range(5):
        print("#" * 100)
        print(" STEP %d   (t = %.1f ms)" % (k + 1, 0.2 * (k + 1)))
        print("#" * 100)
        print("  start of the step: " + "   ".join("%s %s" % (NAMES[i], f(x[i])) for i in range(4)))
        pause()
        xp = predict_state(x, k)
        Pp = predict_P(P)
        x, P = correct(xp, Pp, 2, Z_I[k], R_I,
                       "3) CORRECT WITH THE CURRENT MEASUREMENT  (state 3, noise R = r_i)", "current")
        x, P = correct(x, P, 0, HALL[k], R_TH,
                       "4) CORRECT WITH THE HALL ANGLE  (state 1, noise R = r_th)", "angle")
        P = [[0.5 * (P[r][c] + P[c][r]) for c in range(4)] for r in range(4)]
        print("  5) P = (P + P^T)/2   (removes tiny rounding differences; the diagonal is unchanged)")
        pause()
        t = TRUTH[k]
        print("  RESULT OF STEP %d" % (k + 1))
        print("      %-10s %16s %16s" % ("state", "filter estimate", "true (not seen)"))
        for i in range(4):
            print("      %-10s %16.9g %16.9g   %s" % (NAMES[i], x[i], t[i], UNITS[i]))
        est_cam = x[3] / GLOAD
        true_cam = t[3] / GLOAD
        print("      camshaft torque = torque / Gload = %s / %s = %s N*m   (true %s N*m)" % (f(x[3]), f(GLOAD), f(est_cam), f(true_cam)))
        summary.append((k + 1, x[3], t[3], est_cam, true_cam))
        pause()
    print("=" * 100)
    print(" SUMMARY OF THE 5 STEPS - torque estimate")
    print("=" * 100)
    print("   step    t [ms]    torque est.   torque true    camshaft est.   camshaft true   [N*m]")
    for s in summary:
        print("   %4d    %6.1f    %11.5f   %11.5f    %13.4f   %13.4f" % (s[0], 0.2 * s[0], s[1], s[2], s[3], s[4]))
    print()
    print(" The estimate starts at 0 and reaches the true value within a few steps: each step the current")
    print(" measurement and the Hall angle correct the prediction, and the corrected result starts the next step.")


if __name__ == "__main__":
    intro()
    setup()
    run_steps()
