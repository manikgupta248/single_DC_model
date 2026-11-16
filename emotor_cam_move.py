# =============================================================================
# emotor_cam_move.slx  ->  Python port, block by block (R2025b model, 6th version)
#
# Changes compared to the 3rd upload:
#   1) Camshaft_quantanizer moved IN FRONT of Gcam1: it now quantizes the motor angle
#      (output of 'E-motor velocity and angle') in steps of 0.174533 rad = 10 deg motor angle
#      (= Hall resolution: 6 pole pairs x 6 edges = 36 edges per motor revolution),
#      and Gcam1 (2/73) is applied afterwards.
#   2) PID Controller3: P = 0.0977362, I = 0.710737, D = 0.00368854, N = 400
#   3) NEW OPTION HALL_MODE (below): "INTERPOLATION" or "NOINTERPOLATION".
#      "NOINTERPOLATION" = exactly the Simulink model (quantized motor angle goes into Gcam1).
#      "INTERPOLATION"   = a block between Camshaft_quantanizer and Gcam1 estimates the angle
#                          between two Hall edges using the speed measured from the edge timing.
#   4) NEW OPTIONS DIRECTION, START_ANGLE_DEG, WRAP_LOAD_TABLE (below) to run the cam in reverse.
#      Defaults (FORWARD, 0, False) give exactly the same results as the 5th version.
#
# Block names, parameter values and signal connections are copied from the model.
# Section order follows the signal flow of the model.
#
# ONLY intentional difference to Simulink:
#   Simulink : variable-step solver "VariableStepAuto", RelTol 1e-3, AbsTol auto,
#              MaxStep 1e-4, StopTime 8
#   Python   : fixed step h = 1e-5 s (= sample time of "PID Controller3", the
#              smallest sample time in the model), classic RK4 for the continuous
#              states (Li, Wi, Aint, Wi1).  The controller output is constant
#              during each 1e-5 s step, exactly like the discrete block in Simulink.
# Numerical results can therefore differ slightly from Simulink's variable step.
# =============================================================================
import math

# ----------------------------------------------------------------------------
# Simulation settings (configSet0.xml)
# ----------------------------------------------------------------------------
StartTime = 0.0
StopTime = 8.0
h = 1e-5                 # fixed integration step = sample time of PID Controller3
LOG_DECIMATION = 10      # store every 10th step (= every 1e-4 s, like MaxStep)

# ----------------------------------------------------------------------------
# OPTION: angle that the controller gets (output of Camshaft_quantanizer -> Gcam1)
#   "NOINTERPOLATION" : quantized motor angle goes directly into Gcam1 (as in the Simulink model)
#   "INTERPOLATION"   : angle is interpolated between the Hall edges with the speed measured
#                       from the time between the last two edges
# You can also give it on the command line:  python emotor_cam_move.py INTERPOLATION
# ----------------------------------------------------------------------------
HALL_MODE = "NOINTERPOLATION"     # "INTERPOLATION" or "NOINTERPOLATION"

# ----------------------------------------------------------------------------
# OPTIONS: direction of rotation
#   DIRECTION       : "FORWARD" (ramp +75 deg/s) or "REVERSE" (ramp -75 deg/s)
#   START_ANGLE_DEG : cam angle at t = 0 [deg]; used as Ramp initial output and as initial condition
#                     of the integrators Aint and Wi1 (for REVERSE use e.g. 400 so that the cam
#                     runs backwards through the cam profile)
#   WRAP_LOAD_TABLE : True  -> cam load repeats every 360 deg (needed when the cam passes 0/360 deg)
#                     False -> lookup table clips outside 0...359.5 deg (as in the Simulink model)
# ----------------------------------------------------------------------------
DIRECTION = "FORWARD"             # "FORWARD" or "REVERSE"
START_ANGLE_DEG = 0.0
WRAP_LOAD_TABLE = False

# ----------------------------------------------------------------------------
# OPTION: what drives the motor
#   CONTROL_MODE = "CONTROLLER" : PWM comes from "Simplified, cascaded Controller1" (closed loop)
#   CONTROL_MODE = "CONSTANT"   : PWM = CONSTANT_PWM for the whole run (open loop, controller ignored)
#   CONSTANT_PWM : duty cycle as a fraction, -0.95 ... 0.95 (e.g. 0.5 = 50 % -> Vapp = 0.5 * Vdc)
#                  negative values turn the motor backwards
# ----------------------------------------------------------------------------
CONTROL_MODE = "CONTROLLER"       # "CONTROLLER" or "CONSTANT"
CONSTANT_PWM = 0.02

# ----------------------------------------------------------------------------
# Top level model: Constant blocks
# ----------------------------------------------------------------------------
Bc = 1e-06               # Constant  Bc
Kec = 0.01709              # Constant  Kec
Ktc = 0.01709              # Constant  Ktc
Lc = 0.000066            # Constant  Lc
Rc = 0.1600              # Constant  Rc
Vdc = 13.5               # Constant  Vdc
ThetaTC1 = 0.0           # Constant  ThetaTC1  (feeds controller input 1 "phi_dot_Crank")

# ----------------------------------------------------------------------------
# Top level model: Gain blocks
# ----------------------------------------------------------------------------
Gcam = 1 / 73                                   # Gain  Gcam
Gcam1 = 2 / 73                                  # Gain  Gcam1
Gdeg = 180 / math.pi                            # Gain  Gdeg
Gdeg1 = 2 * math.pi / 180                       # Gain  Gdeg1
Gload = 1 / (73 * 0.8)                          # Gain  Gload
G_convert_camshaft_velocity = 2                 # Gain  "Convert camshaft velocity to crank shaft units [crank rad/sec]" (output unconnected)
G_convert_cam_angle_to_crank = 2                # Gain  "convert cam angle to crank angle in radian" (output unconnected)

# ----------------------------------------------------------------------------
# Subsystem "Inertia calculation"
# ----------------------------------------------------------------------------
Cam_shaft_intertial = 0.000226854                 # Constant
E_motor_intertia = 0.00001857                    # Constant
Gear_box_interia = 0.0011081                  # Constant "Gear_box_interia =sum of Endstop + WG+ Sprocket"
Gear_ratio = 1 / (73 * 73)                      # Gain "Gear ratio"
Sum_inertia = Gear_box_interia + Cam_shaft_intertial     # Sum '|++'
Sum1_inertia = Sum_inertia * Gear_ratio + E_motor_intertia  # Sum1 '|++'
Combined_Inertia = Sum1_inertia                 # Outport "Combined Inertia" -> Jdiv in2
J = Combined_Inertia

# ----------------------------------------------------------------------------
# Subsystem "E-motor velocity and angle"
# ----------------------------------------------------------------------------
Grpm = 60 / (2 * math.pi)                       # Gain Grpm   (-> scope "Velocity eMOTOR RPM")
Grpm1 = 180 / (math.pi)                         # Gain Grpm1  (-> scope "Angle E motor degree")
# Integrator Wi1: input In1 = Wi, output "E-motor angle radian"

# ----------------------------------------------------------------------------
# Block "Target Cam angle"  (Simulink Ramp)
# ----------------------------------------------------------------------------
Ramp_slope = 150/ 2 if DIRECTION.strip().upper() == "FORWARD" else -150 / 2   # Ramp slope [deg/s]
Ramp_start = 0
Ramp_InitialOutput = START_ANGLE_DEG


def Target_Cam_angle(t):
    if t < Ramp_start:
        return Ramp_InitialOutput
    return Ramp_InitialOutput + Ramp_slope * (t - Ramp_start)


# ----------------------------------------------------------------------------
# Block "Camshaft_quantanizer"  (Quantizer)  - now in front of Gcam1, i.e. on the MOTOR angle [rad]
# 0.174533 rad = 10 deg motor angle = one Hall edge (6 pole pairs x 6 edges per electrical revolution)
# ----------------------------------------------------------------------------
QuantizationInterval = 0.174533


def Camshaft_quantanizer(u):
    # Quantizer: y = q * round(u/q)   (round half away from zero)
    x = u / QuantizationInterval
    r = math.floor(abs(x) + 0.5)
    if x < 0:
        r = -r
    return QuantizationInterval * r


class HallInterpolator:
    """Sits between Camshaft_quantanizer and Gcam1 (only used when HALL_MODE = "INTERPOLATION").

    Q  : quantized motor angle [rad] (output of Camshaft_quantanizer)
    t  : time [s]
    At every Hall edge (Q changes) the angle of the edge is known exactly (Q -/+ half a step)
    and the speed is the angle between the last two edges divided by the time between them.
    Between two edges the angle is   edge angle + speed * (time since the edge),
    limited to the interval Q -/+ half a step.
    """

    def __init__(self):
        mode = HALL_MODE.strip().upper()
        if mode not in ("INTERPOLATION", "NOINTERPOLATION"):
            raise ValueError('HALL_MODE must be "INTERPOLATION" or "NOINTERPOLATION", got %r' % (HALL_MODE,))
        self.interpolate = (mode == "INTERPOLATION")
        self.Qprev = 0.0
        self.th_edge = 0.0
        self.t_edge = 0.0
        self.w_hat = 0.0
        self.have_edge = False

    def output(self, Q, t):
        if not self.interpolate:
            return Q                                   # NOINTERPOLATION: pass the quantized angle through
        q = QuantizationInterval
        if Q != self.Qprev:                            # a Hall edge happened
            s = 1.0 if Q > self.Qprev else -1.0
            th_new = Q - s * q / 2                     # angle at which the edge occurs
            if self.have_edge and t > self.t_edge:
                self.w_hat = (th_new - self.th_edge) / (t - self.t_edge)
            self.th_edge = th_new
            self.t_edge = t
            self.Qprev = Q
            self.have_edge = True
        est = self.th_edge + self.w_hat * (t - self.t_edge)
        return min(max(est, Q - q / 2), Q + q / 2)


# ----------------------------------------------------------------------------
# Block "Tab"  (Lookup_n-D, 1-D, linear interpolation, extrapolation = Clip)
# BreakpointsForDimension1 = 0:0.5:359.5
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Block "Tab"  (Lookup_n-D, 1-D, linear interpolation, extrapolation = Clip)
# NEW cam load torque = T_total (sum of T_cyl1 ... T_cyl6) [N*m at the camshaft]
# BreakpointsForDimension1 = 0:1:359   (1 deg steps, 360 points)
# ----------------------------------------------------------------------------
Tab_table = [
    # 0 ... 59 deg
    0.212, 0.74, 1.252, 1.726, 2.156, 2.536, 2.874, 3.17, 3.438, 3.67,
    3.862, 4.0, 4.082, 4.114, 4.102, 4.044, 3.94, 3.794, 3.612, 3.392,
    3.138, 2.848, 2.548, 2.23, 1.91, 1.586, 1.284, 1.0, 0.712, 0.426,
    0.12, -0.208, -0.56, -0.932, -1.306, -1.684, -2.066, -2.442, -2.812, -3.158,
    -3.484, -3.768, -3.974, -4.084, -4.122, -4.13, -4.134, -4.14, -4.132, -4.062,
    -3.904, -3.668, -3.388, -3.052, -2.68, -2.276, -1.834, -1.356, -0.848, -0.322,
    # 60 ... 119 deg
    0.212, 0.74, 1.252, 1.726, 2.156, 2.536, 2.874, 3.17, 3.438, 3.67,
    3.862, 4.0, 4.082, 4.114, 4.102, 4.044, 3.94, 3.794, 3.612, 3.392,
    3.138, 2.848, 2.548, 2.23, 1.91, 1.586, 1.284, 1.0, 0.712, 0.426,
    0.12, -0.208, -0.56, -0.932, -1.306, -1.684, -2.066, -2.442, -2.812, -3.158,
    -3.484, -3.768, -3.974, -4.084, -4.122, -4.13, -4.134, -4.14, -4.132, -4.062,
    -3.904, -3.668, -3.388, -3.052, -2.68, -2.276, -1.834, -1.356, -0.848, -0.322,
    # 120 ... 179 deg
    0.212, 0.74, 1.252, 1.726, 2.156, 2.536, 2.874, 3.17, 3.438, 3.67,
    3.862, 4.0, 4.082, 4.114, 4.102, 4.044, 3.94, 3.794, 3.612, 3.392,
    3.138, 2.848, 2.548, 2.23, 1.91, 1.586, 1.284, 1.0, 0.712, 0.426,
    0.12, -0.208, -0.56, -0.932, -1.306, -1.684, -2.066, -2.442, -2.812, -3.158,
    -3.484, -3.768, -3.974, -4.084, -4.122, -4.13, -4.134, -4.14, -4.132, -4.062,
    -3.904, -3.668, -3.388, -3.052, -2.68, -2.276, -1.834, -1.356, -0.848, -0.322,
    # 180 ... 239 deg
    0.212, 0.74, 1.252, 1.726, 2.156, 2.536, 2.874, 3.17, 3.438, 3.67,
    3.862, 4.0, 4.082, 4.114, 4.102, 4.044, 3.94, 3.794, 3.612, 3.392,
    3.138, 2.848, 2.548, 2.23, 1.91, 1.586, 1.284, 1.0, 0.712, 0.426,
    0.12, -0.208, -0.56, -0.932, -1.306, -1.684, -2.066, -2.442, -2.812, -3.158,
    -3.484, -3.768, -3.974, -4.084, -4.122, -4.13, -4.134, -4.14, -4.132, -4.062,
    -3.904, -3.668, -3.388, -3.052, -2.68, -2.276, -1.834, -1.356, -0.848, -0.322,
    # 240 ... 299 deg
    0.212, 0.74, 1.252, 1.726, 2.156, 2.536, 2.874, 3.17, 3.438, 3.67,
    3.862, 4.0, 4.082, 4.114, 4.102, 4.044, 3.94, 3.794, 3.612, 3.392,
    3.138, 2.848, 2.548, 2.23, 1.91, 1.586, 1.284, 1.0, 0.712, 0.426,
    0.12, -0.208, -0.56, -0.932, -1.306, -1.684, -2.066, -2.442, -2.812, -3.158,
    -3.484, -3.768, -3.974, -4.084, -4.122, -4.13, -4.134, -4.14, -4.132, -4.062,
    -3.904, -3.668, -3.388, -3.052, -2.68, -2.276, -1.834, -1.356, -0.848, -0.322,
    # 300 ... 359 deg
    0.212, 0.74, 1.252, 1.726, 2.156, 2.536, 2.874, 3.17, 3.438, 3.67,
    3.862, 4.0, 4.082, 4.114, 4.102, 4.044, 3.94, 3.794, 3.612, 3.392,
    3.138, 2.848, 2.548, 2.23, 1.91, 1.586, 1.284, 1.0, 0.712, 0.426,
    0.12, -0.208, -0.56, -0.932, -1.306, -1.684, -2.066, -2.442, -2.812, -3.158,
    -3.484, -3.768, -3.974, -4.084, -4.122, -4.13, -4.134, -4.14, -4.132, -4.062,
    -3.904, -3.668, -3.388, -3.052, -2.68, -2.276, -1.834, -1.356, -0.848, -0.322,
]
Tab_bp_start = 0.0
Tab_bp_step = 1.0                   # NEW: 1 deg breakpoints (was 0.5 deg)
Tab_n = len(Tab_table)              # 360 points -> last breakpoint 359 deg


def Tab(u):
    if WRAP_LOAD_TABLE:                              # periodic: cam load repeats every 360 deg
        pos = (u % 360.0) / Tab_bp_step
        k = int(pos)
        frac = pos - k
        return Tab_table[k % Tab_n] + frac * (Tab_table[(k + 1) % Tab_n] - Tab_table[k % Tab_n])
    pos = (u - Tab_bp_start) / Tab_bp_step
    if pos <= 0.0:
        return Tab_table[0]             # Clip
    if pos >= Tab_n - 1:
        return Tab_table[Tab_n - 1]     # Clip
    k = int(pos)
    frac = pos - k
    return Tab_table[k] + frac * (Tab_table[k + 1] - Tab_table[k])


# ----------------------------------------------------------------------------
# Subsystem "Simplified, cascaded Controller1"  (contents of system_98)
#   Inports : 1 phi_dot_Crank [rad/sec], 2 phi_phase_target [rad Crank], 3 phi_phase_act [rad Crank]
#   Outport : PWM [-]
# ----------------------------------------------------------------------------
RT_Ts = 0.00001                     # Rate Transition1/2/4  OutPortSampleTime (1e-5 in the Simulink model)
PID_Ts = 0.00001                    # PID Controller3 SampleTime
Const_73 = 73                       # Constant (SampleTime 1e-5)
Gain_ff = 0.5                       # Gain "Gain"
G_rad_s_to_rpm = 60 / 2 / math.pi   # Gain "rad/s -> rpm"
G_rad_to_deg = 360 / 2 / math.pi    # Gain "rad -> deg"
G_percent = 1 / 100                 # Gain "% -> 0.01"
Sat_upper = 95                      # Saturation
Sat_lower = -95

# PID Controller3: Parallel form, Discrete-time, Forward Euler (integrator and filter)
PID_P = 0.0977362
PID_I = 0.710737
PID_D = 0.00368854
PID_N = 400
PID_IC_integrator = 0
PID_IC_filter = 0


class Controller:
    def __init__(self):
        # PID states (Forward Euler)
        self.xi = PID_IC_integrator     # integrator state
        self.xf = PID_IC_filter         # derivative filter state
        # Rate Transition output registers (held between RT_Ts hits)
        self.RT1 = 0.0                  # Rate Transition1 : phi_phase_target
        self.RT2 = 0.0                  # Rate Transition2 : phi_phase_act
        self.RT4 = 0.0                  # Rate Transition4 : phi_dot_Crank
        # last block outputs (for logging)
        self.Sum = 0.0
        self.rad_deg = 0.0
        self.PID = 0.0
        self.Divide = 0.0
        self.Gain = 0.0
        self.Sum1 = 0.0
        self.Saturation = 0.0
        self.PWM = 0.0

    def sample_rate_transitions(self, phi_dot_Crank, phi_phase_target, phi_phase_act):
        # Rate Transition4 / 1 / 2  (continuous -> RT_Ts zero-order hold)
        self.RT4 = phi_dot_Crank
        self.RT1 = phi_phase_target
        self.RT2 = phi_phase_act

    def output_and_update(self):
        # Sum '+-'  : RT1 - RT2
        self.Sum = self.RT1 - self.RT2
        # Gain "rad -> deg"
        self.rad_deg = self.Sum * G_rad_to_deg
        u = self.rad_deg
        # PID Controller3 (output uses states before update)
        Yd = PID_N * (PID_D * u - self.xf)                  # filtered derivative
        self.PID = PID_P * u + self.xi + Yd
        # Product "Divide" (Inputs '**'): PID * Constant(73)
        self.Divide = self.PID * Const_73
        # Rate Transition4 -> Gain "rad/s -> rpm" -> Gain "Gain"
        self.Gain = self.RT4 * G_rad_s_to_rpm * Gain_ff
        # Sum1 '++'
        self.Sum1 = self.Gain + self.Divide
        # Saturation
        self.Saturation = min(max(self.Sum1, Sat_lower), Sat_upper)
        # Gain "% -> 0.01"
        self.PWM = self.Saturation * G_percent
        # PID state update (Forward Euler)
        self.xi = self.xi + PID_Ts * PID_I * u
        self.xf = self.xf + PID_Ts * Yd
        return self.PWM


# ----------------------------------------------------------------------------
# Continuous part: motor / gearbox / cam
#   states: Li (Integrator), Wi (Integrator), Aint (Integrator), Wi1 (Integrator in
#           subsystem "E-motor velocity and angle"), all initial condition 0
# ----------------------------------------------------------------------------
def plant_derivatives(Li, Wi, Aint, Wi1, Vapp):
    Ir = Li * Rc                                # Product Ir
    Eb = Wi * Kec                               # Product Eb
    Vsum = Vapp - Ir - Eb                       # Sum Vsum '+--'
    Ldiv = Vsum / Lc                            # Product Ldiv '*/'
    Tm = Ktc * Li                               # Product Tm
    Tb = Bc * Wi                                # Product Tb
    Gcam_out = Wi * Gcam                        # Gain Gcam
    Gdeg_out = Aint * Gdeg                      # Gain Gdeg
    Tab_out = Tab(Gdeg_out)                     # Lookup Tab
    Gload_out = Tab_out * Gload                 # Gain Gload
    Tsum = Tm - Tb - Gload_out                  # Sum Tsum '+--'
    Jdiv = Tsum / J                             # Product Jdiv '*/'
    # d(Li)/dt, d(Wi)/dt, d(Aint)/dt, d(Wi1)/dt
    return Ldiv, Jdiv, Gcam_out, Wi


def rk4_step(Li, Wi, Aint, Wi1, Vapp, dt):
    k1 = plant_derivatives(Li, Wi, Aint, Wi1, Vapp)
    k2 = plant_derivatives(Li + 0.5 * dt * k1[0], Wi + 0.5 * dt * k1[1],
                           Aint + 0.5 * dt * k1[2], Wi1 + 0.5 * dt * k1[3], Vapp)
    k3 = plant_derivatives(Li + 0.5 * dt * k2[0], Wi + 0.5 * dt * k2[1],
                           Aint + 0.5 * dt * k2[2], Wi1 + 0.5 * dt * k2[3], Vapp)
    k4 = plant_derivatives(Li + dt * k3[0], Wi + dt * k3[1],
                           Aint + dt * k3[2], Wi1 + dt * k3[3], Vapp)
    return (Li + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            Wi + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            Aint + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            Wi1 + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]))


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate():
    n_steps = int(round((StopTime - StartTime) / h))
    rt_every = int(round(RT_Ts / h))            # 1 step (RT_Ts = h)
    ctrl = Controller()
    hall = HallInterpolator()                   # used between Camshaft_quantanizer and Gcam1
    Li = Wi = 0.0                               # integrators Li, Wi: initial condition 0
    Aint = math.radians(START_ANGLE_DEG)        # integrator Aint: initial condition = start angle [cam rad]
    Wi1 = Aint * 73                             # integrator Wi1 : initial condition = start angle [motor rad]

    names = ["t", "Li", "Wi", "Aint", "Wi1", "Gdeg", "Tab", "Gload", "Tm", "Tb", "Tsum", "Jdiv",
             "Gcam", "Gcam1", "Camshaft_quantanizer", "Hall_out", "Target_Cam_angle", "Gdeg1",
             "RT1", "RT2", "RT4", "Sum", "rad_deg", "PID", "Divide", "Gain", "Sum1",
             "Saturation", "PWM", "Vapp"]
    log = {k: [] for k in names}

    for k in range(n_steps + 1):
        t = StartTime + k * h

        # --- signals that depend only on the continuous states at time t ---
        Gdeg_out = Aint * Gdeg                                  # Gdeg
        Tab_out = Tab(Gdeg_out)                                 # Tab
        Gload_out = Tab_out * Gload                             # Gload
        Gcam_out = Wi * Gcam                                    # Gcam
        Quant_out = Camshaft_quantanizer(Wi1)                   # Camshaft_quantanizer (input: "E-motor angle radian" = Wi1)
        Hall_out = hall.output(Quant_out, t)                    # HALL_MODE: interpolation between Hall edges or pass-through
        Gcam1_out = Hall_out * Gcam1                            # Gcam1  (now behind the quantizer / interpolation)
        Target_out = Target_Cam_angle(t)                        # Target Cam angle
        Gdeg1_out = Target_out * Gdeg1                          # Gdeg1

        # --- controller: Rate Transitions sample every RT_Ts ---
        if k % rt_every == 0:
            ctrl.sample_rate_transitions(ThetaTC1, Gdeg1_out, Gcam1_out)
        PWM = ctrl.output_and_update()                          # "Simplified, cascaded Controller1"
        if CONTROL_MODE.strip().upper() == "CONSTANT":
            PWM = CONSTANT_PWM                                  # open loop: constant duty instead of controller output
        elif CONTROL_MODE.strip().upper() != "CONTROLLER":
            raise ValueError('CONTROL_MODE must be "CONTROLLER" or "CONSTANT", got %r' % (CONTROL_MODE,))

        Vapp = Vdc * PWM                                        # Product Vapp

        Tm = Ktc * Li
        Tb = Bc * Wi
        Tsum = Tm - Tb - Gload_out
        Jdiv_out = Tsum / J

        if k % LOG_DECIMATION == 0:
            vals = dict(t=t, Li=Li, Wi=Wi, Aint=Aint, Wi1=Wi1, Gdeg=Gdeg_out, Tab=Tab_out,
                        Gload=Gload_out, Tm=Tm, Tb=Tb, Tsum=Tsum, Jdiv=Jdiv_out, Gcam=Gcam_out,
                        Gcam1=Gcam1_out, Camshaft_quantanizer=Quant_out, Hall_out=Hall_out,
                        Target_Cam_angle=Target_out, Gdeg1=Gdeg1_out, RT1=ctrl.RT1, RT2=ctrl.RT2,
                        RT4=ctrl.RT4, Sum=ctrl.Sum, rad_deg=ctrl.rad_deg, PID=ctrl.PID,
                        Divide=ctrl.Divide, Gain=ctrl.Gain, Sum1=ctrl.Sum1,
                        Saturation=ctrl.Saturation, PWM=PWM, Vapp=Vapp)
            for key in names:
                log[key].append(vals[key])

        if k < n_steps:
            Li, Wi, Aint, Wi1 = rk4_step(Li, Wi, Aint, Wi1, Vapp, h)

    return log


# ----------------------------------------------------------------------------
# Scopes of the model -> which logged signals they show
# ----------------------------------------------------------------------------
SCOPES = {
    "Acceleration Emotor": ["Jdiv"],
    "Angle Comparision target & Reached": ["Gdeg", "Target_Cam_angle"],
    "Applied Load": ["Tab"],
    "Cam Angle Degree": ["Gdeg"],
    "Cam Angle Degree1": ["Gcam1", "Camshaft_quantanizer"],      # Camshaft_quantanizer = quantized motor angle [rad], Gcam1 = crank angle [rad]
    "Hall angle: quantized vs used": ["Camshaft_quantanizer", "Hall_out"],   # motor angle [rad]
    "Current vs load torque": ["Li", "Tab"],
    "CurrentScope": ["Li"],
    "E-motor Torque": ["Tm"],
    "PWM Scope": ["PWM"],
    "Torque vs camangle calculation": ["Tab", "Gdeg"],
    "Velocity Cam": ["Gcam"],
    "Velocity Emotor": ["Wi"],
    "Controller/OutputPWM": ["PWM"],
    "Controller/ZOH Output Phase_act": ["RT2"],
    "Controller/ZOH Output Target_Act": ["RT1"],
    "E-motor velocity and angle/Angle E motor (rad)": ["Wi1"],
}
def plot_current_vs_load(log, filename="current_vs_load_torque.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13})

    # keep only the first 360 deg of cam travel from the start angle
    ang = np.asarray(log["Gdeg"])
    cur = np.asarray(log["Li"])
    load = np.asarray(log["Tab"])
    travel = np.abs(ang - ang[0])
    end = np.nonzero(travel >= 360.0)[0]
    n = end[0] + 1 if end.size else len(ang)
    ang, cur, load = ang[:n], cur[:n], load[:n]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    l1, = ax1.plot(ang, cur, color="tab:blue", linewidth=1.2, label="Motor currents")
    ax1.set_xlabel("Cam angle [deg]")
    ax1.set_ylabel("Motor current [A]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.4)

    a0 = ang[0]
    a1 = a0 + 360.0 if ang[-1] >= a0 else a0 - 360.0
    ax1.set_xlim(min(a0, a1), max(a0, a1))
    ax1.set_xticks(np.arange(min(a0, a1), max(a0, a1) + 1e-9, 45))
    ax1.xaxis.set_major_formatter(lambda x, pos: "%g°" % x)

    ax2 = ax1.twinx()
    l2, = ax2.plot(ang, load, color="tab:orange", linewidth=2.0, label="Cam torque (Simulation input)")
    ax2.set_ylabel("Load torque", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # scale right axis so both curves share the same zero and equal heights mean
    # "current needed to carry this load" (I = Tab * Gload / Ktc)
    lo, hi = ax1.get_ylim()
    ax2.set_ylim(lo * Ktc / Gload, hi * Ktc / Gload)

    ax1.set_title("Motor current vs cam load torque over one cam revolution", pad=10)
    ax1.legend(handles=[l1, l2], loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print("saved %s" % filename)
    return fig
def compare_speeds(speeds=(75, 150, 250, 350), filename="current_vs_load_speeds.png"):
    """Runs the model once per cam speed [cam deg/s] for one full cam revolution and plots
    motor current (left axis) and cam load torque (right axis) against TIME, one panel per speed,
    all panels on the same time axis. Dotted lines mark 90/180/270/360 deg of cam travel."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    global Ramp_slope, StopTime

    saved = (Ramp_slope, StopTime)
    sign = 1.0 if DIRECTION.strip().upper() == "FORWARD" else -1.0
    runs = []
    for v in speeds:
        Ramp_slope = sign * v                    # cam deg/s
        StopTime = 360.0 / v + 1.0               # one revolution + 1 s margin
        log = simulate()
        t = np.asarray(log["t"])
        ang = np.asarray(log["Gdeg"])
        travel = np.abs(ang - ang[0])
        end = np.nonzero(travel >= 360.0)[0]
        n = end[0] + 1 if end.size else len(ang)
        runs.append((v, t[:n], travel[:n], np.asarray(log["Li"])[:n], np.asarray(log["Tab"])[:n]))
        print("speed %g cam deg/s: 360 deg reached after %.2f s" % (v, t[n - 1]))
    Ramp_slope, StopTime = saved                 # restore the original settings

    plt.rcParams.update({"font.size": 12})
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 3.0 * len(runs)),
                             sharey=True, squeeze=False)
    axes = axes[:, 0]
    tmax = max(r[1][-1] for r in runs)
    ylo = min(r[3].min() for r in runs)
    yhi = max(r[3].max() for r in runs)
    pad = 0.08 * (yhi - ylo)

    for i, (ax1, (v, t, travel, cur, load)) in enumerate(zip(axes, runs)):
        l1, = ax1.plot(t, cur, color=colors[i % len(colors)], linewidth=1.0,
                       label="Motor current at %g °/s cam" % v)
        ax1.set_ylabel("Motor current [A]")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, tmax * 1.02)
        ax1.set_ylim(ylo - pad, yhi + pad)

        ax2 = ax1.twinx()
        l2, = ax2.plot(t, load, color="tab:orange", linewidth=2.0, label="Cam torque (simulation input)")
        ax2.set_ylabel("Load torque", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        lo, hi = ax1.get_ylim()
        ax2.set_ylim(lo * Ktc / Gload, hi * Ktc / Gload)   # same zero, same "current needed" scale

        # mark 90/180/270/360 deg of cam travel
        tr = blended_transform_factory(ax1.transData, ax1.transAxes)
        for a in (90, 180, 270, 360):
            k = np.nonzero(travel >= a)[0]
            if k.size:
                ax1.axvline(t[k[0]], color="gray", linestyle=":", linewidth=1.2)
                ax1.text(t[k[0]], 0.97, "%d°" % a, transform=tr, ha="center", va="top", fontsize=9,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1), clip_on=True)
        ax1.legend(handles=[l1, l2], loc="lower right", fontsize=9)

    # ---- link the zoom by CAM ANGLE: zooming one panel shows the same angle range in all panels ----
    full = (0.0, tmax * 1.02)
    maps = []
    for v, t, travel, cur, load in runs:
        maps.append((t, np.maximum.accumulate(travel)))       # monotonic travel for interpolation
    busy = {"on": False}

    def on_xlim(ax):
        if busy["on"]:
            return
        src = list(axes).index(ax)
        t, trav = maps[src]
        x0, x1 = ax.get_xlim()
        a0 = float(np.interp(x0, t, trav))
        a1 = float(np.interp(x1, t, trav))
        busy["on"] = True
        try:
            for j, other in enumerate(axes):
                if j == src:
                    continue
                if a0 <= 0.0 and a1 >= 360.0:                # whole revolution visible -> common time view
                    other.set_xlim(full)
                else:
                    tj, travj = maps[j]
                    other.set_xlim(float(np.interp(a0, travj, tj)), float(np.interp(a1, travj, tj)))
            fig.canvas.draw_idle()
        finally:
            busy["on"] = False

    for ax1 in axes:
        ax1.callbacks.connect("xlim_changed", on_xlim)
    axes[-1].set_xlabel("Time [s]")
    fig._zoom_link = on_xlim                     # keep a reference to the callback
    axes[0].set_title("Motor current vs cam load torque over one cam revolution at different cam speeds", pad=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print("saved %s" % filename)
    return fig
def compare_speeds_pwm(speeds=(75, 150, 250, 350), filename="pwm_vs_load_speeds.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    global Ramp_slope, StopTime

    saved = (Ramp_slope, StopTime)
    sign = 1.0 if DIRECTION.strip().upper() == "FORWARD" else -1.0
    runs = []
    for v in speeds:
        Ramp_slope = sign * v                    # cam deg/s
        StopTime = 360.0 / v + 1.0               # one revolution + 1 s margin
        log = simulate()
        t = np.asarray(log["t"])
        ang = np.asarray(log["Gdeg"])
        travel = np.abs(ang - ang[0])
        end = np.nonzero(travel >= 360.0)[0]
        n = end[0] + 1 if end.size else len(ang)
        pwm = 100.0 * np.asarray(log["PWM"])[:n]                 # duty cycle in %
        runs.append((v, t[:n], travel[:n], pwm, np.asarray(log["Tab"])[:n]))
        print("speed %g cam deg/s: 360 deg after %.2f s, PWM %.1f ... %.1f %%"
              % (v, t[n - 1], pwm.min(), pwm.max()))
    Ramp_slope, StopTime = saved                 # restore the original settings

    plt.rcParams.update({"font.size": 12})
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 3.0 * len(runs)),
                             sharey=True, squeeze=False)
    axes = axes[:, 0]
    tmax = max(r[1][-1] for r in runs)
    ylo = min(min(r[3].min() for r in runs), 0.0)
    yhi = max(max(r[3].max() for r in runs), 0.0)
    pad = 0.08 * (yhi - ylo)
    lmax = max(np.abs(r[4]).max() for r in runs)

    for i, (ax1, (v, t, travel, pwm, load)) in enumerate(zip(axes, runs)):
        l1, = ax1.plot(t, pwm, color=colors[i % len(colors)], linewidth=1.0,
                       label="PWM at %g °/s cam" % v)
        ax1.set_ylabel("PWM [%]")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, tmax * 1.02)
        ax1.set_ylim(ylo - pad, yhi + pad)
        for lim in (Sat_upper, Sat_lower):                         # controller saturation +-95 %
            if ylo - pad < lim < yhi + pad:
                ax1.axhline(lim, color="black", linestyle="--", linewidth=0.8)

        ax2 = ax1.twinx()
        l2, = ax2.plot(t, load, color="tab:orange", linewidth=2.0, label="Cam torque (simulation input)")
        ax2.set_ylabel("Load torque [N·m]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        lo, hi = ax1.get_ylim()
        k = lmax / max(abs(lo), abs(hi))                            # zero of both axes at the same height
        ax2.set_ylim(lo * k, hi * k)

        tr = blended_transform_factory(ax1.transData, ax1.transAxes)
        for a in (90, 180, 270, 360):
            j = np.nonzero(travel >= a)[0]
            if j.size:
                ax1.axvline(t[j[0]], color="gray", linestyle=":", linewidth=1.2)
                ax1.text(t[j[0]], 0.97, "%d°" % a, transform=tr, ha="center", va="top", fontsize=9,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1), clip_on=True)
        ax1.legend(handles=[l1, l2], loc="lower right", fontsize=9)

    # ---- link the zoom by CAM ANGLE: zooming one panel shows the same angle range in all panels ----
    full = (0.0, tmax * 1.02)
    maps = [(t, np.maximum.accumulate(travel)) for v, t, travel, pwm, load in runs]
    busy = {"on": False}

    def on_xlim(ax):
        if busy["on"]:
            return
        src = list(axes).index(ax)
        t, trav = maps[src]
        x0, x1 = ax.get_xlim()
        a0 = float(np.interp(x0, t, trav))
        a1 = float(np.interp(x1, t, trav))
        busy["on"] = True
        try:
            for j, other in enumerate(axes):
                if j == src:
                    continue
                if a0 <= 0.0 and a1 >= 360.0:                # whole revolution visible -> common time view
                    other.set_xlim(full)
                else:
                    tj, travj = maps[j]
                    other.set_xlim(float(np.interp(a0, travj, tj)), float(np.interp(a1, travj, tj)))
            fig.canvas.draw_idle()
        finally:
            busy["on"] = False

    for ax1 in axes:
        ax1.callbacks.connect("xlim_changed", on_xlim)
    axes[-1].set_xlabel("Time [s]")
    fig._zoom_link = on_xlim                     # keep a reference to the callback
    axes[0].set_title("PWM vs cam load torque over one cam revolution at different cam speeds", pad=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print("saved %s" % filename)
    return fig

if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) > 1:
        HALL_MODE = sys.argv[1]
    print("HALL_MODE = %s   DIRECTION = %s   START_ANGLE_DEG = %g   WRAP_LOAD_TABLE = %s"
          % (HALL_MODE, DIRECTION, START_ANGLE_DEG, WRAP_LOAD_TABLE))
    print("Combined Inertia (Display) = %.6e kg m^2" % J)
    t0 = time.time()
    log = simulate()
    print("simulated %.1f s in %.1f s wall time, %d logged samples" % (StopTime, time.time() - t0, len(log["t"])))

    # a few values to compare with the Simulink scopes
    t = log["t"]
    for tq in (0.5, 1.0, 2.0, 4.0, 8.0):
        i = min(range(len(t)), key=lambda j: abs(t[j] - tq))
        print("t=%.2f  Gdeg=%9.3f deg  target=%8.3f deg  Li=%8.3f A  Wi=%9.3f rad/s  PWM=%7.4f  Tab=%7.3f N*m"
              % (t[i], log["Gdeg"][i], log["Target_Cam_angle"][i], log["Li"][i], log["Wi"][i], log["PWM"][i], log["Tab"][i]))

    # error cam angle - target after 1 s (for a quick check)
    sel = [j for j in range(len(t)) if t[j] > 1.0]
    err = [log["Gdeg"][j] - log["Target_Cam_angle"][j] for j in sel]
    mean_e = sum(err) / len(err)
    std_e = (sum((x - mean_e) ** 2 for x in err) / len(err)) ** 0.5
    sat = sum(1 for j in sel if abs(log["PWM"][j]) >= 0.9499) / len(sel)
    print("error after 1 s: mean %.2f deg  std %.2f deg  max %.2f deg  |  PWM saturated %.0f %% of the time"
          % (mean_e, std_e, max(abs(x) for x in err), 100 * sat))

    # PWM roughness = PWM minus its 50 ms moving average (after 1 s)
    pwm = log["PWM"]
    half = 250                                        # 501 logged samples = 50 ms
    cs = [0.0]
    for x in pwm:
        cs.append(cs[-1] + x)
    rough = []
    for j in sel:
        if j - half >= 0 and j + half < len(pwm):
            rough.append(pwm[j] - (cs[j + half + 1] - cs[j - half]) / (2 * half + 1))
    rstd = (sum(x * x for x in rough) / len(rough)) ** 0.5
    print("PWM roughness after 1 s: std %.3f  peak %.3f" % (rstd, max(abs(x) for x in rough)))
    plot_current_vs_load(log)
    #compare_speeds()
    #compare_speeds_pwm()
    try:
        import matplotlib.pyplot as plt
        names = list(SCOPES.keys())
        ncol = 3
        nrow = (len(names) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3 * nrow), squeeze=False)
        for ax, name in zip(axes.flat, names):
            for sig in SCOPES[name]:
                ax.plot(log["t"], log[sig], label=sig)
            ax.set_title(name, fontsize=9)
            ax.grid(True)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass