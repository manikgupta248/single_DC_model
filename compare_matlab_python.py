"""
Compare the Simulink model (emotor_cam_move.slx) with its Python port (emotor_cam_move.py).

Metrics per signal (Simulink = reference, Python = test):
    RMSE, NRMSE (RMSE / range of reference), MAE, max |error|, bias (mean error), R^2

Two ways to get the Simulink results
------------------------------------
1) MATLAB Engine (default): the script starts MATLAB, loads the model, attaches temporary
   "To Workspace" blocks IN MEMORY to the signals listed in SIGNALS, runs the simulation and
   closes the model WITHOUT saving, so your .slx file is never modified.
       pip install matlabengine      (the version must match your MATLAB release, see below)
       python compare_matlab_python.py --model emotor_cam_move.slx

2) CSV file (no MATLAB needed by Python): a csv with a time column "t" and one column per signal
   name in SIGNALS (e.g. t,PWM,Gdeg,Wi,Li,Tab,Camshaft_quantanizer), exported from MATLAB.
       python compare_matlab_python.py --csv simulink_results.csv

The Python model is imported from emotor_cam_move.py (same folder) and run with
CONTROL_MODE = "CONTROLLER" and HALL_MODE = "NOINTERPOLATION" (= the Simulink model).
"""
import argparse
import os
import sys

import numpy as np

# python signal name -> block name in the Simulink model (output port 1 of that block)
SIGNALS = {
    "PWM": "Simplified, cascaded Controller1",
    "Gdeg": "Gdeg",                              # cam angle [deg]
    "Wi": "Wi",                                  # e-motor velocity [rad/s]
    "Li": "Li",                                  # motor current [A]
    "Tab": "Tab",                                # cam load torque
    "Camshaft_quantanizer": "Camshaft_quantanizer",   # quantized motor angle [rad]
}
# signals that are piecewise constant -> resample with "previous value" instead of linear
STEPWISE = {"Camshaft_quantanizer"}


# ----------------------------------------------------------------------------
# Python model
# ----------------------------------------------------------------------------
def run_python(stop_time, folder):
    sys.path.insert(0, folder)
    import emotor_cam_move as m
    m.CONTROL_MODE = "CONTROLLER"
    m.HALL_MODE = "NOINTERPOLATION"
    m.StopTime = stop_time
    print("Running Python model (StopTime = %g s) ..." % stop_time)
    log = m.simulate()
    return {k: np.asarray(v, dtype=float) for k, v in log.items()}


# ----------------------------------------------------------------------------
# Simulink model through the MATLAB Engine
# ----------------------------------------------------------------------------
def run_simulink(model_path, stop_time):
    import matlab.engine                      # pip install matlabengine
    folder, fname = os.path.split(os.path.abspath(model_path))
    name = os.path.splitext(fname)[0]
    print("Starting MATLAB ...")
    eng = matlab.engine.start_matlab()
    out = {}
    try:
        eng.cd(folder, nargout=0)
        eng.load_system(name, nargout=0)
        eng.eval("set_param('%s','StopTime','%s');" % (name, repr(float(stop_time))), nargout=0)
        for var, blk in SIGNALS.items():     # temporary To Workspace blocks (in memory only)
            src = "%s/%s" % (name, blk)
            tw = "%s/zz_cmp_%s" % (name, var)
            eng.eval(
                "add_block('simulink/Sinks/To Workspace','%s','VariableName','cmp_%s',"
                "'SaveFormat','Timeseries','MaxDataPoints','inf');"
                "ph=get_param('%s','PortHandles');pt=get_param('%s','PortHandles');"
                "add_line('%s',ph.Outport(1),pt.Inport(1));" % (tw, var, src, tw, name),
                nargout=0)
        print("Running Simulink model ...")
        eng.eval("simOut = sim('%s','ReturnWorkspaceOutputs','on');" % name, nargout=0)
        for var in SIGNALS:
            eng.eval("ts_=simOut.get('cmp_%s');t_=ts_.Time(:);d_=ts_.Data;d_=d_(:);" % var, nargout=0)
            out[var] = (np.array(eng.workspace["t_"], dtype=float).ravel(),
                        np.array(eng.workspace["d_"], dtype=float).ravel())
    finally:
        try:
            eng.close_system(name, 0, nargout=0)      # 0 = do NOT save -> .slx stays untouched
        finally:
            eng.quit()
    return out


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(data["t"], dtype=float)
    return {var: (t, np.asarray(data[var], dtype=float)) for var in SIGNALS if var in data.dtype.names}


# ----------------------------------------------------------------------------
# Resampling and metrics
# ----------------------------------------------------------------------------
def resample(t_src, y_src, t_dst, stepwise=False):
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    # variable-step logs can contain repeated time stamps (events): keep the last value of each
    t_u, idx = np.unique(t_src[::-1], return_index=True)
    y_u = y_src[::-1][idx]
    if stepwise:
        k = np.clip(np.searchsorted(t_u, t_dst, side="right") - 1, 0, len(t_u) - 1)
        return y_u[k]
    return np.interp(t_dst, t_u, y_u)


def metrics(ref, test):
    err = test - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sst = float(np.sum((ref - np.mean(ref)) ** 2))
    rng = float(np.max(ref) - np.min(ref))
    return {
        "RMSE": rmse,
        "NRMSE_%": 100.0 * rmse / rng if rng > 0 else float("nan"),
        "MAE": float(np.mean(np.abs(err))),
        "MaxAbsErr": float(np.max(np.abs(err))),
        "Bias": float(np.mean(err)),
        "R2": 1.0 - float(np.sum(err ** 2)) / sst if sst > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="emotor_cam_move.slx", help="Simulink model (MATLAB Engine mode)")
    ap.add_argument("--csv", help="use Simulink results from this csv instead of the MATLAB Engine")
    ap.add_argument("--stop-time", type=float, default=8.0, help="simulation stop time [s] (default 8)")
    ap.add_argument("--t-start", type=float, default=0.0, help="ignore data before this time [s] in the metrics")
    ap.add_argument("--out", default="comparison", help="prefix of the output files (png, csv)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    folder = os.path.dirname(os.path.abspath(__file__))
    py = run_python(a.stop_time, folder)
    sl = load_csv(a.csv) if a.csv else run_simulink(a.model, a.stop_time)

    t = py["t"]
    sel = t >= a.t_start
    rows, curves = [], {}
    for var in SIGNALS:
        if var not in sl:
            print("  (no Simulink data for %s, skipped)" % var)
            continue
        ref = resample(sl[var][0], sl[var][1], t, var in STEPWISE)
        test = py[var]
        m = metrics(ref[sel], test[sel])
        rows.append((var, m))
        curves[var] = (ref, test)

    cols = ["RMSE", "NRMSE_%", "MAE", "MaxAbsErr", "Bias", "R2"]
    print("\nSimulink = reference, Python = test, t >= %g s, %d samples\n" % (a.t_start, int(sel.sum())))
    print("%-22s" % "signal" + "".join("%13s" % c for c in cols))
    for var, m in rows:
        print("%-22s" % var + "".join("%13.5g" % m[c] for c in cols))

    with open(a.out + "_metrics.csv", "w") as f:
        f.write("signal," + ",".join(cols) + "\n")
        for var, m in rows:
            f.write(var + "," + ",".join("%.8g" % m[c] for c in cols) + "\n")
    print("\nsaved %s_metrics.csv" % a.out)

    if not a.no_plot and curves:
        import matplotlib.pyplot as plt
        n = len(curves)
        fig, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n), squeeze=False, sharex=True)
        for i, (var, (ref, test)) in enumerate(curves.items()):
            axes[i, 0].plot(t, ref, label="Simulink", linewidth=1.4)
            axes[i, 0].plot(t, test, "--", label="Python", linewidth=1.0)
            axes[i, 0].set_ylabel(var)
            axes[i, 0].legend(fontsize=7)
            axes[i, 0].grid(True, alpha=0.4)
            axes[i, 1].plot(t, test - ref, color="tab:red", linewidth=0.8)
            axes[i, 1].set_ylabel("error (Py - Sl)")
            axes[i, 1].grid(True, alpha=0.4)
        axes[-1, 0].set_xlabel("Time [s]")
        axes[-1, 1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(a.out + ".png", dpi=200)
        print("saved %s.png" % a.out)
        plt.show()


if __name__ == "__main__":
    main()
