import os
import sys
import json
import argparse
import numpy as np
from scipy.optimize import least_squares

# 抑制 GUI
os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.config as cfg
from neuron import h


def main():
    parser = argparse.ArgumentParser(description="Calibrate background conductance anchors for L5PC.")
    parser.add_argument("--morph_dir", type=str, default="L5PC_NEURON_simulation",
                        help="L5PC 神经元模型文件夹的根目录")
    parser.add_argument("--out_json", type=str, default="configs/bg_anchors.json",
                        help="生成的锚点 json 文件保存路径")
    parser.add_argument("--target_v", type=float, default=cfg.ACTIVE_TARGET_V,
                        help="标定时的目标静息电位")
    parser.add_argument("--g_ratio", type=float, default=cfg.ACTIVE_G_RATIO,
                        help="目标电导相对于绝对静息的放大倍数")
    args = parser.parse_args()

    print("Loading L5PC Model for Calibration...")
    h.load_file('nrngui.hoc')
    h.load_file("import3d.hoc")

    biophys_path = os.path.join(args.morph_dir, "L5PCbiophys5b.hoc")
    template_path = os.path.join(args.morph_dir, "L5PCtemplate_2.hoc")
    morph_path = os.path.join(args.morph_dir, "morphologies", "cell1.asc")

    if not os.path.exists(biophys_path):
        raise FileNotFoundError(f"找不到模型文件: {biophys_path}")

    h.load_file(biophys_path)
    h.load_file(template_path)

    cell = h.L5PCtemplate(morph_path)
    h.celsius = 34.0
    h.dt = 0.1
    h.cvode_active(0)

    soma = cell.soma[0]
    stim = h.IClamp(soma(0.5))

    orig_pas = {}
    for sec in cell.all:
        for seg in sec:
            if hasattr(seg, 'pas'):
                orig_pas[seg] = {'g': seg.g_pas, 'e': seg.e_pas}

    def apply_bg(ge, gi):
        """物理等效公式: -80.0 为纯物理常数(GABA反转电位)，绝不能改"""
        for sec in cell.all:
            for seg in sec:
                if hasattr(seg, 'pas'):
                    g_old = orig_pas[seg]['g']
                    e_old = orig_pas[seg]['e']
                    g_new = g_old + ge + gi
                    e_new = (g_old * e_old + ge * 0.0 + gi * (-80.0)) / g_new
                    seg.g_pas = g_new
                    seg.e_pas = e_new

    def measure_properties():
        """测试此时的电压与输入电阻"""
        stim.amp = 0.0
        h.finitialize(args.target_v)  # 预猜值为 target_v
        h.continuerun(2000.0)
        v_rest = soma(0.5).v

        stim.delay = 2000.0
        stim.dur = 500.0
        stim.amp = -0.1
        h.continuerun(2500.0)
        v_i = soma(0.5).v

        delta_v = v_i - v_rest
        rin = delta_v / -0.1 if delta_v != 0 else float('inf')
        stim.amp = 0.0
        return v_rest, rin

    # --- 测定纯净基线 ---
    apply_bg(0, 0)
    print("\nMeasuring Baseline (No Background)...")
    v_base, rin_base = measure_properties()
    print(f"Baseline: V_rest = {v_base:.2f} mV, R_in = {rin_base:.2f} MOhms")

    # 【核心修正】：利用测算出的真实电导放大率计算目标电阻
    TARGET_RIN = rin_base / args.g_ratio
    print(
        f"\n[Targets] V_rest: {args.target_v} mV, R_in: {TARGET_RIN:.2f} MOhms (Conductance Multiplier: {args.g_ratio}x)")
    print("=========================================\n")

    SCALE = 1e-4
    iteration_count = [0]

    def residuals_func(vars_scaled):
        ge = vars_scaled[0] * SCALE
        gi = vars_scaled[1] * SCALE
        apply_bg(ge, gi)
        v_rest, rin = measure_properties()

        res_v = v_rest - args.target_v
        res_r = ((rin - TARGET_RIN) / TARGET_RIN) * 10

        iteration_count[0] += 1
        print(f"Eval {iteration_count[0]:3d} | G_exc: {ge:.2e}, G_inh: {gi:.2e} | "
              f"V_rest: {v_rest:.3f} mV, R_in: {rin:.2f} MOhm")
        sys.stdout.flush()
        return [res_v, res_r]

    print("Starting Equation Solver (Least Squares)...\n")
    x0_scaled = [0.365, 1.15]
    res = least_squares(
        residuals_func, x0=x0_scaled, bounds=(0, np.inf),
        ftol=1e-3, xtol=1e-3, gtol=1e-3
    )

    best_ge = res.x[0] * SCALE
    best_gi = res.x[1] * SCALE

    apply_bg(best_ge, best_gi)
    v_final, rin_final = measure_properties()
    print("\n" + "=" * 40)
    print(f"Final V_rest = {v_final:.2f} mV")
    print(f"Final R_in   = {rin_final:.2f} MOhms")
    print("=" * 40)

    anchor_data = {
        "g_exc_max": float(best_ge),
        "g_inh_max": float(best_gi),
        "target_v_max": args.target_v
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w") as f:
        json.dump(anchor_data, f, indent=4)
    print(f"✅ Anchor values successfully saved to {args.out_json}")


if __name__ == "__main__":
    main()