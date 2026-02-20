import os

os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'
import sys
import json
import numpy as np
from scipy.optimize import least_squares
from neuron import h

print("Loading L5PC Model...")
h.load_file('nrngui.hoc')
h.load_file("import3d.hoc")
h.load_file("L5PC_NEURON_simulation/L5PCbiophys5b.hoc")
h.load_file("L5PC_NEURON_simulation/L5PCtemplate_2.hoc")

cell = h.L5PCtemplate("L5PC_NEURON_simulation/morphologies/cell1.asc")
h.celsius = 34.0
h.dt = 0.1
h.cvode_active(0)

soma = cell.soma[0]
stim = h.IClamp(soma(0.5))

# 记录原始的 pas 属性，避免被覆盖
orig_pas = {}
for sec in cell.all:
    for seg in sec:
        if hasattr(seg, 'pas'):
            orig_pas[seg] = {'g': seg.g_pas, 'e': seg.e_pas}


def apply_bg(ge, gi):
    """根据物理等效公式，应用全局背景电导密度"""
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
    """进行一次虚拟测量：测静息电位 + 输入电阻"""
    stim.amp = 0.0
    h.finitialize(-67.7)
    h.continuerun(2000.0)
    v_rest = soma(0.5).v

    stim.delay = 2000.0
    stim.dur = 500.0
    stim.amp = -0.1

    h.continuerun(2500.0)
    v_i = soma(0.5).v

    delta_v = v_i - v_rest
    rin = delta_v / -0.1

    stim.amp = 0.0

    return v_rest, rin


# ==========================================
# 标定目标设置
# ==========================================
TARGET_V = -67.7

apply_bg(0, 0)
print("\nMeasuring Baseline (No Background)...")
v_base, rin_base = measure_properties()
print(f"Baseline: V_rest = {v_base:.2f} mV, R_in = {rin_base:.2f} MOhms")

TARGET_RIN = rin_base * 0.25
print(f"\n[Targets] V_rest: {TARGET_V} mV, R_in: {TARGET_RIN:.2f} MOhms")
print("=========================================\n")

# ==========================================
# 优化循环 (残差方程求解法)
# ==========================================
SCALE = 1e-4
iteration = 0


def residuals_func(vars_scaled):
    """
    返回当前状态与目标状态的差距（残差）。
    当返回值为 [0, 0] 时，代表完美达成目标。
    """
    global iteration
    ge = vars_scaled[0] * SCALE
    gi = vars_scaled[1] * SCALE

    apply_bg(ge, gi)
    v_rest, rin = measure_properties()

    # 计算残差 (Residuals)
    res_v = v_rest - TARGET_V

    # 将电阻残差按比例放大，使其量级与电压残差接近，有助于求解器平衡两个目标
    res_r = ((rin - TARGET_RIN) / TARGET_RIN) * 10

    iteration += 1
    print(f"Eval {iteration:3d} | G_exc: {ge:.2e}, G_inh: {gi:.2e} | "
          f"V_rest: {v_rest:.3f} mV, R_in: {rin:.2f} MOhm")
    sys.stdout.flush()

    return [res_v, res_r]


print("Starting Equation Solver (Least Squares)...\n")

# x0 是缩放后的初始猜测值
x0_scaled = [1.22, 5.54]

# bounds=(0, np.inf) 确保求解器探索的所有变量（缩放后）都大于等于 0
res = least_squares(
    residuals_func,
    x0=x0_scaled,
    bounds=(0, np.inf),
    ftol=1e-3,  # 只要残差平方和小于 0.001 就立刻停
    xtol=1e-3,  # 只要变量不再发生明显变化就停
    gtol=1e-3  # 只要梯度足够平缓就停
)

# 还原缩放比例
best_ge = res.x[0] * SCALE
best_gi = res.x[1] * SCALE

# ==========================================
# 验证与保存结果
# ==========================================
print("\nSolver finished. Success:", res.success)
print("Reason:", res.message)

apply_bg(best_ge, best_gi)
v_final, rin_final = measure_properties()

print("\n" + "=" * 40)
print(f"Final V_rest = {v_final:.2f} mV")
print(f"Final R_in   = {rin_final:.2f} MOhms")
print("=" * 40)

anchor_data = {
    "g_exc_max": float(best_ge),
    "g_inh_max": float(best_gi),
    "target_v_max": TARGET_V
}
with open("bg_anchors.json", "w") as f:
    json.dump(anchor_data, f, indent=4)
print("✅ Anchor values saved to bg_anchors.json")