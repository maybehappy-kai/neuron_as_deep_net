# tools/calibrate_invivo_params.py

import os
import sys
import math
import time
import numpy as np
import concurrent.futures

# 抑制 GUI
os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from neuron import h

# --- 实验配置 ---
DURATION = 60000.0  # 60 秒长程仿真
EVAL_WINDOW = 58000.0  # 舍弃前 2 秒瞬态，只评估稳态
DT = 0.1

# --- 搜索网格配置 ---
MU_E_LIST = [10.5, 11.0, 11.5]
MU_I_LIST = [7.5, 8.0, 8.5]
SIGMA_RATIO_LIST = [0.3, 0.5, 0.7]  # 发放率波动的强度 (std/mean)
TAU_FLUC = 50.0  # OU 过程的时间常数 (ms)


def generate_ou_rate_trace(mu, sigma_ratio, tau, duration_ms, dt, rng):
    """生成符合 OU 过程的非齐次发放率迹"""
    steps = int(duration_ms / dt)
    sigma = mu * sigma_ratio
    rate_trace = np.zeros(steps)
    curr_rate = mu

    dw_factor = sigma * math.sqrt(2 * dt / tau)
    decay_factor = dt / tau

    for i in range(steps):
        curr_rate += (mu - curr_rate) * decay_factor + dw_factor * rng.standard_normal()
        rate_trace[i] = max(0.1, curr_rate)
    return rate_trace


def generate_inhomogeneous_spikes(rate_trace, dt, rng):
    """根据发放率迹生成脉冲 (二项分布近似法)"""
    prob = rate_trace * (dt / 1000.0)
    rand_vals = rng.rand(len(rate_trace))
    spike_indices = np.where(rand_vals < prob)[0]
    return spike_indices * dt


def worker_evaluate_pattern(mu_e, mu_i, s_ratio, base_g_in):
    """评估单个非齐次参数模式"""
    print(f"  [Worker] 正在测算模式: E_avg={mu_e}, I_avg={mu_i}, Sigma_Ratio={s_ratio}")

    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )
    env.warmup(target_v=None)

    topo_meta = env.get_topology_metadata()
    num_synapses = topo_meta["num_synapses"]
    input_map = topo_meta["input_map"]
    syn_types = topo_meta["synapse_types"]
    seg_lengths = np.array([seg.sec.L / seg.sec.nseg for seg in env.segments], dtype=np.float32)
    mean_length = np.mean(seg_lengths)

    soma = env.cell.soma[0]
    stim = h.IClamp(soma(0.5))
    stim.delay = 0.0
    stim.dur = DURATION

    seed = int(mu_e * 1000 + mu_i * 100 + s_ratio * 10)
    rng = np.random.RandomState(seed)

    trace_e = generate_ou_rate_trace(mu_e, s_ratio, TAU_FLUC, DURATION, DT, rng)
    trace_i = generate_ou_rate_trace(mu_i, s_ratio, TAU_FLUC, DURATION, DT, rng)

    spike_events = []
    for syn_idx in range(num_synapses):
        seg_idx = input_map[syn_idx]
        length_factor = seg_lengths[seg_idx] / mean_length
        base_trace = trace_e if syn_types[syn_idx] == 1 else trace_i
        scaled_trace = base_trace * length_factor

        spikes = generate_inhomogeneous_spikes(scaled_trace, DT, rng)
        for t in spikes:
            spike_events.append((syn_idx, t))

    # Run A: 测 V_mean 和 F_out
    stim.amp = 0.0
    v_trace_0 = env.run_simulation(spike_events, total_duration=DURATION)
    eval_steps = int(EVAL_WINDOW / DT)
    v_eval = v_trace_0[-eval_steps:]
    v_mean = np.mean(v_eval)
    v_std = np.std(v_eval)
    spikes_out = np.sum((v_eval[:-1] < -20.0) & (v_eval[1:] >= -20.0))
    f_out = spikes_out / (EVAL_WINDOW / 1000.0)

    # Run B: 测电导
    stim.amp = -0.1
    v_trace_stim = env.run_simulation(spike_events, total_duration=DURATION)
    v_mean_stim = np.mean(v_trace_stim[-eval_steps:])
    delta_v = v_mean_stim - v_mean
    r_in = delta_v / -0.1 if delta_v != 0 else float('inf')
    g_ratio = (1.0 / r_in) / base_g_in if base_g_in else 1.0

    return mu_e, mu_i, s_ratio, v_mean, v_std, f_out, g_ratio


def get_base_conductance():
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )
    env.warmup(target_v=None)
    soma = env.cell.soma[0]
    stim = h.IClamp(soma(0.5))
    stim.delay, stim.dur = 0.0, DURATION

    stim.amp = 0.0
    v_0 = env.run_simulation([], total_duration=DURATION)

    stim.amp = -0.1
    v_stim = env.run_simulation([], total_duration=DURATION)

    eval_steps = int(EVAL_WINDOW / DT)
    delta_v = np.mean(v_stim[-eval_steps:]) - np.mean(v_0[-eval_steps:])
    return 1.0 / (delta_v / -0.1)


def main():
    base_g_in = get_base_conductance()
    print(f"✅ 测得静息基础电导 G_base: {base_g_in:.6f} uS")

    patterns = []
    for mu_e in MU_E_LIST:
        for mu_i in MU_I_LIST:
            for s_ratio in SIGMA_RATIO_LIST:
                patterns.append((mu_e, mu_i, s_ratio))

    print(f"\n🚀 开始长程(60s)非齐次模式搜索，共 {len(patterns)} 组配置...")
    results = []
    start_t = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(worker_evaluate_pattern, p[0], p[1], p[2], base_g_in): p for p in patterns}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            # res[4] 现在是 v_std，res[5] 是 f_out，res[6] 是 g_ratio
            marker = "🌟 [命中目标]" if 0.8 <= res[5] <= 1.5 else ""
            print(
                f"  [完成] E={res[0]}, I={res[1]}, Sigma={res[2]} -> V={res[3]:.2f}mV (±{res[4]:.2f}mV), F_out={res[5]:.2f}Hz, G={res[6]:.2f}x {marker}")

    results = np.array(results)
    best_idx = np.argmin(np.abs(results[:, 5] - 1.0))  # 用 F_out 逼近 1Hz 寻找
    best = results[best_idx]

    print(f"\n🎉 搜索结束！耗时 {(time.time() - start_t) / 60:.2f} 分钟。")
    print(f"📌 最接近 1Hz 的配置: E_avg={best[0]}Hz, I_avg={best[1]}Hz, Fluctuation_Ratio={best[2]}")
    print(
        f"   产生指标: V_mean={best[3]:.2f}mV (±{best[4]:.2f}mV), Output_Rate={best[5]:.2f}Hz, Conductance={best[6]:.2f}x")


if __name__ == '__main__':
    main()