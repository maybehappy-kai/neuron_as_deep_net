import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

try:
    import hdf5plugin
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.l5pc_env import L5PC_Env


def force_align(trace, target_len):
    """安全函数：强制数组对齐至 target_len，防止浮点误差导致少/多一个步长"""
    if len(trace) > target_len:
        return trace[:target_len]
    elif len(trace) < target_len:
        return np.pad(trace, (0, target_len - len(trace)), 'edge')
    return trace


def shift_trace(trace, shift_steps, pad_value):
    """【新增核心】将一维数组平移指定的步数，并用背景电压填充空缺，完美模拟不同时刻的刺激"""
    if shift_steps == 0:
        return trace.copy()

    result = np.full_like(trace, pad_value)
    if shift_steps > 0:
        # 向右平移 (时间延后)
        result[shift_steps:] = trace[:-shift_steps]
    else:
        # 向左平移 (时间提前)
        shift_abs = abs(shift_steps)
        result[:-shift_abs] = trace[shift_abs:]
    return result


def main():
    h5_path = "results/temporal_coupling/L5PC_TemporalCoupling.h5"
    out_dir = "results/temporal_coupling/visualizations"
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔍 正在加载时序耦合数据集: {h5_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"文件不存在: {h5_path}。请确认已跑完时序耦合实验。")

    with h5py.File(h5_path, 'r') as f:
        target_v = f.attrs['target_v_mV']
        dt_step = f.attrs['dt']
        total_steps = f.attrs['total_steps']
        num_synapses = f.attrs['num_synapses']

        inputs = f['dataset/train/inputs'][:]
        targets = f['dataset/train/targets'][:]
        conditions_dt = f['dataset/train/conditions_dt'][:]

    num_trials = inputs.shape[0]
    time_axis = np.arange(total_steps) * dt_step

    # 1. 解析出到底测试了哪几对突触
    pair_trials = {}
    for k in range(num_trials):
        active_syns = np.where(np.sum(inputs[k], axis=0) > 0)[0]
        if len(active_syns) == 2:
            e, i = active_syns[0], active_syns[1]
            pair = (e, i)
            if pair not in pair_trials:
                pair_trials[pair] = []
            pair_trials[pair].append(k)

    print(f"✅ 解析出 {len(pair_trials)} 个 E-I 突触对。")

    # 2. 【智能双轨切换】基准波形提取
    baseline_h5 = "results/exp_active_pairs/final_dataset.h5"
    use_mock = True
    env = None
    single_traces_cache = None

    if os.path.exists(baseline_h5):
        print(f"📦 命中大规模配对数据集: {baseline_h5} \n⚡ 启用极速数组平移替代 NEURON 仿真！")
        use_mock = False
        with h5py.File(baseline_h5, 'r') as f_base:
            # 第一批数据的前 num_synapses 个 trial 刚好就是我们要的基准波形
            single_traces_cache = f_base['dataset/train/targets'][:num_synapses, :, 0]
    else:
        print("⚠️ 未检测到大规模基准数据集。将启动 NEURON 引擎进行单突触响应重构 (Mock 模式)...")
        env = L5PC_Env(
            morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
            biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
            template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
            dt=dt_step
        )
        env.warmup(target_v=target_v, anchors_path="configs/bg_anchors.json")

    TOTAL_DURATION = total_steps * dt_step
    BASE_STIM_TIME = 200.0

    # 大规模实验的基准脉冲时间设定是 100.0 ms
    ORIGINAL_STIM_TIME = 100.0

    # 3. 开始对每个对进行深度分析与绘图
    for pair_idx, (e, i) in enumerate(pair_trials.keys()):
        print(f"\n📊 正在分析第 {pair_idx + 1} 对: Syn_E={e}, Syn_I={i}")

        trials = pair_trials[(e, i)]
        dt_list, beta_list, r2_list = [], [], []

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f"Bilinear Rule Analysis for Temporal Coupling (Syn_E: {e}, Syn_I: {i})", fontsize=18,
                     fontweight='bold')

        ax_wave = plt.subplot2grid((2, 2), (0, 0))
        ax_scatter = plt.subplot2grid((2, 2), (1, 0))
        ax_beta = plt.subplot2grid((2, 2), (0, 1))
        ax_r2 = plt.subplot2grid((2, 2), (1, 1))

        for k in trials:
            dt_val = float(conditions_dt[k][0])
            dt_list.append(dt_val)

            v_c = targets[k, :, 0]

            if dt_val < 0:
                t_e, t_i = BASE_STIM_TIME + dt_val, BASE_STIM_TIME
            else:
                t_e, t_i = BASE_STIM_TIME, BASE_STIM_TIME - dt_val

            # 获取基准波形 (极速替换逻辑)
            if use_mock:
                v_e_raw = env.run_simulation([(e, t_e)], total_duration=TOTAL_DURATION).flatten()
                v_i_raw = env.run_simulation([(i, t_i)], total_duration=TOTAL_DURATION).flatten()
                v_e = force_align(v_e_raw, total_steps)
                v_i = force_align(v_i_raw, total_steps)
            else:
                # 算出目标时间距离原实验 100ms 的平移步数
                shift_e_steps = int(round((t_e - ORIGINAL_STIM_TIME) / dt_step))
                shift_i_steps = int(round((t_i - ORIGINAL_STIM_TIME) / dt_step))

                # 瞬间拿到精准的平移波形
                v_e = shift_trace(single_traces_cache[e], shift_e_steps, target_v)
                v_i = shift_trace(single_traces_cache[i], shift_i_steps, target_v)

            delta_c = v_c - target_v
            delta_e = v_e - target_v
            delta_i = v_i - target_v

            nl = delta_c - (delta_e + delta_i)
            prod = delta_e * delta_i

            # 回归窗口 (截取到主峰附近的区域)
            valid_start = int(max(0, (BASE_STIM_TIME - 50) / dt_step))
            valid_end = int(min(total_steps, (BASE_STIM_TIME + 250) / dt_step))

            nl_window = nl[valid_start:valid_end]
            prod_window = prod[valid_start:valid_end]

            # 线性回归
            slope, intercept, r_value, p_value, std_err = linregress(prod_window, nl_window)
            r2 = r_value ** 2

            beta_list.append(slope)
            r2_list.append(r2)

            # ---------------- 图 1 绘制：零时差波形 (保持你极其优秀的画图逻辑不变) ----------------
            if np.isclose(dt_val, 0.0, atol=1e-4):
                ax_wave.plot(time_axis, delta_e, label='ΔV_E', color='blue', alpha=0.5, linewidth=1.5, zorder=2)
                ax_wave.plot(time_axis, delta_i, label='ΔV_I', color='red', alpha=0.5, linewidth=1.5, zorder=2)
                ax_wave.plot(time_axis, delta_e + delta_i, label='Linear Sum', color='gray', alpha=0.4, linewidth=6,
                             zorder=1)
                ax_wave.plot(time_axis, delta_c, label='Coupled (ΔV_C)', color='black', linewidth=1.5, zorder=3)
                predicted_c = delta_e + delta_i + slope * prod
                ax_wave.plot(time_axis, predicted_c, label=f'Bilinear Pred (β={slope:.4f})', color='orange',
                             linestyle='--', linewidth=2, zorder=4)

                ax_wave.set_xlim(BASE_STIM_TIME - 20, BASE_STIM_TIME + 150)
                ax_wave.set_title(f"Waveforms at Δt = 0 ms", fontsize=14)
                ax_wave.set_xlabel("Time (ms)")
                ax_wave.set_ylabel("Membrane Potential ΔV (mV)")
                ax_wave.legend()
                ax_wave.grid(True, alpha=0.3)

                ax_scatter.scatter(prod_window, nl_window, color='purple', alpha=0.5, s=10)
                x_vals = np.array([np.min(prod_window), np.max(prod_window)])
                y_vals = intercept + slope * x_vals
                ax_scatter.plot(x_vals, y_vals, color='orange', linewidth=2,
                                label=f"Fit: y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r2:.4f}")
                ax_scatter.set_title("Point-by-Point Regression (Δt = 0 ms)", fontsize=14)
                ax_scatter.set_xlabel("Product: ΔV_E(t) × ΔV_I(t)")
                ax_scatter.set_ylabel("Nonlinearity: ΔV_C(t) - (ΔV_E(t) + ΔV_I(t))")
                ax_scatter.legend()
                ax_scatter.grid(True, alpha=0.3)

        # 排序并绘制趋势图
        sort_idx = np.argsort(dt_list)
        dt_arr = np.array(dt_list)[sort_idx]
        beta_arr = np.array(beta_list)[sort_idx]
        r2_arr = np.array(r2_list)[sort_idx]

        ax_beta.plot(dt_arr, beta_arr, marker='o', color='teal', linewidth=2)
        ax_beta.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax_beta.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax_beta.set_title("Bilinear Coefficient (β) vs Temporal Delay (Δt)", fontsize=14)
        ax_beta.set_xlabel("Δt = t_E - t_I (ms)")
        ax_beta.set_ylabel("β Coefficient")
        ax_beta.grid(True, alpha=0.3)

        ax_r2.plot(dt_arr, r2_arr, marker='s', color='darkred', linewidth=2)
        ax_r2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax_r2.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax_r2.set_title("Goodness of Fit ($R^2$) vs Temporal Delay (Δt)", fontsize=14)
        ax_r2.set_xlabel("Δt = t_E - t_I (ms)")
        ax_r2.set_ylabel("$R^2$ Score")
        ax_r2.set_ylim(-0.1, 1.1)
        ax_r2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        img_name = os.path.join(out_dir, f"bilinear_analysis_pair_{e}_{i}.png")
        plt.savefig(img_name, dpi=200)
        plt.close()
        print(f"  ✅ 可视化已保存: {img_name}")


if __name__ == "__main__":
    main()