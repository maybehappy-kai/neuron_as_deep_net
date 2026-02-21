import os
import sys
import time
import argparse
import numpy as np
import h5py
import hdf5plugin
from scipy.stats import linregress

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer
from core.config import ACTIVE_TARGET_V

# --- 实验全局配置 ---
TOTAL_DURATION = 600.0
DT = 0.1
BASE_STIM_TIME = 200.0  # 耦合实验中由于可能向前偏移，基准设在 200ms
DELAYS = np.arange(-50.0, 51.0, 5.0)


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def get_pair_idx(u, v, N):
    """数学公式计算组合索引"""
    offset = u * N - u - (u * (u - 1)) // 2
    return offset + (v - u - 1)


def extract_strongest_ei_pairs(h5_path, top_k=5):
    """使用点对点线性回归 (linregress) 严谨提取具有最强双线性系数 (β) 的配对"""
    print(f"\n🔍 正在分析大规模配对数据集: {h5_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"找不到数据集 {h5_path}。")

    with h5py.File(h5_path, 'r') as f:
        target_v = f.attrs['target_v_mV']
        dt_step = f.attrs['dt']
        num_synapses = f.attrs['num_synapses']
        syn_types = f['static_info/synapse_types'][:]
        targets = f['dataset/train/targets']

        # 第一个实验的原始刺激时间是 100.0 ms
        ORIGINAL_STIM_TIME = 100.0
        stim_step = int(ORIGINAL_STIM_TIME / dt_step)

        # 回归窗口：刺激前 50ms 到 刺激后 250ms
        valid_start = int(max(0, (ORIGINAL_STIM_TIME - 50) / dt_step))
        valid_end = int((ORIGINAL_STIM_TIME + 250) / dt_step)

        print("  1. 正在提取所有单突触 PSP 波形并校准至 target_v ...")
        v_singles = targets[:num_synapses, :, 0]
        delta_v_singles = v_singles - target_v

        # 仍然使用峰值进行剪枝，这是合理的（去除根本不产生明显波动的突触）
        window_delta_v = delta_v_singles[:, stim_step:valid_end]
        peaks = np.zeros(num_synapses)
        peaks[syn_types == 1] = np.max(window_delta_v[syn_types == 1], axis=1)
        peaks[syn_types == -1] = np.min(window_delta_v[syn_types == -1], axis=1)

        print("  2. 执行启发式剪枝 (|Peak| > Mean)...")
        mean_abs_peak = np.mean(np.abs(peaks))
        strong_E = np.where((syn_types == 1) & (np.abs(peaks) > mean_abs_peak))[0]
        strong_I = np.where((syn_types == -1) & (np.abs(peaks) > mean_abs_peak))[0]
        print(f"     保留了 {len(strong_E)} 个强 E 突触 和 {len(strong_I)} 个强 I 突触。")

        print("  3. 正在检索成对耦合数据并执行点对点线性回归 (Linregress)...")
        results = []
        total_evals = len(strong_E) * len(strong_I)
        count = 0

        for e in strong_E:
            delta_e = delta_v_singles[e]
            for i in strong_I:
                trial_idx = num_synapses + get_pair_idx(e, i, num_synapses)

                v_C = targets[trial_idx, :, 0]
                delta_c = v_C - target_v
                delta_i = delta_v_singles[i]

                # 核心非线性计算逻辑
                nl = delta_c - (delta_e + delta_i)
                prod = delta_e * delta_i

                nl_window = nl[valid_start:valid_end]
                prod_window = prod[valid_start:valid_end]

                # 保护机制：如果两者的乘积几乎不变（全为0），防止回归失败
                if np.std(prod_window) < 1e-9:
                    coef = 0.0
                else:
                    slope, _, _, _, _ = linregress(prod_window, nl_window)
                    coef = slope  # 斜率就是我们梦寐以求的双线性系数 β

                results.append({'e': e, 'i': i, 'coef': coef})

                count += 1
                if count % 1000 == 0:
                    print(f"     已分析 {count}/{total_evals} 对...")

        # 排序：挑出绝对值最大的 β (无论超线性放大还是强力分流抑制)
        results.sort(key=lambda x: abs(x['coef']), reverse=True)
        best_pairs = results[:top_k]

        print(f"  🏆 检索完毕！找到最强的 {top_k} 个 E-I 耦合对:")
        for rank, item in enumerate(best_pairs):
            print(f"     Top {rank + 1}: Syn E={item['e']:4d}, Syn I={item['i']:4d} | Coef (β) = {item['coef']:.4f}")

        return [(item['e'], item['i']) for item in best_pairs]


def main():
    parser = argparse.ArgumentParser(description="Temporal Coupling Experiment for Strongest E-I Pairs")
    parser.add_argument("--pairs_h5", type=str, default="results/exp_active_pairs/final_dataset.h5",
                        help="第一个实验生成的 HDF5 文件路径")
    parser.add_argument("--out_dir", type=str, default="results/temporal_coupling", help="输出目录")
    parser.add_argument("--top_k", type=int, default=10, help="挑选最强的多少个对进行时域扫描")
    parser.add_argument("--mock_pairs", action="store_true", help="如果没有大规模数据，使用此项随机生成几对用于冒烟测试")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_h5_path = os.path.join(args.out_dir, "L5PC_TemporalCoupling.h5")

    # 1. 获取目标对
    if args.mock_pairs:
        print("\n⚠️ 启用 Mock 模式：随机选择 5 个 E-I 突触对进行测试。")
        best_pairs = [(1, 663), (223, 831), (246, 886), (262, 901), (264, 904)]
    else:
        best_pairs = extract_strongest_ei_pairs(args.pairs_h5, top_k=args.top_k)

    # 2. 启动神经元物理环境
    print("\n🚀 正在启动神经元仿真环境...")
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    target_v = ACTIVE_TARGET_V
    warmup_meta = env.warmup(target_v=target_v, anchors_path="configs/bg_anchors.json")
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}
    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)

    # 3. 初始化 HDF5 写入器并手动追加 Condition 组
    writer = HDF5Writer(out_h5_path)
    writer.initialize(full_metadata, total_steps, num_synapses, split="train")

    comp_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    if 'conditions_dt' not in writer.f['dataset/train']:
        writer.f['dataset/train'].create_dataset(
            'conditions_dt', shape=(0, 1), maxshape=(None, 1), dtype='float32', chunks=(1, 1), **comp_kwargs
        )
    dset_conditions = writer.f['dataset/train/conditions_dt']

    # 4. 执行时序扫描
    print(f"\n⏳ 开始执行时序扫描 (共 {len(best_pairs)} 对, 每对 {len(DELAYS)} 个时间点)...")

    total_sims = len(best_pairs) * len(DELAYS)
    count = 0
    start_time = time.time()

    for e, i in best_pairs:
        inputs_buffer, targets_buffer, dt_buffer = [], [], []

        for dt_val in DELAYS:
            if dt_val < 0:
                t_e, t_i = BASE_STIM_TIME + dt_val, BASE_STIM_TIME
            else:
                t_e, t_i = BASE_STIM_TIME, BASE_STIM_TIME - dt_val

            spike_events = [(e, t_e), (i, t_i)]

            v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
            input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

            inputs_buffer.append(input_mat)
            targets_buffer.append(v_trace)
            dt_buffer.append([dt_val])
            count += 1

        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

        curr_size = dset_conditions.shape[0]
        dset_conditions.resize(curr_size + len(dt_buffer), axis=0)
        dset_conditions[curr_size:] = np.array(dt_buffer, dtype=np.float32)
        writer.f.flush()

        print(f"  -> 已完成对 (E={e}, I={i}) 的时序扫描 | 进度: {count}/{total_sims}")

    writer.close()
    print(f"\n🎉 实验圆满完成！耗时 {time.time() - start_time:.2f} 秒。")


if __name__ == "__main__":
    main()