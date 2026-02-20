import os
import sys
import time
import argparse
import numpy as np
import h5py
import hdf5plugin

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer

# --- 实验全局配置 ---
TOTAL_DURATION = 600.0
DT = 0.1
BASE_STIM_TIME = 200.0  # 基准刺激时间稍微延后，给复杂的时差留出空间
DELAYS = np.arange(-50.0, 51.0, 5.0)  # dt = t_E - t_I. 负数代表 E 先发，正数代表 I 先发


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def get_pair_idx(u, v, N):
    """数学公式：在 itertools.combinations(range(N), 2) 中，计算对 (u, v) [u < v] 的确切索引"""
    offset = u * N - u - (u * (u - 1)) // 2
    return offset + (v - u - 1)


def extract_strongest_ei_pairs(h5_path, top_k=5):
    """从第一个实验的结果中，精准检索并计算双线性耦合系数，挑选最强 E-I 对"""
    print(f"\n🔍 正在分析大规模配对数据集: {h5_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"找不到数据集 {h5_path}。请先完成第一个大规模配对实验，或使用 --mock_pairs 参数进行冒烟测试。")

    with h5py.File(h5_path, 'r') as f:
        target_v = f.attrs['target_v_mV']
        num_synapses = f.attrs['num_synapses']
        syn_types = f['static_info/synapse_types'][:]
        targets = f['dataset/train/targets']

        print("  1. 正在提取所有单突触 PSP 峰值...")
        # 前 num_synapses 个 trial 是单突触测试
        v_singles = targets[:num_synapses, :, 0]
        delta_v = v_singles - target_v

        peaks = np.zeros(num_synapses)
        # 兴奋性看最大去极化，抑制性看最大超极化
        peaks[syn_types == 1] = np.max(delta_v[syn_types == 1], axis=1)
        peaks[syn_types == -1] = np.min(delta_v[syn_types == -1], axis=1)

        print("  2. 执行启发式剪枝 (|Peak| > Mean)...")
        mean_abs_peak = np.mean(np.abs(peaks))
        strong_E = np.where((syn_types == 1) & (np.abs(peaks) > mean_abs_peak))[0]
        strong_I = np.where((syn_types == -1) & (np.abs(peaks) > mean_abs_peak))[0]
        print(f"     保留了 {len(strong_E)} 个强 E 突触 和 {len(strong_I)} 个强 I 突触。")

        print("  3. 正在检索成对耦合数据并计算双线性系数...")
        results = []
        for e in strong_E:
            for i in strong_I:
                # E 的索引必然小于 I 的索引 (0~638 vs 639~1277)
                trial_idx = num_synapses + get_pair_idx(e, i, num_synapses)

                # 精确读取该特定试次的电压响应
                v_C = targets[trial_idx, :, 0]
                delta_v_C = v_C - target_v

                # 寻找最大绝对偏离点作为耦合峰值
                max_deflection_idx = np.argmax(np.abs(delta_v_C))
                peak_C = delta_v_C[max_deflection_idx]

                # 双线性规则系数公式：(V_C - (V_E + V_I)) / (V_E * V_I)
                coef = (peak_C - (peaks[e] + peaks[i])) / (peaks[e] * peaks[i])
                results.append({'e': e, 'i': i, 'coef': coef})

        # 按系数绝对值排序，取出作用最强的前 K 个对
        results.sort(key=lambda x: abs(x['coef']), reverse=True)
        best_pairs = results[:top_k]

        print(f"  🏆 检索完毕！找到最强的 {top_k} 个 E-I 耦合对:")
        for rank, item in enumerate(best_pairs):
            print(f"     Top {rank + 1}: Syn E={item['e']:4d}, Syn I={item['i']:4d} | Coef = {item['coef']:.4f}")

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
        print("\n⚠️ 启用 Mock 模式：选择 5 个 E-I 突触对进行测试。")
        best_pairs = [(262, 901), (246, 886), (223, 831), (1, 663), (264, 904)]
    else:
        best_pairs = extract_strongest_ei_pairs(args.pairs_h5, top_k=args.top_k)

    # 2. 启动神经元物理环境 (活跃态 -67.7 mV)
    print("\n🚀 正在启动神经元仿真环境...")
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    target_v = -67.7
    warmup_meta = env.warmup(target_v=target_v, anchors_path="configs/bg_anchors.json")
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}
    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)

    # 3. 初始化 HDF5 写入器并手动追加 Condition 组
    writer = HDF5Writer(out_h5_path)
    writer.initialize(full_metadata, total_steps, num_synapses, split="train")

    # 手动往 writer 的 f 对象里追加深度学习所需的 conditions_dt 变量
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
            # 计算绝对时间
            # 规则: dt_val = t_E - t_I
            # 我们固定较晚发生的那个突触在 BASE_STIM_TIME
            if dt_val < 0:
                # E 先发
                t_e = BASE_STIM_TIME + dt_val
                t_i = BASE_STIM_TIME
            else:
                # I 先发
                t_e = BASE_STIM_TIME
                t_i = BASE_STIM_TIME - dt_val

            spike_events = [(e, t_e), (i, t_i)]

            # 仿真与矩阵转换
            v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
            input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

            inputs_buffer.append(input_mat)
            targets_buffer.append(v_trace)
            dt_buffer.append([dt_val])
            count += 1

        # 写入一对的完整扫描数据
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

        # 追加 condition 数据
        curr_size = dset_conditions.shape[0]
        dset_conditions.resize(curr_size + len(dt_buffer), axis=0)
        dset_conditions[curr_size:] = np.array(dt_buffer, dtype=np.float32)
        writer.f.flush()

        print(f"  -> 已完成对 (E={e}, I={i}) 的时序扫描 | 进度: {count}/{total_sims}")

    writer.close()
    print(f"\n🎉 实验圆满完成！耗时 {time.time() - start_time:.2f} 秒。")
    print(f"📁 条件化数据集已保存至: {out_h5_path}")


if __name__ == "__main__":
    main()