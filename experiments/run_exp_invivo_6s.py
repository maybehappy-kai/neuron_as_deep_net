import os
import sys
import math
import time
import argparse
import numpy as np

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer
import core.config as cfg  # 引入统一配置

# --- 实验全局配置 ---
TOTAL_DURATION = 6000.0  # 6秒长程仿真 (ms)
DT = 0.1


def generate_ou_rate_trace(mu, sigma_ratio, tau, duration_ms, dt, rng):
    """生成符合 OU 过程的非齐次发放率迹 (基于你的测算算法)"""
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
    """根据非齐次发放率迹生成脉冲"""
    prob = rate_trace * (dt / 1000.0)
    rand_vals = rng.rand(len(rate_trace))
    spike_indices = np.where(rand_vals < prob)[0]
    return spike_indices * dt


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def main():
    parser = argparse.ArgumentParser(description="6s In-vivo Inhomogeneous Poisson Bombardment")
    parser.add_argument("--job_id", type=int, default=0, help="Worker ID")
    parser.add_argument("--total_jobs", type=int, default=1, help="Total Workers")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_trials", type=int, default=2000, help="总共要生成多少个 6 秒片段")
    parser.add_argument("--batch_size", type=int, default=10, help="单次落盘批次大小")
    args = parser.parse_args()

    h5_filepath = os.path.join(args.out_dir, f"part_{args.job_id}.h5")
    meta_filepath = os.path.join(args.out_dir, "meta.h5")

    print(f"[Worker {args.job_id}] Initializing deep rest environment...")
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    warmup_meta = env.warmup(target_v=None)
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}

    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)
    syn_types = topo_meta["synapse_types"]
    input_map = topo_meta["input_map"]

    # 动态获取树突片段物理长度
    seg_lengths = np.array([seg.sec.L / seg.sec.nseg for seg in env.segments], dtype=np.float32)
    mean_length = np.mean(seg_lengths)

    writer = HDF5Writer(h5_filepath)
    writer.initialize(full_metadata, total_steps, num_synapses, split="train")

    if args.job_id == 0 and not os.path.exists(meta_filepath):
        meta_writer = HDF5Writer(meta_filepath)
        meta_writer.initialize(full_metadata, total_steps, num_synapses, split="train")
        meta_writer.close()

    my_trials = range(args.job_id, args.num_trials, args.total_jobs)
    inputs_buffer, targets_buffer = [], []
    start_time = time.time()

    # 每个 Trial / Worker 使用不同种子，确保数据集多样性
    rng = np.random.RandomState(42 + args.job_id)

    for i, trial_idx in enumerate(my_trials):
        # 1. 为本次 trial 生成统一的集群波动迹 (OU Process)
        trace_e = generate_ou_rate_trace(cfg.INVIVO_MU_E, cfg.INVIVO_SIGMA_RATIO, cfg.INVIVO_TAU, TOTAL_DURATION, DT,
                                         rng)
        trace_i = generate_ou_rate_trace(cfg.INVIVO_MU_I, cfg.INVIVO_SIGMA_RATIO, cfg.INVIVO_TAU, TOTAL_DURATION, DT,
                                         rng)

        spike_events = []
        for syn_idx in range(num_synapses):
            seg_idx = input_map[syn_idx]
            length_factor = seg_lengths[seg_idx] / mean_length

            base_trace = trace_e if syn_types[syn_idx] == 1 else trace_i
            scaled_trace = base_trace * length_factor

            spikes = generate_inhomogeneous_spikes(scaled_trace, DT, rng)
            for t in spikes:
                spike_events.append((syn_idx, t))

        # 物理积分
        v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
        input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

        inputs_buffer.append(input_mat)
        targets_buffer.append(v_trace)

        if len(inputs_buffer) >= args.batch_size:
            writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))
            inputs_buffer, targets_buffer = [], []
            print(f"[Worker {args.job_id}] Progress: {i + 1}/{len(my_trials)} | Time: {time.time() - start_time:.1f}s")

    if inputs_buffer:
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

    writer.close()


if __name__ == "__main__":
    main()