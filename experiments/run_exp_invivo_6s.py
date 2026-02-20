import os
import sys
import time
import math
import argparse
import numpy as np

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer

# --- 实验全局配置 ---
TOTAL_DURATION = 6000.0  # 6秒长程仿真 (ms)
DT = 0.1  # 保持 0.1ms 的高精度积分
RATE_E = 5.0  # 兴奋性泊松发放率 (Hz)
RATE_I = 5.0  # 抑制性泊松发放率 (Hz) - 经过 E/I 数量配比修正


def generate_poisson_spikes(rate_hz, duration_ms, rng):
    """生成泊松分布的脉冲时间序列"""
    rate_ms = rate_hz / 1000.0
    spikes = []
    t = 0.0
    while True:
        # 指数分布的间隔时间 (ISI)
        isi = -math.log(1.0 - rng.rand()) / rate_ms
        t += isi
        if t >= duration_ms:
            break
        spikes.append(t)
    return spikes


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    """将极度密集的脉冲事件转化为 uint8 张量"""
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def main():
    parser = argparse.ArgumentParser(description="6-second In-vivo Poisson Bombardment Simulation")
    parser.add_argument("--job_id", type=int, default=0, help="Worker ID")
    parser.add_argument("--total_jobs", type=int, default=1, help="Total Workers")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_trials", type=int, default=2000, help="总共要生成多少个 6 秒片段")
    parser.add_argument("--batch_size", type=int, default=10, help="由于单 trial 数据量极大，需减小落盘批次")
    args = parser.parse_args()

    h5_filepath = os.path.join(args.out_dir, f"part_{args.job_id}.h5")
    meta_filepath = os.path.join(args.out_dir, "meta.h5")

    # 1. 启动神经元物理环境 (深度静息态，绝对无背景电导干涉)
    print(f"[Worker {args.job_id}] Initializing deep rest environment (NO artificial target_v)...")
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    # 传入 target_v=None，完全凭借突触轰炸抬高电压
    warmup_meta = env.warmup(target_v=None)
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}

    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)  # 60000 步
    syn_types = topo_meta["synapse_types"]

    # 2. 初始化 HDF5 写入器
    writer = HDF5Writer(h5_filepath)
    writer.initialize(full_metadata, total_steps, num_synapses, split="train")

    if args.job_id == 0 and not os.path.exists(meta_filepath):
        meta_writer = HDF5Writer(meta_filepath)
        meta_writer.initialize(full_metadata, total_steps, num_synapses, split="train")
        meta_writer.close()

    # 3. 任务分配
    my_trials = range(args.job_id, args.num_trials, args.total_jobs)
    print(f"[Worker {args.job_id}] Allocated {len(my_trials)} 6-second in-vivo trials.")

    # 4. 执行 In-vivo 轰炸主循环
    inputs_buffer = []
    targets_buffer = []
    start_time = time.time()

    # 为每个 worker 设置独立的随机数生成器，确保泊松噪声绝对不重复
    rng = np.random.RandomState(42 + args.job_id)

    for i, trial_idx in enumerate(my_trials):
        spike_events = []

        # 为每个突触独立生成 6 秒的泊松脉冲列
        for syn_idx in range(num_synapses):
            rate = RATE_E if syn_types[syn_idx] == 1 else RATE_I
            spikes = generate_poisson_spikes(rate, TOTAL_DURATION, rng)
            for t in spikes:
                spike_events.append((syn_idx, t))

        # 6秒内约有近万个脉冲，交给环境进行物理积分
        v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
        input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

        inputs_buffer.append(input_mat)
        targets_buffer.append(v_trace)

        # 批次落盘 (60000 * 1278 的矩阵极其庞大，10个 trial 就会占用不少内存，必须高频 flush)
        if len(inputs_buffer) >= args.batch_size:
            writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))
            inputs_buffer, targets_buffer = [], []

            elapsed = time.time() - start_time
            print(f"[Worker {args.job_id}] Progress: {i + 1}/{len(my_trials)} | Time elapsed: {elapsed:.1f}s")

    if inputs_buffer:
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

    writer.close()
    print(f"[Worker {args.job_id}] 🎉 In-vivo data generation completed!")


if __name__ == "__main__":
    main()