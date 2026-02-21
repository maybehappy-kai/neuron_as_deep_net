import os
import sys
import time
import math
import argparse
import itertools
import numpy as np

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer
from core.config import ACTIVE_TARGET_V

# --- 实验全局配置 ---
TOTAL_DURATION = 600.0  # 每次仿真的总时长 (ms)
DT = 0.1  # 时间步长
STIM_TIME = 100.0  # 刺激发生的绝对时间 (ms)，留出 100ms 观察基线平稳度


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    """将稀疏的脉冲事件转化为密集二进制张量，用于 HDF5 存储"""
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def main():
    parser = argparse.ArgumentParser(description="Large-scale PSP Pairs Simulation (Active State)")
    parser.add_argument("--job_id", type=int, default=0, help="Worker ID")
    parser.add_argument("--total_jobs", type=int, default=1, help="Total Workers")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--target_v", type=float, default=ACTIVE_TARGET_V, help="活跃态背景电压")
    parser.add_argument("--batch_size", type=int, default=100, help="每积累多少个 trial 写入一次硬盘")
    args = parser.parse_args()

    h5_filepath = os.path.join(args.out_dir, f"part_{args.job_id}.h5")
    meta_filepath = os.path.join(args.out_dir, "meta.h5")

    # 1. 初始化神经元环境
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    # 预热并读取状态。注意：这里强制设定为活跃态
    warmup_meta = env.warmup(target_v=args.target_v, anchors_path="configs/bg_anchors.json")
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}

    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)  # 严格对齐步长，不加 1

    # 2. 初始化 HDF5 写入器 (写入 /dataset/train)
    writer = HDF5Writer(h5_filepath)
    writer.initialize(full_metadata, total_steps, num_synapses, split="train")

    # Job 0 负责生成供 Orchestrator 最终合并用的 meta.h5
    if args.job_id == 0 and not os.path.exists(meta_filepath):
        meta_writer = HDF5Writer(meta_filepath)
        meta_writer.initialize(full_metadata, total_steps, num_synapses, split="train")
        meta_writer.close()

    # 3. 构建全排列任务池 (单突触 + 所有两两配对)
    all_syns = list(range(num_synapses))
    singles = [[syn] for syn in all_syns]
    pairs = [list(p) for p in itertools.combinations(all_syns, 2)]

    all_tasks = singles + pairs  # 约 81.7 万个任务
    total_tasks = len(all_tasks)

    # 4. 任务均匀切分
    chunk_size = math.ceil(total_tasks / args.total_jobs)
    start_idx = args.job_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_tasks)
    my_tasks = all_tasks[start_idx:end_idx]

    print(f"[Worker {args.job_id}] Allocated task range {start_idx} to {end_idx} (Total: {len(my_tasks)} trials)")

    # 5. 执行主循环
    inputs_buffer = []
    targets_buffer = []
    start_time = time.time()

    for i, task in enumerate(my_tasks):
        # 将任务转化为刺激事件：[(syn1, 100.0), (syn2, 100.0)]
        spike_events = [(syn, STIM_TIME) for syn in task]

        # 跑仿真 (底层 API 不再关心你有几个突触，照单全收)
        v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
        input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

        inputs_buffer.append(input_mat)
        targets_buffer.append(v_trace)

        # 批次落盘机制
        if len(inputs_buffer) >= args.batch_size:
            writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))
            inputs_buffer, targets_buffer = [], []

        # 进度汇报
        if (i + 1) % (args.batch_size * 5) == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            print(f"[Worker {args.job_id}] Progress: {i + 1}/{len(my_tasks)} | Speed: {speed:.2f} trials/sec")

    # 收尾，将 buffer 中不足 batch_size 的零头写入
    if inputs_buffer:
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

    writer.close()
    print(f"[Worker {args.job_id}] 🎉 All assigned tasks completed in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    main()