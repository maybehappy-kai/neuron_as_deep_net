import os
import sys
import time
import argparse
import itertools
import numpy as np

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer

# --- 实验全局配置 ---
TOTAL_DURATION = 600.0  # 单次仿真总时长 (ms)，贴合你的 600ms 需求
DT = 0.1  # 时间步长
STIM_TIME = 100.0  # 统一的刺激发生时刻 (ms)，留出 100ms 基线用于分析 PSP


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    """将稀疏事件列表转化为用于存储的密集二进制张量"""
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def main():
    parser = argparse.ArgumentParser(description="Large-scale Pairwise PSP Simulation")
    parser.add_argument("--job_id", type=int, default=0, help="Worker ID")
    parser.add_argument("--total_jobs", type=int, default=1, help="Total Workers")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    # 默认值设为你要求的 -67.7 活跃态
    parser.add_argument("--target_v", type=float, default=-67.7, help="目标背景稳态电压")
    args = parser.parse_args()

    h5_filepath = os.path.join(args.out_dir, f"part_{args.job_id}.h5")
    meta_filepath = os.path.join(args.out_dir, "meta.h5")

    print(f"[Worker {args.job_id}] Booting Environment (Target V: {args.target_v} mV)...")

    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    # 注意：新的默认绝对静息态已更新为 -81.1 mV
    # 只有当目标电压不等于绝对静息态时，才需要读取锚点文件注入背景电导
    anchors_path = "configs/bg_anchors.json" if args.target_v != -81.1 else None

    warmup_meta = env.warmup(target_v=args.target_v if args.target_v != -81.1 else None,
                             anchors_path=anchors_path)

    # 修正：如果在无干涉静息态，手动覆盖 meta 中的基准值为 -81.1
    if args.target_v == -81.1:
        warmup_meta["target_v_mV"] = -81.1

    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}
    num_synapses = topo_meta["num_synapses"]

    # 严格对齐步数，舍弃最后一个点以满足深度学习张量对齐规范
    total_steps = int(TOTAL_DURATION / DT)

    # 初始化写入器
    writer = HDF5Writer(h5_filepath)
    writer.initialize(full_metadata, total_steps, num_synapses)

    if args.job_id == 0 and not os.path.exists(meta_filepath):
        meta_writer = HDF5Writer(meta_filepath)
        meta_writer.initialize(full_metadata, total_steps, num_synapses)
        meta_writer.close()

    # --- 生成实验任务池 ---
    # 1. 单个突触刺激 (1278 个)
    single_tasks = [[(i, STIM_TIME)] for i in range(num_synapses)]

    # 2. 所有可能的双突触同时刺激配对 (C(1278, 2) = 816,003 个)
    pair_tasks = [[(pair[0], STIM_TIME), (pair[1], STIM_TIME)]
                  for pair in itertools.combinations(range(num_synapses), 2)]

    all_tasks = single_tasks + pair_tasks
    total_tasks = len(all_tasks)

    if args.job_id == 0:
        print(f"[Master Info] Total tasks generated: {total_tasks} "
              f"({len(single_tasks)} singles, {len(pair_tasks)} pairs)")

    # 均匀切分任务给各个 Worker
    chunk_size = int(np.ceil(total_tasks / args.total_jobs))
    start_idx = args.job_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_tasks)
    my_tasks = all_tasks[start_idx:end_idx]

    print(f"[Worker {args.job_id}] Assigned tasks {start_idx} to {end_idx - 1} ({len(my_tasks)} tasks).")

    # --- 执行仿真并按批次落盘 ---
    BATCH_SIZE = 100  # 每 100 次 trial 落盘一次，平衡内存与 I/O 速度
    inputs_buffer, targets_buffer = [], []

    start_time = time.time()
    last_report_time = start_time

    for idx, spike_events in enumerate(my_tasks):
        v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
        input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

        inputs_buffer.append(input_mat)
        targets_buffer.append(v_trace)

        if len(inputs_buffer) >= BATCH_SIZE:
            writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))
            inputs_buffer, targets_buffer = [], []

            # 简易进度汇报
            if idx > 0 and idx % 2000 == 0:
                elapsed = time.time() - last_report_time
                rate = 2000 / elapsed if elapsed > 0 else 0
                print(f"[Worker {args.job_id}] Progress: {idx}/{len(my_tasks)} | Speed: {rate:.1f} sims/s")
                last_report_time = time.time()

    # 清空尾部缓存
    if inputs_buffer:
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

    writer.close()
    total_time = time.time() - start_time
    print(f"[Worker {args.job_id}] Finished {len(my_tasks)} tasks in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()