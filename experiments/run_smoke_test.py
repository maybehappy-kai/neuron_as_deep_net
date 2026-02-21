import os
import sys
import time
import argparse
import numpy as np

# 将项目根目录加入 sys.path，以便导入 core 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer
from core.config import ACTIVE_TARGET_V
import core.config as cfg

# --- 实验全局配置 ---
TOTAL_DURATION = 500.0  # 每次仿真的总时长 (ms)
DT = 0.1  # 时间步长


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    """辅助函数：将稀疏的脉冲事件列表转化为密集的二进制张量，用于 HDF5 存储"""
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def main():
    parser = argparse.ArgumentParser(description="Smoke Test for L5PC Pipeline")
    parser.add_argument("--job_id", type=int, default=0, help="Worker ID")
    parser.add_argument("--total_jobs", type=int, default=1, help="Total Workers")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--state", type=str, choices=['rest', 'active'], default='rest',
                        help="测试的状态: rest (无干涉深度静息) 或 active (使用 config 中的活跃态)")
    args = parser.parse_args()

    h5_filepath = os.path.join(args.out_dir, f"part_{args.job_id}.h5")
    meta_filepath = os.path.join(args.out_dir, "meta.h5")

    # 根据传入的标签，自动去 config 里拿值
    if args.state == 'active':
        target_v = cfg.ACTIVE_TARGET_V
        anchors_path = "configs/bg_anchors.json"
    else:
        target_v = cfg.REST_TARGET_V  # 这里即为 None
        anchors_path = None

    print(f"[Worker {args.job_id}] Initializing Environment (State: {args.state.upper()}, Target V: {target_v})...")

    # 1. 启动环境
    # 注意：请确保 morphologies 和 mods 等相关文件在 L5PC_NEURON_simulation/ 下
    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    # 预热并获取物理状态元数据
    # 只要 target_v 是 None，就不传 anchors_path
    anchors_path = "configs/bg_anchors.json" if target_v is not None else None
    warmup_meta = env.warmup(target_v=target_v, anchors_path=anchors_path)
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}
    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)

    # 2. 启动写入器
    writer = HDF5Writer(h5_filepath)
    writer.initialize(full_metadata, total_steps, num_synapses)

    # 仅由 Job 0 负责写入 meta.h5，供最后合并使用
    if args.job_id == 0 and not os.path.exists(meta_filepath):
        meta_writer = HDF5Writer(meta_filepath)
        meta_writer.initialize(full_metadata, total_steps, num_synapses)
        meta_writer.close()

    # 3. 构建极度多样的测试范式 (Smoke Test Core)
    # 为了测试方便，我们挑前 2 个兴奋性突触和前 2 个抑制性突触
    ex_syns = np.where(env.synapse_types == 1)[0][:2]
    in_syns = np.where(env.synapse_types == -1)[0][:2]

    all_test_cases = []

    # 范式 A: 单个刺激 (测试基础兴奋和抑制)
    for syn in ex_syns: all_test_cases.append([(syn, 100.0)])
    for syn in in_syns: all_test_cases.append([(syn, 100.0)])

    # 范式 B: 同时刺激配对 (测试非线性空间叠加)
    all_test_cases.append([(ex_syns[0], 100.0), (ex_syns[1], 100.0)])  # E + E
    all_test_cases.append([(ex_syns[0], 100.0), (in_syns[0], 100.0)])  # E + I

    # 范式 C: 时间差耦合刺激 (测试时间积分特性)
    all_test_cases.append([(ex_syns[0], 100.0), (ex_syns[1], 120.0)])  # 相差 20ms
    all_test_cases.append([(ex_syns[0], 100.0), (in_syns[0], 150.0)])  # 相差 50ms

    # 范式 D: In-vivo 随机泊松噪声 (测试长时间抗压能力)
    np.random.seed(42 + args.job_id)  # 确保每个 case 不同
    random_events = []
    test_pool = list(ex_syns) + list(in_syns)
    for syn in test_pool:
        # 在 50ms 到 450ms 之间随机生成 4 个脉冲
        spikes = np.random.uniform(50, 450, size=4)
        for t in spikes:
            random_events.append((syn, t))
    all_test_cases.append(random_events)

    # 4. 任务分发 (让多进程把这些 test case 瓜分掉)
    my_cases = all_test_cases[args.job_id:: args.total_jobs]
    print(f"[Worker {args.job_id}] Assigned {len(my_cases)} test cases.")

    # 5. 执行仿真并落盘
    inputs_buffer = []
    targets_buffer = []

    start_time = time.time()
    for idx, spike_events in enumerate(my_cases):
        # 核心：纯粹的环境调用
        v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
        input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

        inputs_buffer.append(input_mat)
        targets_buffer.append(v_trace)

        # 批次写入 (冒烟测试数据少，满 2 个就写入测试一下 flush 机制)
        if len(inputs_buffer) >= 2:
            writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))
            inputs_buffer, targets_buffer = [], []

    # 清空剩余缓存
    if inputs_buffer:
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

    writer.close()
    print(f"[Worker {args.job_id}] Done in {time.time() - start_time:.2f}s!")


if __name__ == "__main__":
    main()