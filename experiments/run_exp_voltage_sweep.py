import os
import sys
import time
import numpy as np
import concurrent.futures
import h5py
import hdf5plugin

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.l5pc_env import L5PC_Env
from core.hdf5_writer import HDF5Writer

# --- 实验全局配置 ---
TOTAL_DURATION = 600.0
DT = 0.1
STIM_TIME = 100.0
# 135个电压点: -81.1 到 -67.7 (使用 round 防止浮点数精度问题)
VOLTAGES = np.round(np.arange(-81.1, -67.6, 0.1), 1)
BASE_OUT_DIR = "results/voltage_sweep"


def build_dense_input_matrix(spike_events, total_steps, num_synapses, dt):
    """将稀疏脉冲事件转换为密集二进制张量"""
    input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)
    for syn_idx, t in spike_events:
        step = int(t / dt)
        if 0 <= step < total_steps:
            input_matrix[step, syn_idx] = 1
    return input_matrix


def worker_simulate_voltage(target_v):
    """【子进程核心】：处理单一电压的完整仿真流程（仅 warmup 一次，跑 1278 个突触）"""
    print(f"[Worker V={target_v}] 开始分配内存并进行背景电导预热...")
    out_dir = os.path.join(BASE_OUT_DIR, f"v_{target_v}")
    os.makedirs(out_dir, exist_ok=True)
    h5_filepath = os.path.join(out_dir, "L5PC.h5")

    # 断点续传：如果该电压已经跑完，直接返回路径
    if os.path.exists(h5_filepath):
        return h5_filepath

    env = L5PC_Env(
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        biophys_file="L5PC_NEURON_simulation/L5PCbiophys5b.hoc",
        template_file="L5PC_NEURON_simulation/L5PCtemplate_2.hoc",
        dt=DT
    )

    # 精确设定当前 Worker 的唯一背景稳态
    warmup_meta = env.warmup(target_v=target_v, anchors_path="configs/bg_anchors.json")
    topo_meta = env.get_topology_metadata()
    full_metadata = {**topo_meta, **warmup_meta}
    num_synapses = topo_meta["num_synapses"]
    total_steps = int(TOTAL_DURATION / DT)

    # 生成绝对符合单文件规范的 H5
    writer = HDF5Writer(h5_filepath)
    writer.initialize(full_metadata, total_steps, num_synapses, split="train")

    inputs_buffer, targets_buffer = [], []

    # 遍历测定 1278 个单突触
    for syn_idx in range(num_synapses):
        spike_events = [(syn_idx, STIM_TIME)]
        v_trace = env.run_simulation(spike_events, total_duration=TOTAL_DURATION)
        input_mat = build_dense_input_matrix(spike_events, total_steps, num_synapses, DT)

        inputs_buffer.append(input_mat)
        targets_buffer.append(v_trace)

        if len(inputs_buffer) >= 200:
            writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))
            inputs_buffer, targets_buffer = [], []

    if inputs_buffer:
        writer.append(np.stack(inputs_buffer, axis=0), np.stack(targets_buffer, axis=0))

    writer.close()
    print(f"[Worker V={target_v}] ✅ 单电压实验完成! 数据已落盘至 {h5_filepath}")
    return h5_filepath


def merge_to_conditional_dataset(h5_files, final_path):
    """【整合逻辑】：将分散的135个电压文件，融合成带有条件变量的终极 DL 数据集"""
    print(f"\n🔄 正在将 {len(h5_files)} 个独立电压文件融合成深度学习条件化数据集...")

    # 从第一个文件中提取公共静态拓扑
    with h5py.File(h5_files[0], 'r') as f_ref:
        static_info = {k: v[()] for k, v in f_ref['static_info'].items()}
        attrs = dict(f_ref.attrs)
        # 移除静态的 target_v_mV，因为接下来它将变成动态特征
        if 'target_v_mV' in attrs:
            del attrs['target_v_mV']
        target_time_steps = attrs['total_steps']
        num_synapses = attrs['num_synapses']

    total_samples = len(h5_files) * num_synapses  # 135 * 1278 = 172,530 个试次
    comp_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

    with h5py.File(final_path, 'w') as f_out:
        for k, v in attrs.items():
            f_out.attrs[k] = v
        f_out.attrs['description'] = "Voltage Sweep PSP Dataset (-81.1mV to -67.7mV)"

        g_static = f_out.create_group('static_info')
        for k, v in static_info.items():
            g_static.create_dataset(k, data=v)

        g_train = f_out.create_group('dataset/train')
        dset_inputs = g_train.create_dataset('inputs', shape=(total_samples, target_time_steps, num_synapses),
                                             dtype='uint8', chunks=(1, target_time_steps, num_synapses), **comp_kwargs)
        dset_targets = g_train.create_dataset('targets', shape=(total_samples, target_time_steps, 1), dtype='float32',
                                              chunks=(1, target_time_steps, 1), **comp_kwargs)

        # 【核心改进】：增加条件变量数据集 (shape: N x 1)
        dset_conditions = g_train.create_dataset('conditions_v', shape=(total_samples, 1), dtype='float32',
                                                 chunks=(num_synapses, 1), **comp_kwargs)

        global_idx = 0
        for fpath in h5_files:
            with h5py.File(fpath, 'r') as f_src:
                v_val = f_src.attrs['target_v_mV']
                src_inputs = f_src['dataset/train/inputs'][:]
                src_targets = f_src['dataset/train/targets'][:]
                n_file = src_inputs.shape[0]

                end_idx = global_idx + n_file
                dset_inputs[global_idx:end_idx] = src_inputs
                dset_targets[global_idx:end_idx] = src_targets

                # 为这 1278 个 trial 打上当前的电压条件标签
                dset_conditions[global_idx:end_idx] = np.full((n_file, 1), v_val, dtype=np.float32)

                global_idx = end_idx
            print(f"  合并完成: {os.path.basename(os.path.dirname(fpath))} (V={v_val} mV)")

    print(f"✅ 大功告成！跨电压条件化数据集已完美生成: {final_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Voltage Sweep Orchestrator")
    parser.add_argument("--workers", type=int, default=10, help="最多同时拉起几个电压的仿真进程 (注意内存占用)")
    args = parser.parse_args()

    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    h5_file_list = []

    print(f"🚀 开始大规模电压扫参实验 (共 {len(VOLTAGES)} 个目标电压点)...")

    # 1. 以电压为粒度进行并发，最大化避免 brentq 的重复计算损耗
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_simulate_voltage, v): v for v in VOLTAGES}
        for future in concurrent.futures.as_completed(futures):
            try:
                h5_file_list.append(future.result())
            except Exception as exc:
                print(f"❌ Worker error for voltage {futures[future]}: {exc}")

    # 按电压升序排列文件路径，确保写入时的顺序物理逻辑一致
    h5_file_list = sorted(h5_file_list, key=lambda x: float(x.split('v_')[-1].split('/')[0]))

    # 2. 自动融合成深度学习规格的 Conditional Dataset
    final_h5_path = os.path.join(BASE_OUT_DIR, "L5PC_VoltageSweep_Conditional.h5")
    if h5_file_list:
        merge_to_conditional_dataset(h5_file_list, final_h5_path)


if __name__ == "__main__":
    main()