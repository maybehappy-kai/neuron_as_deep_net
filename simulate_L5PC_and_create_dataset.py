import os
os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'
import sys
import time
import math
import argparse
import itertools
import numpy as np
import h5py
import hdf5plugin  # 新增
import neuron
from neuron import h
import json
from scipy.optimize import brentq


# ==========================================
# 1. 全局配置 (Configuration) - 增加动态文件名支持
# ==========================================
class Config:
    # 仿真参数
    DT = 0.1
    T_SETTLE = 100.0  # 稳定时间
    T_RECORD = 500.0  # 记录时间 (有效数据)
    STIM_DELAY = 100.0  # 刺激发生时刻 (相对于 T_RECORD 开始)

    # 电压初始值
    V_INIT = -80.0
    CELSIUS = 34.0

    # 突触参数
    SYN_GMAX_AMPA = 0.0004
    SYN_GMAX_NMDA = 0.0004
    SYN_GMAX_GABA = 0.001

    # 文件路径
    MORPHOLOGY_FILE = "L5PC_NEURON_simulation/morphologies/cell1.asc"
    BIOPHYS_FILE = "L5PC_NEURON_simulation/L5PCbiophys5b.hoc"
    TEMPLATE_FILE = "L5PC_NEURON_simulation/L5PCtemplate_2.hoc"

    # 输出文件名将在 main 中动态生成
    OUTPUT_FILE = "L5PC_output.h5"


# ==========================================
# 2. 神经元模型包装器 (The Model Wrapper)
# ==========================================
class L5PC_Model:
    def __init__(self, config):
        self.cfg = config
        self.cell = None
        self.synapses = []
        self.segments = []
        self.segment_info = {}
        self.savestate = None  # 新增：用于存储稳态

        self._init_neuron()
        self._setup_model_and_synapses()

        # 初始化完成后，自动执行一次预热
        self._warmup()

    def _init_neuron(self):
        h.load_file('nrngui.hoc')
        h.load_file("import3d.hoc")
        h.load_file(self.cfg.BIOPHYS_FILE)
        h.load_file(self.cfg.TEMPLATE_FILE)

        h.celsius = self.cfg.CELSIUS
        h.dt = self.cfg.DT

        self.cell = h.L5PCtemplate(self.cfg.MORPHOLOGY_FILE)
        h.cvode_active(0)

        # 核心逻辑分发：是否启用活跃态缩放
        if getattr(self.cfg, 'TARGET_V', None) is not None:
            self._apply_scaled_background(self.cfg.TARGET_V)
        else:
            # 完全不干涉，设置 V_INIT 为其自然静息态（通常在 -80 左右）
            self.cfg.V_INIT = -80.0
            print("Background state: 0 interference (Deep Rest). V_INIT = -80.0 mV")

    def _apply_scaled_background(self, target_v):
        """读取锚点并使用 Brent 方法极速寻找匹配 target_v 的缩放系数 alpha"""
        try:
            with open("bg_anchors.json", "r") as f:
                anchors = json.load(f)
            ge_max = anchors["g_exc_max"]
            gi_max = anchors["g_inh_max"]
        except FileNotFoundError:
            raise RuntimeError("bg_anchors.json not found! Please run find_bg_conductance.py first.")

        orig_pas = {}
        for sec in self.cell.all:
            for seg in sec:
                if hasattr(seg, 'pas'):
                    orig_pas[seg] = {'g': seg.g_pas, 'e': seg.e_pas}

        def apply_alpha(alpha):
            ge = alpha * ge_max
            gi = alpha * gi_max
            for sec in self.cell.all:
                for seg in sec:
                    if hasattr(seg, 'pas'):
                        g_old = orig_pas[seg]['g']
                        e_old = orig_pas[seg]['e']
                        g_new = g_old + ge + gi
                        e_new = (g_old * e_old + ge * 0.0 + gi * (-80.0)) / g_new
                        seg.g_pas = g_new
                        seg.e_pas = e_new

        # 核心改造：定义一个误差函数 (返回当前电压与目标的差值)
        def voltage_error(alpha):
            apply_alpha(alpha)
            h.finitialize(target_v)
            h.continuerun(2000.0)  # 跑一小段看稳态去向
            current_v = self.cell.soma[0](0.5).v
            print(f"  Testing alpha = {alpha:.4f} -> V_rest = {current_v:.3f} mV")
            return current_v - target_v

        print(f"Calibrating alpha for Target V = {target_v} mV using Brent's method...")

        # 极速求解：brentq 会自动在 [0.0, 1.0] 之间寻找让 voltage_error 为 0 的根
        # 只要保证 f(0) 和 f(1) 符号相反即可 (即 target_v 必须在静息和最大活跃电压之间)
        try:
            final_alpha = brentq(voltage_error, 0.0, 1.0, xtol=1e-4)
        except ValueError:
            raise ValueError(f"Target voltage {target_v} is out of achievable bounds [~ -81mV, ~ -67.7mV]")

        # 应用最终算出的 alpha
        apply_alpha(final_alpha)

        # 更新全局 V_INIT 使得之后的 Warmup 极快
        self.cfg.V_INIT = target_v
        print(f"🎯 Calibration successful! Found alpha = {final_alpha:.4f}. Steady state set to {target_v} mV")

    def _setup_model_and_synapses(self):
        """
        1. 遍历所有 segment (包括 Soma, Basal, Apical)
        2. 记录拓扑信息 (parent_indices) - 使用基于名称的稳健查找
        3. 在 Dendrites 上创建突触 (排除 Soma)
        """
        self.segments = []
        self.synapses = []

        # 1. 收集所有 Section
        # L5PCtemplate 提供的引用
        all_sections = []
        soma_sections = list(self.cell.soma)
        dend_sections = list(self.cell.dend)
        apic_sections = list(self.cell.apic)

        # 顺序：Soma -> Basal -> Apical
        all_sections = soma_sections + dend_sections + apic_sections

        # 2. 展平为 Segments 并建立索引
        # 我们需要一个稳健的查找表：{section_name: [global_seg_indices]}
        sec_name_to_indices = {}

        for sec in all_sections:
            sec_name = sec.name()
            sec_name_to_indices[sec_name] = []

            for seg in sec:
                global_idx = len(self.segments)
                self.segments.append(seg)
                sec_name_to_indices[sec_name].append(global_idx)

        print(f"Total segments collected: {len(self.segments)}")

        # 3. 构建拓扑结构 (Parent Indices)
        self.parent_indices = np.zeros(len(self.segments), dtype=np.int32)

        for i, seg in enumerate(self.segments):
            sec = seg.sec
            sec_name = sec.name()

            # 检查是否是该 Section 的第一个 Segment
            # 逻辑：比较当前全局索引是否等于该 Section 记录的第一个索引
            is_first_seg = (i == sec_name_to_indices[sec_name][0])

            if not is_first_seg:
                # 内部连接：父节点就是前一个索引
                self.parent_indices[i] = i - 1
            else:
                # Section 头部：寻找 Section 的父级
                parent_seg = sec.parentseg()

                if parent_seg is None:
                    # 真正的物理根节点 (通常是 Soma[0])
                    self.parent_indices[i] = -1
                else:
                    # 查找 Parent Segment 的全局索引
                    # 难点：parent_seg 是一个新对象，不能直接查字典
                    # 解决：通过 Section 名字和 x 坐标 (位置) 来匹配
                    p_sec_name = parent_seg.sec.name()
                    p_x = parent_seg.x

                    if p_sec_name in sec_name_to_indices:
                        # 获取父 Section 包含的所有全局索引
                        candidate_indices = sec_name_to_indices[p_sec_name]

                        # 在这些候选者中，找到位置 x 最接近 p_x 的那个
                        # (NEURON 的 parentseg 通常连接到父 segment 的中心)
                        best_idx = -1
                        min_dist = 1.0

                        for idx in candidate_indices:
                            c_seg = self.segments[idx]
                            dist = abs(c_seg.x - p_x)
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = idx

                        self.parent_indices[i] = best_idx
                    else:
                        # 父节点不在记录的列表里 (比如连到了 Axon，如果我们没包含 Axon)
                        # 在 L5PC 中，Dend/Apic 都连在 Soma，Soma 在列表里，所以这里应该安全
                        print(f"Warning: Parent section {p_sec_name} not found for {sec_name}")
                        self.parent_indices[i] = -1

        # 4. 布置突触 (排除 Soma)
        # 策略：只在 Basal 和 Apical 上放突触
        self.input_map = []
        self.synapse_types = []

        # 辅助集合用于快速判断
        # 注意：这里判断 Section 对象是否属于 soma 列表
        soma_sec_set = set(soma_sections)

        # 定义辅助函数创建突触 (保持你修改后的版本)
        def create_synapse(seg, syn_type):
            syn = None
            if syn_type == 'AMPA_NMDA':
                if hasattr(h, 'ProbAMPANMDA_EMS'):
                    syn = h.ProbAMPANMDA_EMS(seg)
                elif hasattr(h, 'ProbAMPANMDA2'):
                    syn = h.ProbAMPANMDA2(seg)
                if syn is None: raise RuntimeError("Mechanism not found")
                if hasattr(syn, 'tau_r_AMPA'):
                    syn.tau_r_AMPA = 0.3;
                    syn.tau_d_AMPA = 3.0
                    syn.tau_r_NMDA = 2.0;
                    syn.tau_d_NMDA = 70.0
                if hasattr(syn, 'gmax'):
                    syn.gmax = self.cfg.SYN_GMAX_NMDA
                elif hasattr(syn, 'gMax'):
                    syn.gMax = self.cfg.SYN_GMAX_NMDA
            elif syn_type == 'GABA_A':
                if hasattr(h, 'ProbGABAAB_EMS'):
                    syn = h.ProbGABAAB_EMS(seg)
                elif hasattr(h, 'ProbUDFsyn2'):
                    syn = h.ProbUDFsyn2(seg)
                if syn is None: raise RuntimeError("Mechanism not found")
                if hasattr(syn, 'tau_r_GABAA'):
                    syn.tau_r_GABAA = 0.2;
                    syn.tau_d_GABAA = 8.0;
                    syn.e_GABAA = -80.0
                    syn.tau_r_GABAB = 3.5;
                    syn.tau_d_GABAB = 260.9;
                    syn.e_GABAB = -97.0
                    if hasattr(syn, 'GABAB_ratio'): syn.GABAB_ratio = 0.0
                elif hasattr(syn, 'tau_r'):
                    syn.tau_r = 0.2;
                    syn.tau_d = 8.0;
                    syn.e = -80.0
                if hasattr(syn, 'gmax'):
                    syn.gmax = self.cfg.SYN_GMAX_GABA
                elif hasattr(syn, 'gMax'):
                    syn.gMax = self.cfg.SYN_GMAX_GABA
            try:
                syn.Use = 1.0;
                syn.u0 = 0.0;
                syn.Dep = 0.0;
                syn.Fac = 0.0
            except:
                pass
            return syn

        # 预先准备两个临时列表
        ex_syns_temp = []
        inh_syns_temp = []

        for i, seg in enumerate(self.segments):
            if seg.sec in soma_sec_set:
                continue

            # 分别生成突触并附带对应的全局 segment 索引 (i)
            syn_ex = create_synapse(seg, 'AMPA_NMDA')
            ex_syns_temp.append((syn_ex, i, 1))  # (突触对象, seg索引, 类型)

            syn_inh = create_synapse(seg, 'GABA_A')
            inh_syns_temp.append((syn_inh, i, -1))

        # 强制拼接：先兴奋，后抑制！
        all_ordered_syns = ex_syns_temp + inh_syns_temp

        # 注册到正式列表
        for syn, seg_idx, s_type in all_ordered_syns:
            self.synapses.append(syn)
            self.input_map.append(seg_idx)
            self.synapse_types.append(s_type)

        # 转换为 numpy 数组
        self.input_map = np.array(self.input_map, dtype=np.int32)
        self.synapse_types = np.array(self.synapse_types, dtype=np.int8)

        print(f"Model Setup Complete: {len(self.segments)} segments, {len(self.synapses)} synapses.")
        # 验证数量
        expected_synapses = (len(self.segments) - 1) * 2  # 假设 Soma 只有1个 segment
        if len(self.synapses) != expected_synapses:
            # 如果 Soma 不止一个 segment，或者 dend/apic 计数有误，这里会提示
            # 但对于 L5PC，通常 segments=640, soma=1, 预期 639*2 = 1278
            pass

        # 5. 预创建所有 NetCon (关键修改)
        self.netcons = []
        for syn in self.synapses:
            # 使用 None 作为源，表示我们将手动通过 nc.event() 发送脉冲
            nc = h.NetCon(None, syn)
            nc.weight[0] = 1.0
            nc.delay = 0.0
            self.netcons.append(nc)

        print(f"Model Setup Complete: {len(self.segments)} segments, {len(self.synapses)} synapses and NetCons.")

    def get_static_metadata(self):
        """返回用于写入 HDF5 /static_info 的字典"""
        return {
            "num_subunits": len(self.segments),
            "parent_indices": self.parent_indices,
            "input_map": self.input_map,
            "synapse_types": self.synapse_types,
            "dt": self.cfg.DT,
            "total_time": int((self.cfg.T_SETTLE + self.cfg.T_RECORD) / self.cfg.DT)
        }

    def _warmup(self):
        """
        执行一次长时仿真以达到稳态，并保存状态。
        """
        print("Performing warmup to reach steady state...")

        # 1. 初始化
        h.finitialize(self.cfg.V_INIT)

        # 2. 跑足够长的时间 (3000ms 足够让最慢的 Ih 通道平衡)
        h.continuerun(3000.0)

        # 3. 检查当前电压
        v_soma = self.cell.soma[0](0.5).v
        print(f"Warmup complete. Steady state voltage: {v_soma:.4f} mV")

        # 4. 保存状态
        self.savestate = h.SaveState()
        self.savestate.save()
        print("System state saved.")

    def run_simulation(self, active_synapse_indices):
        """
        运行单次模拟 (使用 SaveState 加速)
        """
        total_duration = self.cfg.T_SETTLE + self.cfg.T_RECORD
        total_steps = int(total_duration / self.cfg.DT) + 1
        stim_time_abs = self.cfg.T_SETTLE + self.cfg.STIM_DELAY

        # 1. 设置记录器 (保持不变)
        t_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec = h.Vector()
        v_vec.record(self.cell.soma[0](0.5)._ref_v)

        # 2. 恢复状态 (注意：此时内存中的 NetCon 数量与 SaveState 记录的一致)
        h.finitialize(self.cfg.V_INIT)
        self.savestate.restore()

        # 强制重置时间与状态
        h.t = 0.0
        h.tstop = total_duration
        h.fcurrent()

        # 3. 发送事件 (关键修改：直接使用 self.netcons)
        for syn_idx in active_synapse_indices:
            self.netcons[syn_idx].event(stim_time_abs)

        # 4. 运行
        h.continuerun(total_duration)

        # 5. 数据处理
        v_trace = np.array(v_vec.to_python(), dtype=np.float32)

        if len(v_trace) > total_steps:
            v_trace = v_trace[:total_steps]
        elif len(v_trace) < total_steps:
            pad_width = total_steps - len(v_trace)
            v_trace = np.pad(v_trace, (0, pad_width), 'edge')

        num_synapses = len(self.synapses)
        input_matrix = np.zeros((total_steps, num_synapses), dtype=np.uint8)

        stim_step_idx = int(stim_time_abs / self.cfg.DT)
        if 0 <= stim_step_idx < total_steps:
            for syn_idx in active_synapse_indices:
                input_matrix[stim_step_idx, syn_idx] = 1

        # 在返回前直接强行截断至目标时间步 (例如 6000)
        target_steps = int((self.cfg.T_SETTLE + self.cfg.T_RECORD) / self.cfg.DT)
        input_matrix = input_matrix[:target_steps, :]
        v_trace = v_trace[:target_steps]
        return input_matrix, v_trace.reshape(-1, 1)


# ==========================================
# 3. HDF5 管理器 (The Data Writer)
# ==========================================
class H5_Manager:
    def __init__(self, filepath, config):
        self.filepath = filepath
        self.cfg = config
        self.f = None
        self.dsets = {}  # 保存 dataset 对象的引用

    def initialize_file(self, static_metadata, total_sim_steps, num_synapses):
        """
        创建 HDF5 文件结构，写入静态信息
        如果文件存在，且结构匹配，则追加；否则覆盖 (或报错)
        """
        # 使用 'a' 模式：读/写，如果不存在则创建
        self.f = h5py.File(self.filepath, 'a')

        # 1. 写入 /static_info (仅当不存在时)
        if 'static_info' not in self.f:
            static_grp = self.f.create_group('static_info')
            for key, value in static_metadata.items():
                static_grp.create_dataset(key, data=value)

            # 写入根属性
            self.f.attrs['neuron_name'] = "L5PC"
            self.f.attrs['dt'] = static_metadata['dt']
            self.f.attrs['total_time'] = static_metadata['total_time']
            self.f.attrs['num_synapses'] = num_synapses

            print(f"Initialized new HDF5 file: {self.filepath}")
        else:
            print(f"Opened existing HDF5 file: {self.filepath}")

        # 准备 Blosc 压缩参数 (使用 lz4 算法，等级 5，开启 SHUFFLE 优化连续零的压缩率)
        comp_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

        # 2. 准备动态数据集 /dataset
        if 'dataset' not in self.f:
            dset_grp = self.f.create_group('dataset')

            # A. inputs (输入脉冲矩阵)
            dset_grp.create_dataset(
                'inputs',
                shape=(0, total_sim_steps, num_synapses),
                maxshape=(None, total_sim_steps, num_synapses),
                dtype='uint8',
                chunks=(1, total_sim_steps, num_synapses),
                **comp_kwargs  # 替换原来的 compression 参数
            )

            # B. targets (输出膜电位)
            dset_grp.create_dataset(
                'targets',
                shape=(0, total_sim_steps, 1),
                maxshape=(None, total_sim_steps, 1),
                dtype='float32',
                chunks=(1, total_sim_steps, 1),
                **comp_kwargs  # 替换原来的 compression 参数
            )

        # 保存 dataset 引用以便快速写入
        self.dsets['inputs'] = self.f['dataset']['inputs']
        self.dsets['targets'] = self.f['dataset']['targets']

    def append_data(self, inputs_batch, targets_batch):
        """
        将一批数据写入文件
        inputs_batch: (Batch, Time, Synapses) - uint8
        targets_batch: (Batch, Time, 1) - float32
        """
        # 确保文件是打开的
        if not self.f:
            raise RuntimeError("HDF5 file is not initialized.")

        # 获取当前样本数
        current_size = self.dsets['inputs'].shape[0]
        batch_size = inputs_batch.shape[0]
        new_size = current_size + batch_size

        # 1. Resize 数据集以容纳新数据
        self.dsets['inputs'].resize(new_size, axis=0)
        self.dsets['targets'].resize(new_size, axis=0)

        # 2. 写入新数据
        self.dsets['inputs'][current_size:new_size] = inputs_batch
        self.dsets['targets'][current_size:new_size] = targets_batch

        # 3. Flush 确保数据落盘 (防止程序中途崩溃导致数据丢失)
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            print("HDF5 file closed.")


# ==========================================
# 4. 主执行逻辑 (Main Execution)
# ==========================================
def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="L5PC Parallel Simulation")
    parser.add_argument("--job_id", type=int, default=0, help="当前任务ID (0-indexed)")
    parser.add_argument("--total_jobs", type=int, default=1, help="总并行任务数")
    parser.add_argument("--mode", type=str, choices=['setup', 'pairs'], default='setup',
                        help="运行模式: 'setup'或 'pairs'")
    parser.add_argument("--batch_size", type=int, default=50, help="H5写入批次大小")
    # 新增两个参数
    parser.add_argument("--out_dir", type=str, default="results", help="输出文件夹路径")
    parser.add_argument("--smoke_test", action="store_true", help="开启冒烟测试 (仅抽取极少量突触)")
    parser.add_argument("--target_v", type=float, default=None,
                        help="目标稳态电压 (如 -75.0)。若不指定，则保持原始零背景静息态。")
    args = parser.parse_args()

    cfg = Config()
    cfg.TARGET_V = args.target_v  # 将参数挂载到 cfg 上

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. 动态设置输出文件名 (加入目录路径)
    if args.mode == 'setup':
        cfg.OUTPUT_FILE = os.path.join(args.out_dir, "L5PC_setup.h5")
        print(f"=== Running Mode: SETUP ===")
    else:
        cfg.OUTPUT_FILE = os.path.join(args.out_dir, f"L5PC_pairs_part_{args.job_id}.h5")
        print(f"=== Running Mode: PAIRS | Job {args.job_id + 1}/{args.total_jobs} ===")

    # 3. 初始化模型
    print("Initializing Model...")
    model = L5PC_Model(cfg)
    meta = model.get_static_metadata()
    total_synapses = len(model.synapses)
    total_sim_steps = meta['total_time']

    print(f"Total Synapses: {total_synapses}")

    # 4. 初始化 HDF5
    print(f"Initializing HDF5: {cfg.OUTPUT_FILE}")
    h5_mgr = H5_Manager(cfg.OUTPUT_FILE, cfg)
    h5_mgr.initialize_file(meta, total_sim_steps, total_synapses)

    # 5. 任务分配逻辑 (核心修改点)
    def get_tasks():
        # 如果是冒烟测试，我们固定随机种子并只挑 15 个突触
        # 必须固定种子！这样并行的所有 Job 才能“看到”同一个突触池，正确进行切片
        if args.smoke_test:
            np.random.seed(42)
            subset = np.random.choice(total_synapses, size=15, replace=False).tolist()
            subset.sort()
        else:
            subset = list(range(total_synapses))

        if args.mode == 'setup':
            yield []
            for i in subset:
                yield [i]
        elif args.mode == 'pairs':
            # 只生成 subset 内的组合
            all_pairs = list(itertools.combinations(subset, 2))
            total_tasks = len(all_pairs)

            chunk_size = math.ceil(total_tasks / args.total_jobs)
            start_idx = args.job_id * chunk_size
            end_idx = min(start_idx + chunk_size, total_tasks)

            my_tasks = all_pairs[start_idx:end_idx]
            print(f"Task Allocation: Processing pairs {start_idx} to {end_idx} (Count: {len(my_tasks)})")

            for pair in my_tasks:
                yield list(pair)

    # 6. 执行循环
    inputs_buffer = []
    targets_buffer = []
    count = 0
    start_time = time.time()

    try:
        for active_indices in get_tasks():
            # 运行模拟
            inp, targ = model.run_simulation(active_indices)

            inputs_buffer.append(inp)
            targets_buffer.append(targ)

            # 缓冲区满则写入
            if len(inputs_buffer) >= args.batch_size:
                batch_inputs = np.stack(inputs_buffer, axis=0)
                batch_targets = np.stack(targets_buffer, axis=0)

                h5_mgr.append_data(batch_inputs, batch_targets)

                inputs_buffer = []
                targets_buffer = []

                count += args.batch_size
                if count % (args.batch_size * 2) == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"Progress: {count} sims done. Rate: {rate:.2f} sim/s")

        # 写入剩余数据
        if inputs_buffer:
            batch_inputs = np.stack(inputs_buffer, axis=0)
            batch_targets = np.stack(targets_buffer, axis=0)
            h5_mgr.append_data(batch_inputs, batch_targets)
            count += len(inputs_buffer)

        print(f"\n[Done] Job finished. Total samples generated: {count}")
        print(f"Data saved to: {cfg.OUTPUT_FILE}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Closing file safely...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        h5_mgr.close()


if __name__ == "__main__":
    main()