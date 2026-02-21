import os
# 抑制 GUI
os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'
import json
import numpy as np
from scipy.optimize import brentq
import neuron
from neuron import h


class L5PC_Env:
    """
    纯粹的 L5PC 神经元物理环境。
    只负责初始化拓扑、设置背景电导达到目标电压、接受脉冲事件并返回电压响应。
    """

    def __init__(self, morphology_file, biophys_file, template_file, dt=0.1, celsius=34.0):
        self.dt = dt
        self.celsius = celsius
        self.morphology_file = morphology_file
        self.biophys_file = biophys_file
        self.template_file = template_file

        self.cell = None
        self.synapses = []
        self.segments = []
        self.netcons = []
        self.savestate = None
        self.steady_state_v = None

        self._init_neuron()
        self._setup_model_and_synapses()

    def _init_neuron(self):
        h.load_file('nrngui.hoc')
        h.load_file("import3d.hoc")
        h.load_file(self.biophys_file)
        h.load_file(self.template_file)

        h.celsius = self.celsius
        h.dt = self.dt
        self.cell = h.L5PCtemplate(self.morphology_file)
        h.cvode_active(0)

    def _apply_scaled_background(self, target_v, anchors_path="bg_anchors.json"):
        """读取锚点并使用 Brent 方法极速寻找匹配 target_v 的缩放系数 alpha"""
        try:
            with open(anchors_path, "r") as f:
                anchors = json.load(f)
            ge_max = anchors["g_exc_max"]
            gi_max = anchors["g_inh_max"]
        except FileNotFoundError:
            raise RuntimeError(f"{anchors_path} not found! Please run find_bg_conductance.py first.")

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

        def voltage_error(alpha):
            apply_alpha(alpha)
            h.finitialize(target_v)
            h.continuerun(2000.0)
            return self.cell.soma[0](0.5).v - target_v

        # 极速求解：将上限放宽到 2.0，给浮点数计算和更高电压目标留出交叉空间
        try:
            final_alpha = brentq(voltage_error, 0.0, 2.0, xtol=1e-4)
        except ValueError:
            # 如果放宽到 2.0 依然找不到（说明你要的 target_v 太高或太低，超出了物理极限）
            # 我们先尝试打印当时的极限值以便 debug
            v_0 = voltage_error(0.0) + target_v
            v_2 = voltage_error(2.0) + target_v
            raise ValueError(f"目标电压 {target_v}mV 无法达到！"
                             f"当前模型可调节范围大约在 [{v_0:.2f}mV, {v_2:.2f}mV] 之间。")
        apply_alpha(final_alpha)
        self.steady_state_v = target_v

        return final_alpha, ge_max * final_alpha, gi_max * final_alpha

    def _setup_model_and_synapses(self):
        """记录拓扑并在树突上布置突触，预创建 NetCon (精简了冗余打印)"""
        all_sections = list(self.cell.soma) + list(self.cell.dend) + list(self.cell.apic)
        sec_name_to_indices = {}

        for sec in all_sections:
            sec_name = sec.name()
            sec_name_to_indices[sec_name] = []
            for seg in sec:
                sec_name_to_indices[sec_name].append(len(self.segments))
                self.segments.append(seg)

        # 构建父节点索引
        self.parent_indices = np.zeros(len(self.segments), dtype=np.int32)
        for i, seg in enumerate(self.segments):
            sec_name = seg.sec.name()
            is_first_seg = (i == sec_name_to_indices[sec_name][0])
            if not is_first_seg:
                self.parent_indices[i] = i - 1
            else:
                parent_seg = seg.sec.parentseg()
                if parent_seg is None:
                    self.parent_indices[i] = -1
                else:
                    p_sec_name = parent_seg.sec.name()
                    if p_sec_name in sec_name_to_indices:
                        candidate_indices = sec_name_to_indices[p_sec_name]
                        # 找位置 x 最接近的
                        best_idx = min(candidate_indices, key=lambda idx: abs(self.segments[idx].x - parent_seg.x))
                        self.parent_indices[i] = best_idx
                    else:
                        self.parent_indices[i] = -1

        # 布置突触 (辅助函数保持不变，仅省略了长长的参数设置以节省空间，建议原样拷贝你之前的 create_synapse 逻辑)
        def create_synapse(seg, syn_type):
            if syn_type == 'AMPA_NMDA':
                # 严格对齐官方 NMDA 突触
                syn = h.ProbAMPANMDA2(seg)
                syn.tau_r_AMPA = 0.3
                syn.tau_d_AMPA = 3.0
                syn.tau_r_NMDA = 2.0
                syn.tau_d_NMDA = 70.0
                syn.gmax = 0.0004
                syn.e = 0.0
            elif syn_type == 'GABA_A':
                # 严格对齐官方 GABA_A 突触
                syn = h.ProbUDFsyn2(seg)
                syn.tau_r = 0.2
                syn.tau_d = 8.0
                syn.gmax = 0.001
                syn.e = -80.0
            else:
                raise ValueError(f"Unknown synapse type: {syn_type}")

            # 统一的突触短时程可塑性参数 (当前均设为无)
            syn.Use = 1.0
            syn.u0 = 0.0
            syn.Dep = 0.0
            syn.Fac = 0.0

            return syn

        self.input_map = []
        self.synapse_types = []
        soma_sec_set = set(list(self.cell.soma))

        ex_syns_temp = []
        inh_syns_temp = []
        for i, seg in enumerate(self.segments):
            if seg.sec in soma_sec_set: continue
            ex_syns_temp.append((create_synapse(seg, 'AMPA_NMDA'), i, 1))
            inh_syns_temp.append((create_synapse(seg, 'GABA_A'), i, -1))

        for syn, seg_idx, s_type in (ex_syns_temp + inh_syns_temp):
            self.synapses.append(syn)
            self.input_map.append(seg_idx)
            self.synapse_types.append(s_type)

            nc = h.NetCon(None, syn)
            nc.weight[0] = 1.0
            nc.delay = 0.0
            self.netcons.append(nc)

        self.input_map = np.array(self.input_map, dtype=np.int32)
        self.synapse_types = np.array(self.synapse_types, dtype=np.int8)

        # 【修复核心2】: 记录每个片段的物理长度 (L / nseg)
        self.seg_lengths = np.array([seg.sec.L / seg.sec.nseg for seg in self.segments], dtype=np.float32)

    def warmup(self, target_v=None, anchors_path="bg_anchors.json"):
        """设定背景稳态并保存系统状态，动态读取真实平衡电压"""
        meta_info = {"alpha": 0.0, "g_exc": 0.0, "g_inh": 0.0}

        if target_v is not None:
            alpha, ge, gi = self._apply_scaled_background(target_v, anchors_path)
            meta_info.update({"alpha": alpha, "g_exc": ge, "g_inh": gi})
            # 给出求解器初值猜测
            initial_guess_v = target_v
        else:
            # 无背景干涉时的初值猜测（仅供 NEURON 初始化使用，并非最终结果）
            initial_guess_v = -81.1

        h.finitialize(initial_guess_v)
        h.continuerun(3000.0)  # 跑 3 秒让 Ih 等慢通道彻底平衡

        # 【核心修正】：无论有无干涉，一律读取胞体此时的真实电位！
        actual_steady_v = self.cell.soma[0](0.5).v
        self.steady_state_v = actual_steady_v
        meta_info["target_v_mV"] = float(actual_steady_v)

        self.savestate = h.SaveState()
        self.savestate.save()

        print(f"[Environment] Warmup complete. Actual Steady State V: {actual_steady_v:.3f} mV")
        return meta_info

    def get_topology_metadata(self):
        """返回不随仿真时间改变的静态拓扑元数据"""
        return {
            "num_subunits": len(self.segments),
            "num_synapses": len(self.synapses),
            "parent_indices": self.parent_indices,
            "input_map": self.input_map,
            "synapse_types": self.synapse_types,
            "dt": self.dt
        }

    def run_simulation(self, spike_events, total_duration):
        """
        核心 API：接受灵活的脉冲事件列表，返回电压响应
        :param spike_events: List of Tuples -> [(synapse_index, spike_time_ms), ...]
        :param total_duration: float, 仿真总时长(ms)
        :return: np.array (v_trace, shape=(steps, 1))
        """
        v_vec = h.Vector()
        v_vec.record(self.cell.soma[0](0.5)._ref_v)

        h.finitialize(self.steady_state_v)
        self.savestate.restore()

        h.t = 0.0
        h.tstop = total_duration
        h.fcurrent()

        # 灵活分配所有的脉冲事件
        for syn_idx, spike_time in spike_events:
            self.netcons[syn_idx].event(float(spike_time))

        h.continuerun(total_duration)

        v_trace = np.array(v_vec.to_python(), dtype=np.float32)
        target_steps = int(total_duration / self.dt)

        # 裁剪或填充以防数值误差导致的步数不一
        if len(v_trace) > target_steps:
            v_trace = v_trace[:target_steps]
        elif len(v_trace) < target_steps:
            v_trace = np.pad(v_trace, (0, target_steps - len(v_trace)), 'edge')

        return v_trace.reshape(-1, 1)