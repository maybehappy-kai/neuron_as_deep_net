# L5PC 神经元深度学习数据生成流水线 (Neuron-as-Deep-Net)

本项目将复杂的 L5PC（第五层锥体细胞）计算神经科学仿真（NEURON）与现代深度学习数据管道完美结合，支持大规模并行仿真、极速稳态逼近，并生成支持 Lazy Loading 的标准 HDF5 数据集。

---

## 📂 项目目录结构

* `core/` (底层基石，纯物理引擎与落盘工具，不可随意修改)
  * `config.py`: **全局物理参数配置中心**（统一管理活跃态电压目标值、OU 过程轰炸发放率等）。
  * `l5pc_env.py`: 神经元物理环境 API（读取真实稳态电压、动态缩放背景电导、接受脉冲矩阵积分）。
  * `hdf5_writer.py`: HDF5 纯数据写入器，执行 Blosc lz4 压缩和分块追加。
* `tools/` (支撑工具)
  * `find_bg_conductance.py`: 离线寻找目标电压对应的兴奋/抑制背景电导锚点。
  * `orchestrator.py`: 多进程并行调度器与碎片 H5 极速合并器。
* `experiments/` (实验策略脚本，日常科研阵地)
  * `run_smoke_test.py`: 冒烟测试（验证多种刺激范式和基准电压）。
  * `run_exp_active_pairs.py`: 81.7万次单/双突触配对 PSP 大规模仿真。
  * `run_exp_voltage_sweep.py`: 电压扫参主控脚本。
  * `run_exp_temporal_coupling.py`: 筛选最强 E-I 耦合对进行时差扫描。
  * `run_exp_invivo_6s.py`: 基于树突物理表面积缩放的 6 秒在体非齐次泊松轰炸。
* `configs/` & `results/` (产出物，自动生成)

---

## 💾 HDF5 数据规范与差异说明

所有实验最终生成的 `.h5` 文件均遵循以下核心结构：
* **根属性 (`.attrs`)**: 包含 `neuron_name`, `dt`, `total_steps`, `num_synapses`, `target_v_mV`（引擎实测的绝对真实稳态电压）。
* **静态拓扑组 (`/static_info`)**: 包含 `num_subunits`, `parent_indices`, `input_map`, `synapse_types` 等树突几何拓扑。
* **动态数据组 (`/dataset/train`)**:
  * `inputs`: `uint8` 稀疏矩阵，Shape 为 `(Trials, Steps, 1278)`。前 639 维严格为兴奋性，后 639 维为抑制性。
  * `targets`: `float32` 胞体电压，Shape 为 `(Trials, Steps, 1)`。

**各实验生成文件的微小差异：**
1. **电压扫参实验**：生成的 H5 内部会多出一个 `/dataset/train/conditions_v` 张量，记录当前 trial 对应的稳态电压（用于条件网络输入）。
2. **时序耦合实验**：内部会多出一个 `/dataset/train/conditions_dt` 张量，记录当前 trial 的 $\Delta t$（$t_E - t_I$）。

---

## 🚀 标准操作指南 (SOP) 与运行命令

**⚠️ 【核心避坑法则：何时该删除旧数据？】**
由于底层使用高效的追加写入（Append）模式，**如果您修改了实验脚本或 `config.py` 里的任何参数（如仿真时间、目标电压、发放率），必须先删除旧数据再运行，否则新旧数据将混合导致彻底污染！**
* **必须删除的情况**：修改了实验参数；或者使用 `orchestrator.py` 运行的实验中途被强杀/中断（会留下半成品碎片文件）。
* **无需删除的情况（仅限电压扫参）**：`run_exp_voltage_sweep.py` 自带断点续传。如果参数没变只是服务器中途重启了，直接重跑，它会跳过已完成的电压点。

### 1. 标定背景电导 (所有实验的前置要求)
计算达到活跃态（由 `config.py` 决定）所需的背景电导系数。此步骤耗时较高，但只需跑一次。
> **命令：**
> `rm -rf configs/bg_anchors.json && python tools/find_bg_conductance.py --out_json configs/bg_anchors.json`

### 2. 架构贯通测试 (Smoke Test)
测试基础架构是否畅通。包含单突触、成对、时序、随机等多种微型样本。
> **测定无干涉深度静息态：**
> `rm -rf results/smoke_test_rest && python tools/orchestrator.py --script "experiments/run_smoke_test.py --state rest" --workers 3 --out_dir results/smoke_test_rest`
>
> **测定高电导活跃态：**
> `rm -rf results/smoke_test_active && python tools/orchestrator.py --script "experiments/run_smoke_test.py --state active" --workers 3 --out_dir results/smoke_test_active`

### 3. 大规模配对仿真 (Active Pairs)
在活跃态下穷举所有 81.7 万个突触组合。计算量巨大，请分配尽可能多的 workers。
> **命令：**
> `rm -rf results/exp_active_pairs && python tools/orchestrator.py --script experiments/run_exp_active_pairs.py --workers 50 --out_dir results/exp_active_pairs`

### 4. 电压扫参实验 (Voltage Sweep)
测定不同背景电压下的单突触 PSP 响应。
*(注意：此脚本内置多进程主控，无需调用 orchestrator；自带断点续传，中途崩溃重跑时请去掉 `rm -rf` 以继续计算)*
> **命令：**
> `rm -rf results/voltage_sweep && python experiments/run_exp_voltage_sweep.py --workers 20`

### 5. 时序耦合扫描 (Temporal Coupling)
此步骤依赖实验 3 的结果。自动检索提取作用最强的前 K 个 E-I 对，并进行时差扫描。
*(注意：若想在没有实验 3 数据时直接测试运行，请在末尾加上 `--mock_pairs` 参数)*
> **命令：**
> `rm -f results/temporal_coupling/L5PC_TemporalCoupling.h5 && python experiments/run_exp_temporal_coupling.py --top_k 10`

### 6. 6秒 In-vivo 长时泊松轰炸
禁用人造背景干涉，完全根据树突各分段的物理表面积进行缩放，施加**基于 OU (Ornstein-Uhlenbeck) 过程的非齐次泊松轰炸**（具体基准发放率见 `config.py`），逼真还原在体（In-vivo）皮层网络的高电导震荡。
> **命令：**
> `rm -rf results/exp_invivo_6s && python tools/orchestrator.py --script experiments/run_exp_invivo_6s.py --workers 20 --out_dir results/exp_invivo_6s`