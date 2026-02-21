import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import hdf5plugin


def main():
    # 相对路径指向我们实验输出的数据文件
    filepath = "results/exp_invivo_6s/final_dataset.h5"

    print(f"📂 正在加载数据集: {filepath}")
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath} (请确认 6 秒 In-vivo 实验已跑完并成功合并)")
        sys.exit(1)

    with h5py.File(filepath, 'r') as f:
        targets = f['dataset/train/targets'][:]
        dt = f.attrs.get('dt', 0.1)
        total_steps = f.attrs.get('total_steps', 60000)

    num_trials = targets.shape[0]
    # 将总步数转换为时长 (秒)
    duration_s = (total_steps * dt) / 1000.0

    print(f"✅ 成功加载 {num_trials} 个 {int(duration_s)} 秒时长的 Trial 数据。\n")

    # --------------------------------------------------
    # 1. 全局电压统计
    # --------------------------------------------------
    mean_v = np.mean(targets)
    std_v = np.std(targets)

    print("📊 --- 全局电压统计 ---")
    print(f"  均值 (Mean Voltage): {mean_v:.2f} mV")
    print(f"  标准差 (Std Fluctuation): {std_v:.2f} mV\n")

    # --------------------------------------------------
    # 2. 发放率统计 (Spike Firing Rate)
    # --------------------------------------------------
    # 设定脉冲阈值，当电压从低于阈值越过高于阈值时记为 1 个 Spike
    SPIKE_THRESHOLD = -20.0

    spike_counts = np.zeros(num_trials, dtype=int)
    for i in range(num_trials):
        v_trace = targets[i, :, 0]
        # 寻找向上穿过阈值的点
        crossings = np.sum((v_trace[:-1] < SPIKE_THRESHOLD) & (v_trace[1:] >= SPIKE_THRESHOLD))
        spike_counts[i] = crossings

    firing_rates = spike_counts / duration_s
    avg_rate = np.mean(firing_rates)
    max_rate = np.max(firing_rates)
    max_count = np.max(spike_counts)

    print("⚡ --- 发放率统计 (Spike Firing Rate) ---")
    print(f"  平均发放率: {avg_rate:.2f} Hz")
    print(f"  最高发放率: {max_rate:.2f} Hz (对应 {max_count} 个脉冲)\n")

    # --------------------------------------------------
    # 3. 可视化大图
    # --------------------------------------------------
    print("🎨 正在随机抽取 20 个 Trial 生成可视化大图...")

    # 随机挑选 20 个不重复的索引
    np.random.seed(int(time.time()) if 'time' in sys.modules else 42)
    sample_indices = np.random.choice(num_trials, 20, replace=False)

    # 创建 5行 4列 的子图网格
    fig, axes = plt.subplots(5, 4, figsize=(20, 15), sharex=True, sharey=True)
    time_axis_s = np.arange(total_steps) * dt / 1000.0  # x轴转化为秒

    axes_flat = axes.flatten()
    for i, idx in enumerate(sample_indices):
        ax = axes_flat[i]
        v_trace = targets[idx, :, 0]

        ax.plot(time_axis_s, v_trace, color='#1f77b4', linewidth=0.8, alpha=0.9)
        # 标注该图的编号和对应的发放率
        ax.set_title(f"Trial {idx} (Rate: {firing_rates[idx]:.1f} Hz)", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

        # 美化：只在最左侧和最下方显示刻度标签
        if i >= 16:  # 最后一行
            ax.set_xlabel("Time (s)", fontsize=11)
        if i % 4 == 0:  # 第一列
            ax.set_ylabel("Voltage (mV)", fontsize=11)

    # 紧凑布局
    plt.tight_layout()

    out_img = "invivo_random_20_samples.png"
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    print(f"✅ 绘图完成！图片已保存至当前目录: {out_img}")


if __name__ == "__main__":
    # 如果 sys.modules 中没有 time，补引一下用于随机种子
    import time

    main()