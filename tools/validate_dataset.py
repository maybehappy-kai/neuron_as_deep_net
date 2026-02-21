import os
import argparse
import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt


def validate_and_extract(filepath):
    print(f"\n{'=' * 60}\n🔍 正在严格验证: {filepath}\n{'=' * 60}")
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath} (请确认实验已跑完)")
        return None, None, None

    with h5py.File(filepath, 'r') as f:
        # --------------------------------------------------
        # 1. 检查根属性 (.attrs)
        # --------------------------------------------------
        print("--- 检查根属性 (.attrs) ---")
        expected_attrs = ['neuron_name', 'dt', 'total_steps', 'num_synapses', 'target_v_mV']
        for attr in expected_attrs:
            if attr in f.attrs:
                print(f"  ✅ {attr}: {f.attrs[attr]}")
            else:
                print(f"  ❌ 缺失核心属性: {attr}")

        dt = f.attrs.get('dt', 0.1)
        total_steps = f.attrs.get('total_steps', 0)
        num_synapses = f.attrs.get('num_synapses', 0)
        target_v = f.attrs.get('target_v_mV', np.nan)

        # --------------------------------------------------
        # 2. 检查静态信息组 (/static_info)
        # --------------------------------------------------
        print("\n--- 检查 /static_info ---")
        if 'static_info' in f:
            static_grp = f['static_info']
            expected_static = ['num_subunits', 'parent_indices', 'input_map', 'synapse_types']
            for key in expected_static:
                if key in static_grp:
                    print(f"  ✅ {key:15}: shape {static_grp[key].shape}, dtype {static_grp[key].dtype}")
                else:
                    print(f"  ❌ 缺失静态拓扑信息: {key}")

            # 【专项检查】E/I 排序 (兴奋性1必须全在抑制性-1前面)
            if 'synapse_types' in static_grp:
                syn_types = static_grp['synapse_types'][:]
                diffs = np.diff(syn_types)
                if np.any(diffs > 0):
                    print("  ❌ E/I 排序非法！发现兴奋性突触排在了抑制性突触的后面。")
                else:
                    exc_count = np.sum(syn_types == 1)
                    inh_count = np.sum(syn_types == -1)
                    print(f"  ✅ E/I 排序正确 (前 {exc_count} 个为 E，后 {inh_count} 个为 I)。")
        else:
            print("  ❌ 缺失组: /static_info")

        # --------------------------------------------------
        # 3. 检查数据集 (/dataset/train)
        # --------------------------------------------------
        print("\n--- 检查 /dataset/train 数据层级与格式 ---")
        if 'dataset/train/inputs' in f and 'dataset/train/targets' in f:
            inputs = f['dataset/train/inputs']
            targets = f['dataset/train/targets']

            # Inputs 验证
            print(f"  [inputs] Shape: {inputs.shape}")
            if inputs.shape[1] == total_steps and inputs.shape[2] == num_synapses:
                print("    ✅ 维度对齐正确")
            else:
                print(f"    ❌ 维度对齐错误！期望: (N, {total_steps}, {num_synapses})")

            print(f"  [inputs] Dtype: {inputs.dtype}")
            if inputs.dtype == np.uint8:
                print("    ✅ 数据类型正确 (uint8)")
            else:
                print("    ❌ 数据类型错误！期望: uint8")

            print(f"  [inputs] Chunks: {inputs.chunks}")
            if inputs.chunks == (1, total_steps, num_synapses):
                print("    ✅ Chunking 切块策略完美符合要求")
            else:
                print(f"    ❌ Chunking 策略错误！期望: (1, {total_steps}, {num_synapses})")

            # Targets 验证
            print(f"  [targets] Shape: {targets.shape}")
            if targets.shape[1] == total_steps and targets.shape[2] == 1:
                print("    ✅ 维度对齐正确")
            else:
                print(f"    ❌ 维度对齐错误！期望: (N, {total_steps}, 1)")

            print(f"  [targets] Dtype: {targets.dtype}")
            if targets.dtype == np.float32:
                print("    ✅ 数据类型正确 (float32)")
            else:
                print("    ❌ 数据类型错误！期望: float32")

            print(f"  [targets] Chunks: {targets.chunks}")
            if targets.chunks == (1, total_steps, 1):
                print("    ✅ Chunking 切块策略完美符合要求")
            else:
                print(f"    ❌ Chunking 策略错误！期望: (1, {total_steps}, 1)")

            extracted_targets = targets[:]
            return extracted_targets, dt, target_v
        else:
            print("  ❌ 缺失组: /dataset/train/inputs 或 targets")
            return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Validate and Plot HDF5 Dataset")
    parser.add_argument("--file1", type=str, default="results/smoke_test_rest/final_dataset.h5",
                        help="第一个要验证的文件")
    parser.add_argument("--file2", type=str, default="results/smoke_test_active/final_dataset.h5",
                        help="第二个要验证的文件")
    parser.add_argument("--out_img", type=str, default="dataset_validation_traces.png", help="保存的图片名称")
    args = parser.parse_args()

    traces1, dt1, target_v1 = validate_and_extract(args.file1)
    traces2, dt2, target_v2 = validate_and_extract(args.file2)

    # --------------------------------------------------
    # 4. 可视化电压轨迹
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("🎨 正在生成电压轨迹可视化图...")

    if traces1 is not None or traces2 is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

        if traces1 is not None:
            time_axis1 = np.arange(traces1.shape[1]) * dt1
            for i in range(traces1.shape[0]):
                axes[0].plot(time_axis1, traces1[i, :, 0], label=f'Trial {i + 1}', linewidth=1.2, alpha=0.8)
            axes[0].set_title(f"Dataset 1 Traces ({target_v1:.2f} mV)", fontsize=14, fontweight='bold')
            axes[0].set_xlabel("Time (ms)", fontsize=12)
            axes[0].set_ylabel("Membrane Voltage (mV)", fontsize=12)
            axes[0].grid(True, linestyle='--', alpha=0.6)

        if traces2 is not None:
            time_axis2 = np.arange(traces2.shape[1]) * dt2
            for i in range(traces2.shape[0]):
                axes[1].plot(time_axis2, traces2[i, :, 0], label=f'Trial {i + 1}', linewidth=1.2, alpha=0.8)
            axes[1].set_title(f"Dataset 2 Traces ({target_v2:.2f} mV)", fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Time (ms)", fontsize=12)
            axes[1].set_ylabel("Membrane Voltage (mV)", fontsize=12)
            axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(args.out_img)
        print(f"✅ 可视化完成！图片已保存至当前目录: {args.out_img}")
    else:
        print("❌ 没有提取到数据用于画图，请检查文件路径是否正确。")


if __name__ == "__main__":
    main()