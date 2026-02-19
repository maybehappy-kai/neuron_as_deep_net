import h5py
import numpy as np
import glob
import os
import sys
import time

# ================= 配置区 =================
OUTPUT_FILENAME = "L5PC_final.h5"
SETUP_FILENAME = "L5PC_setup.h5"
PARTS_PATTERN = "L5PC_pairs_part_*.h5"

# 目标时间步长 (切掉多余的1帧)
TARGET_TIME_STEPS = 6000

# 压缩设置
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4

# 批处理大小 (样本数)
# 1个样本 (6000x1278 uint8) ≈ 7.6 MB
# 100个样本 ≈ 760 MB. A100服务器内存通常很大，设为 200 (1.5GB) 是安全的
BATCH_SIZE = 200


# =========================================

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def get_synapse_permutation(setup_file):
    """计算突触重排索引"""
    print(f"Reading metadata from {setup_file}...")
    with h5py.File(setup_file, 'r') as f:
        old_types = f['static_info']['synapse_types'][:]
        old_map = f['static_info']['input_map'][:]
        num_subunits = f['static_info']['num_subunits'][()]
        parent_indices = f['static_info']['parent_indices'][:]
        attrs = dict(f.attrs)

    ex_indices = np.where(old_types == 1)[0]
    inh_indices = np.where(old_types == -1)[0]
    perm_idx = np.concatenate([ex_indices, inh_indices])

    # 验证
    new_types = old_types[perm_idx]
    new_map = old_map[perm_idx]
    assert np.all(new_types[:len(ex_indices)] == 1)

    return {
        'perm_idx': perm_idx,
        'new_types': new_types,
        'new_map': new_map,
        'num_subunits': num_subunits,
        'parent_indices': parent_indices,
        'attrs': attrs
    }


def scan_total_samples(file_list):
    """快速扫描所有文件获取总样本数"""
    total = 0
    print("Scanning files to calculate total size...")
    for fpath in file_list:
        try:
            with h5py.File(fpath, 'r') as f:
                if 'dataset/inputs' in f:
                    total += f['dataset']['inputs'].shape[0]
        except Exception as e:
            print(f"  [WARN] Skipping broken file {fpath}: {e}")
    print(f"  -> Total samples found: {total}")
    return total


def main():
    # 0. 清理旧文件
    if os.path.exists(OUTPUT_FILENAME):
        print(f"Removing old {OUTPUT_FILENAME}...")
        os.remove(OUTPUT_FILENAME)

    # 1. 准备文件列表
    if not os.path.exists(SETUP_FILENAME):
        print(f"Error: Setup file {SETUP_FILENAME} not found!")
        return
    part_files = sorted(glob.glob(PARTS_PATTERN), key=lambda x: int(x.split('_part_')[-1].split('.')[0]))
    all_files = [SETUP_FILENAME] + part_files

    # 2. 获取元数据与重排索引
    meta = get_synapse_permutation(SETUP_FILENAME)
    perm_idx = meta['perm_idx']

    # 3. 预计算总大小
    total_samples = scan_total_samples(all_files)
    if total_samples == 0:
        print("No samples found. Exiting.")
        return

    # 4. 创建目标文件并分配空间 (一次性分配)
    print_header(f"Initializing Output File: {OUTPUT_FILENAME}")
    with h5py.File(OUTPUT_FILENAME, 'w') as f_out:
        # 写入静态信息
        for k, v in meta['attrs'].items():
            f_out.attrs[k] = v
        # 修正 total_time 属性
        f_out.attrs['total_time'] = TARGET_TIME_STEPS

        g_static = f_out.create_group('static_info')
        g_static.create_dataset('num_subunits', data=meta['num_subunits'])
        g_static.create_dataset('parent_indices', data=meta['parent_indices'])
        g_static.create_dataset('input_map', data=meta['new_map'])
        g_static.create_dataset('synapse_types', data=meta['new_types'])

        g_train = f_out.create_group('dataset/train')

        num_synapses = meta['attrs']['num_synapses']

        print(f"Allocating datasets: ({total_samples}, {TARGET_TIME_STEPS}, {num_synapses})...")

        # 创建固定大小的数据集 (比 maxshape=None 性能更好)
        dset_inputs = g_train.create_dataset(
            'inputs',
            shape=(total_samples, TARGET_TIME_STEPS, num_synapses),
            dtype='uint8',
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
            chunks=(1, TARGET_TIME_STEPS, num_synapses)
        )

        dset_targets = g_train.create_dataset(
            'targets',
            shape=(total_samples, TARGET_TIME_STEPS, 1),
            dtype='float32',
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
            chunks=(1, TARGET_TIME_STEPS, 1)
        )

        # 5. 开始填充数据
        print_header("Starting Optimized Merge")
        global_idx = 0
        start_time = time.time()

        for idx, fpath in enumerate(all_files):
            print(f"[{idx + 1}/{len(all_files)}] Reading {fpath}...", end='', flush=True)

            try:
                with h5py.File(fpath, 'r') as f_src:
                    if 'dataset/inputs' not in f_src:
                        print(" -> Skipped (No data)")
                        continue

                    src_inputs = f_src['dataset']['inputs']
                    src_targets = f_src['dataset']['targets']
                    n_file = src_inputs.shape[0]

                    # 分块读取并写入
                    for i in range(0, n_file, BATCH_SIZE):
                        end = min(i + BATCH_SIZE, n_file)

                        # 1. 读取数据
                        # 切片 [:, :6000, :] 丢弃最后一个时间步
                        X_chunk = src_inputs[i:end, :TARGET_TIME_STEPS, :]
                        Y_chunk = src_targets[i:end, :TARGET_TIME_STEPS, :]

                        # 2. 重排突触 (Permutation)
                        # 利用 Numpy 高级索引重排最后一维
                        X_chunk = X_chunk[:, :, perm_idx]

                        # 3. 写入目标
                        write_end = global_idx + (end - i)
                        dset_inputs[global_idx:write_end] = X_chunk
                        dset_targets[global_idx:write_end] = Y_chunk

                        global_idx = write_end

                    print(f" -> Done ({n_file} samples)")

            except Exception as e:
                print(f"\n[ERROR] Failed on {fpath}: {e}")
                # 即使出错也不要退出，继续处理下一个文件？
                # 这里选择继续，但最终数据可能有空洞（0）。
                # 鉴于是一次性分配，未写入部分将保持为 0。

        duration = time.time() - start_time
        print_header("Merge Complete")
        print(f"Expected Samples: {total_samples}")
        print(f"Written Samples:  {global_idx}")
        print(f"Time Taken: {duration:.2f} seconds")
        print(f"Output: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()