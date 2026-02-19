import h5py
import hdf5plugin  # 必须导入以解码 Blosc
import numpy as np
import glob
import os
import time
import argparse # 新增

def main():
    # 增加命令行参数读取目录
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="包含碎片的目录路径")
    args = parser.parse_args()

    # 将路径拼接到目标目录下
    OUTPUT_FILENAME = os.path.join(args.data_dir, "L5PC_final.h5")
    SETUP_FILENAME = os.path.join(args.data_dir, "L5PC_setup.h5")
    PARTS_PATTERN = os.path.join(args.data_dir, "L5PC_pairs_part_*.h5")

    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)

    # 1. 获取文件列表
    part_files = sorted(glob.glob(PARTS_PATTERN), key=lambda x: int(x.split('_part_')[-1].split('.')[0]))
    all_files = [SETUP_FILENAME] + part_files

    # 2. 快速扫描总样本数并提取静态信息
    total_samples = 0
    with h5py.File(SETUP_FILENAME, 'r') as f_setup:
        # 直接拷贝静态数据属性
        static_info = {k: v[()] for k, v in f_setup['static_info'].items()}
        attrs = dict(f_setup.attrs)

        target_time_steps = attrs['total_time']
        num_synapses = attrs['num_synapses']

    for fpath in all_files:
        with h5py.File(fpath, 'r') as f:
            if 'dataset/inputs' in f:
                total_samples += f['dataset']['inputs'].shape[0]

    print(f"Total samples to merge: {total_samples}")
    if total_samples == 0:
        return

    # 3. 创建目标文件 (同样使用 Blosc)
    comp_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

    with h5py.File(OUTPUT_FILENAME, 'w') as f_out:
        for k, v in attrs.items():
            f_out.attrs[k] = v

        g_static = f_out.create_group('static_info')
        for k, v in static_info.items():
            g_static.create_dataset(k, data=v)

        g_train = f_out.create_group('dataset/train')

        dset_inputs = g_train.create_dataset(
            'inputs', shape=(total_samples, target_time_steps, num_synapses),
            dtype='uint8', chunks=(1, target_time_steps, num_synapses), **comp_kwargs
        )
        dset_targets = g_train.create_dataset(
            'targets', shape=(total_samples, target_time_steps, 1),
            dtype='float32', chunks=(1, target_time_steps, 1), **comp_kwargs
        )

        # 4. 零消耗合并：整块直接拷贝
        print("\nStarting Fast Merge...")
        global_idx = 0
        start_time = time.time()

        for idx, fpath in enumerate(all_files):
            print(f"[{idx + 1}/{len(all_files)}] Merging {fpath}...", end='', flush=True)
            with h5py.File(fpath, 'r') as f_src:
                if 'dataset/inputs' not in f_src:
                    print(" -> Skipped")
                    continue

                src_inputs = f_src['dataset']['inputs']
                src_targets = f_src['dataset']['targets']
                n_file = src_inputs.shape[0]

                end_idx = global_idx + n_file

                # 因为内存连续，也没有突触重排，直接一次性将整个文件的张量写入！
                dset_inputs[global_idx:end_idx] = src_inputs[:]
                dset_targets[global_idx:end_idx] = src_targets[:]

                global_idx = end_idx
                print(f" -> Done ({n_file} samples)")

        print(f"\nMerge Complete! Time Taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()