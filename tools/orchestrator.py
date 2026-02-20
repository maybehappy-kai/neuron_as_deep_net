import os
import sys
import time
import glob
import h5py
import argparse
import subprocess
import hdf5plugin
import shlex


def fast_merge(data_dir, output_name="final_dataset.h5"):
    """
    零消耗快速合并逻辑，直接在块级别搬运 Blosc 压缩数据。
    """
    output_filepath = os.path.join(data_dir, output_name)
    meta_file = os.path.join(data_dir, "meta.h5")
    parts_pattern = os.path.join(data_dir, "part_*.h5")

    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    part_files = sorted(glob.glob(parts_pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    if not os.path.exists(meta_file) or not part_files:
        print(f"❌ Merge failed: Could not find meta.h5 or part files in {data_dir}")
        return

    # 1. 扫描总样本数并提取静态信息
    total_samples = 0
    with h5py.File(meta_file, 'r') as f_meta:
        static_info = {k: v[()] for k, v in f_meta['static_info'].items()}
        attrs = dict(f_meta.attrs)
        target_time_steps = attrs['total_steps']
        num_synapses = attrs['num_synapses']

    for fpath in part_files:
        with h5py.File(fpath, 'r') as f:
            if 'dataset/train/inputs' in f:  # 加上 /train
                total_samples += f['dataset/train/inputs'].shape[0]

    print(f"\nTotal samples to merge: {total_samples}")
    if total_samples == 0:
        return

    # 2. 创建目标文件并写入元数据
    comp_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

    with h5py.File(output_filepath, 'w') as f_out:
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

        # 3. 整块拷贝合并
        print("Starting Fast Merge...")
        global_idx = 0
        start_time = time.time()

        for idx, fpath in enumerate(part_files):
            print(f"  [{idx + 1}/{len(part_files)}] Merging {os.path.basename(fpath)}...", end='', flush=True)
            with h5py.File(fpath, 'r') as f_src:
                if 'dataset/train/inputs' not in f_src: # 加上 /train
                    print(" -> Skipped (Empty)")
                    continue

                src_inputs = f_src['dataset/train/inputs']
                src_targets = f_src['dataset/train/targets']
                n_file = src_inputs.shape[0]

                end_idx = global_idx + n_file
                dset_inputs[global_idx:end_idx] = src_inputs[:]
                dset_targets[global_idx:end_idx] = src_targets[:]
                global_idx = end_idx

                print(f" -> Done ({n_file} samples)")

        print(f"✅ Merge Complete! Final file: {output_filepath} (Time: {time.time() - start_time:.2f}s)\n")

    # 合并成功后，可选清理碎片文件
    for fpath in part_files + [meta_file]:
        os.remove(fpath)
    print("🧹 Cleaned up temporary chunk files.")


def main():
    parser = argparse.ArgumentParser(description="Experiment Orchestrator for L5PC simulations.")
    parser.add_argument("--script", type=str, required=True,
                        help="要运行的实验脚本路径 (例如: experiments/run_exp_psp_pairs.py)")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数")
    parser.add_argument("--out_dir", type=str, required=True, help="数据输出目录")
    parser.add_argument("--merge_only", action="store_true", help="仅执行合并操作（跳过仿真）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_only:
        fast_merge(args.out_dir)
        return

    print(f"🚀 Starting Orchestrator for script: {args.script}")
    print(f"📂 Output Directory: {args.out_dir}")
    print(f"⚡ Launching {args.workers} parallel workers...\n")

    processes = []
    start_time = time.time()

    # 拉起所有子进程
    for job_id in range(args.workers):
        # 核心修复点：把字符串切分成列表
        script_args = shlex.split(args.script)

        cmd = [sys.executable] + script_args + [
            "--job_id", str(job_id),
            "--total_jobs", str(args.workers),
            "--out_dir", args.out_dir
        ]

        # 将子进程的输出重定向到日志文件，避免终端被打印刷屏
        log_file = open(os.path.join(args.out_dir, f"worker_{job_id}.log"), "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((job_id, p, log_file))

    # 监控进程状态
    try:
        while processes:
            for job_id, p, log_file in processes[:]:
                retcode = p.poll()
                if retcode is not None:  # 进程结束
                    log_file.close()
                    processes.remove((job_id, p, log_file))
                    if retcode == 0:
                        print(f"🟢 Worker {job_id} finished successfully.")
                    else:
                        print(f"🔴 Worker {job_id} failed with return code {retcode}. Check worker_{job_id}.log!")
            time.sleep(1)  # 每秒轮询一次
    except KeyboardInterrupt:
        print("\n⚠️ Orchestrator interrupted! Terminating all workers...")
        for _, p, log_file in processes:
            p.terminate()
            log_file.close()
        sys.exit(1)

    print(f"\n🏁 All workers finished in {time.time() - start_time:.2f} seconds.")

    # 自动触发无缝合并
    print("🔄 Automatically triggering data merge...")
    fast_merge(args.out_dir)


if __name__ == "__main__":
    main()