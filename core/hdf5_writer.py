import os
import h5py
import hdf5plugin
import numpy as np


class HDF5Writer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.f = None
        self.dsets = {}
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    def initialize(self, metadata, total_sim_steps, num_synapses, split="train"):
        """
        :param split: 数据集划分，默认为 "train"，也可设为 "val" 或 "test"
        """
        self.f = h5py.File(self.filepath, 'a')

        # 1. 严格按照规范写入根属性 (.attrs)
        if 'neuron_name' not in self.f.attrs:
            self.f.attrs['neuron_name'] = metadata.get("neuron_name", "L5PC")
            self.f.attrs['dt'] = float(metadata.get("dt", 0.1))
            self.f.attrs['total_steps'] = int(total_sim_steps)
            self.f.attrs['num_synapses'] = int(num_synapses)
            self.f.attrs['target_v_mV'] = float(metadata.get("target_v_mV", np.nan))
            print(f"Initialized new HDF5 file: {self.filepath} (Target V: {self.f.attrs['target_v_mV']} mV)")

        # 2. 严格按照规范写入静态组 (/static_info)
        if 'static_info' not in self.f:
            static_grp = self.f.create_group('static_info')
            keys_to_write = ['num_subunits', 'parent_indices', 'input_map', 'synapse_types', 'alpha', 'g_exc', 'g_inh']
            for k in keys_to_write:
                if k in metadata:
                    static_grp.create_dataset(k, data=metadata[k])

        # 3. 启用 Blosc lz4 等级5 + SHUFFLE 压缩
        comp_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

        # 4. 创建指定的 split 层级结构 (e.g., /dataset/train)
        grp_path = f'dataset/{split}'
        if grp_path not in self.f:
            dset_grp = self.f.require_group(grp_path)

            # 严格按照规范的 chunking 策略: (1, Time_Steps, Num_Synapses)
            dset_grp.create_dataset(
                'inputs',
                shape=(0, total_sim_steps, num_synapses),
                maxshape=(None, total_sim_steps, num_synapses),
                dtype='uint8',
                chunks=(1, total_sim_steps, num_synapses),
                **comp_kwargs
            )
            dset_grp.create_dataset(
                'targets',
                shape=(0, total_sim_steps, 1),
                maxshape=(None, total_sim_steps, 1),
                dtype='float32',
                chunks=(1, total_sim_steps, 1),
                **comp_kwargs
            )

        self.dsets['inputs'] = self.f[grp_path]['inputs']
        self.dsets['targets'] = self.f[grp_path]['targets']

    def append(self, inputs_batch, targets_batch):
        if not self.f:
            raise RuntimeError("HDF5Writer is not initialized.")

        current_size = self.dsets['inputs'].shape[0]
        new_size = current_size + inputs_batch.shape[0]

        self.dsets['inputs'].resize(new_size, axis=0)
        self.dsets['targets'].resize(new_size, axis=0)

        self.dsets['inputs'][current_size:new_size] = inputs_batch
        self.dsets['targets'][current_size:new_size] = targets_batch
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()