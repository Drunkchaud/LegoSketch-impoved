import os.path

import numpy as np
import torch
import os

project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def convert_np2tensor(device, index_np, weight_np):
    support_x = torch.tensor(index_np, device=device).view(-1, 1)
    support_y = torch.tensor(weight_np, device=device).view(-1)
    if torch.is_floating_point(support_y):
        support_y = support_y.round().type(torch.int32)
    query_x = support_x.clone()
    query_y = support_y.clone().view(-1, 1)
    return support_x, support_y, query_x, query_y


def load_synthetic_0_5_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/0.5_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_0_7_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/0.7_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_0_9_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/0.9_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_1_1_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/1.1_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_1_3_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/1.3_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


def load_synthetic_1_5_generator(device, chunksize=10000000):
    dataset_path = os.path.join(project_path, "EvalModel/EvalDataset/Datasets/synthetic/1.5_zipf/0.npz")
    npz_file = np.load(dataset_path)
    yield convert_np2tensor(device, np.arange(npz_file['support_y'].shape[0]), npz_file['support_y'])


dataset_dic = {
    "synthetic_0_5": load_synthetic_0_5_generator,
    "synthetic_0_7": load_synthetic_0_7_generator,
    "synthetic_0_9": load_synthetic_0_9_generator,
    "synthetic_1_1": load_synthetic_1_1_generator,
    "synthetic_1_3": load_synthetic_1_3_generator,
    "synthetic_1_5": load_synthetic_1_5_generator,
}


def register_frequency_npz(name, npz_path):
    """Register a full frequency table; yield one batch for the existing evaluator.

    NPZ needs int64 items and positive integer freqs. chunksize is intentionally
    ignored: current Cluster evaluation only supports one write batch.
    """
    def load_generator(device, chunksize=10000000):
        with np.load(npz_path, allow_pickle=False) as data:
            items = data['items']
            freqs = data['freqs']
            if items.ndim != 1 or freqs.ndim != 1 or items.shape != freqs.shape:
                raise ValueError('Expected equal-length one-dimensional items/freqs')
            if items.dtype != np.int64 or not np.issubdtype(freqs.dtype, np.integer):
                raise ValueError('Expected int64 items and integer frequencies')
            if not len(items) or np.any(freqs <= 0):
                raise ValueError('Expected nonempty, positive frequencies')
            yield convert_np2tensor(device, items, freqs)
    dataset_dic[name] = load_generator


# Override this directory when copying the repository to a different layout.
real_dataset_dir = os.environ.get(
    'LEGO_REAL_DATA_DIR', os.path.abspath(os.path.join(project_path, '..', '..', '数据集', 'processed')))
for _name in ('kosarak', 'webdocs'):
    register_frequency_npz(_name, os.path.join(real_dataset_dir, _name + '.npz'))
register_frequency_npz('mawi_src_ip', os.path.join(real_dataset_dir, '202401011400_src-ip.npz'))
