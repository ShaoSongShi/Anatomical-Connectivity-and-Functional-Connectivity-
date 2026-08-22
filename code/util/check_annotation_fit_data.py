import scipy.io as sio
import pandas as pd
import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat
import h5py
import sys


def load_mat_auto(filepath):
    """自动加载 .mat 文件（支持 v7 和 v7.3）"""
    try:
        data = loadmat(filepath, simplify_cells=False)
        return data
    except NotImplementedError:
        # v7.3 格式
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if key.startswith('#'):
                    continue
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    val = item[()]
                    if hasattr(val, 'dtype') and h5py.check_dtype(ref=val.dtype):
                        cell_data = []
                        for ref in val.flat:
                            cell_data.append(np.array(f[ref][()]).T)
                        data[key] = cell_data
                    else:
                        data[key] = val if val.shape == () else np.array(val).T
                elif isinstance(item, h5py.Group):
                    cell_data = []
                    subkeys = sorted(item.keys(),
                                     key=lambda x: int(x) if x.isdigit() else float('inf'))
                    for subkey in subkeys:
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            arr = subitem[()]
                            if hasattr(arr, 'dtype') and h5py.check_dtype(ref=arr.dtype):
                                ref = arr.flat[0] if arr.size > 0 else None
                                if ref is not None and ref in f:
                                    cell_data.append(np.array(f[ref][()]).T)
                                else:
                                    cell_data.append(np.array(arr))
                            else:
                                cell_data.append(np.array(arr).T)
                    data[key] = cell_data
        return data

# 加载您的 prepared.mat
prepared_path = '/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_method/fly_aggregated_by_celltype_normalized_nm_2_10_prepared.mat'
print(f"加载 prepared: {prepared_path}")
prep_data = load_mat_auto(prepared_path)
labels = prep_data.get('labels', None)
if labels is not None:
    # 处理从 v7.3 读取的 cell 数组（列表，每个元素为数值数组）
    if isinstance(labels, list) and len(labels) > 0:
        # 检测第一个元素是否为数值类型的 numpy 数组
        first = labels[0]
        if isinstance(first, np.ndarray) and first.dtype.kind in 'iu':
            decoded_labels = []
            for arr in labels:
                # 将一维或二维的 ASCII 码数组转为字符串
                if arr.ndim == 1:
                    chars = [chr(c) for c in arr if c != 0]
                elif arr.ndim == 2:
                    # 可能形状为 (1, N) 或 (N, 1)
                    if arr.shape[0] == 1:
                        chars = [chr(c) for c in arr[0] if c != 0]
                    elif arr.shape[1] == 1:
                        chars = [chr(c) for c in arr[:, 0] if c != 0]
                    else:
                        chars = [str(arr)]
                else:
                    chars = [str(arr)]
                decoded_labels.append(''.join(chars))
            labels = decoded_labels
        else:
            # 其他情况直接转为字符串
            labels = [str(l) for l in labels]
    elif isinstance(labels, np.ndarray):
        # 若为 numpy 数组，展平并转为字符串
        labels = [str(l) for l in labels.ravel()]
    else:
        labels = [str(labels)]


# 读取 CSV
anno = pd.read_csv('/home/wangzezhen/ShenRuihong/corr_principal_validation/flywire_annotations-2.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv')

# 检查列
print(anno.columns.tolist())

# 构建 cell_type → superclass 映射（去重）
mapping = anno.groupby('type')['cell_class'].first().to_dict()
print(f"映射了 {len(mapping)} 个细胞类型")

# 检查您的标签中有多少能匹配到 anno['cell_type']
matched = set(labels) & set(anno['type'].unique())
# print(matched)
print(f"可匹配的类型数type：{len(matched)} / {len(set(labels))}")

matched_cell_class = set(labels) & set(anno['cell_class'].unique())
print(f"可匹配的类型数cell_class：{len(matched_cell_class)} / {len(set(labels))}")