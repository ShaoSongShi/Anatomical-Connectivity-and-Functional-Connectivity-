#!/usr/bin/env python3
"""
extract_reconstructed_corr.py

从已完成的 Sammon MDS 结果中提取指定方法在指定维度的重构相关性矩阵 C2，
并保存为 CSV 文件。

用法示例：
    python extract_reconstructed_corr.py \
        --results results.mat \
        --prepared prepared.mat \
        --method Classic \
        --dim 5 \
        --output C2_Classic_dim5.csv
['Classic', 'Huber', 'IRLS']
若需同时保存原始相关矩阵，添加 --save-original
"""

import argparse
import numpy as np
import pandas as pd
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


def get_cell_array(data, key, n):
    """将 .mat cell 转为 Python list"""
    v = data[key]
    if isinstance(v, list):
        return v
    v = np.asarray(v, dtype=object)
    return [v.flat[i] for i in range(n)]


def extract_reconstructed_corr(results_path, prepared_path, method, dim,
                               output_csv, save_original=False):
    """
    提取指定方法在指定维度的重构相关矩阵并保存为 CSV。
    method: 'Classic', 'Huber', 'IRLS'（大小写敏感）
    """
    # 加载数据
    print(f"加载 results: {results_path}")
    res_data = load_mat_auto(results_path)
    print(f"加载 prepared: {prepared_path}")
    prep_data = load_mat_auto(prepared_path)

    # 检查 method 对应的字段名
    method_map = {
        'Classic': ('classic_coords', 'classic_stress'),
        'Huber': ('huber_coords', 'huber_stress'),
        'IRLS': ('irls_coords', 'irls_stress'),
    }
    if method not in method_map:
        raise ValueError(f"method 必须是 {list(method_map.keys())} 之一，得到 {method}")
    coords_field, _ = method_map[method]

    # 读取 prepared 中的基本参数
    dims = np.asarray(prep_data['dims']).flatten().astype(int)
    if dim not in dims:
        raise ValueError(f"维度 {dim} 不在 dims 列表 {list(dims)} 中")
    idx = np.where(dims == dim)[0][0]

    mu_vec = np.asarray(prep_data['mu_vec']).flatten()
    epsilon = float(np.asarray(prep_data['epsilon']).flatten()[0])
    mu = mu_vec[idx]
    Corr = np.asarray(prep_data['Corr'], dtype=float)
    labels = prep_data.get('labels', None)
    if labels is not None:
        # 处理 v7.3 读取的 cell 数组（列表），每个元素是一个 numpy 数组（标量）
        if isinstance(labels, list) and len(labels) > 0:
            first = labels[0]
            # 如果第一个元素是 numpy 数组且为整数类型
            if isinstance(first, np.ndarray) and np.issubdtype(first.dtype, np.integer):
                # 提取每个数组的第一个元素（因为每个数组只包含一个数值），转为字符串
                labels = [str(int(arr.flat[0])) for arr in labels]
            else:
                # 其他情况（例如已经是字符串数组）
                labels = [str(l) for l in labels]
        elif isinstance(labels, np.ndarray):
            # 如果 labels 本身是 numpy 数组（v7 格式），展平转为字符串
            labels = [str(l) for l in labels.ravel()]
        else:
            labels = [str(labels)]

        # 长度校验
        if len(labels) != Corr.shape[0]:
            print(f"警告: labels 长度 {len(labels)} 与矩阵维度 {Corr.shape[0]} 不匹配，使用默认索引")
            labels = [f"Region_{i}" for i in range(Corr.shape[0])]
    else:
        labels = [f"Region_{i}" for i in range(Corr.shape[0])]
    # 读取坐标
    coords_data = res_data[coords_field]
    if isinstance(coords_data, np.ndarray) and coords_data.ndim == 2:
        # 直接使用该矩阵（单个维度的情况）
        X = np.asarray(coords_data, dtype=float)
        # 确保行数匹配
        if X.shape[0] != Corr.shape[0]:
            X = X.T
    else:
        # cell 数组（多个维度的情况）
        coords_list = get_cell_array(res_data, coords_field, len(dims))
        X = np.asarray(coords_list[idx], dtype=float)
        # 处理可能的转置
        if X.ndim == 3:
            X = X.squeeze()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != Corr.shape[0]:
            X = X.T

    # 计算重构距离矩阵 D2
    D2 = squareform(pdist(X, metric='euclidean'))
    np.fill_diagonal(D2, 0.0)

    # ERM 反解得到 C2
    C2 = epsilon ** mu * np.power(D2 ** 2 + epsilon ** 2, -mu / 2.0)

    # 创建 DataFrame
    df_C2 = pd.DataFrame(C2, index=labels, columns=labels)
    df_C2.to_csv(output_csv)
    print(f"重构相关矩阵 C2 已保存至: {output_csv}")

    if save_original:
        df_Corr = pd.DataFrame(Corr, index=labels, columns=labels)
        orig_csv = output_csv.replace('.csv', '_original.csv')
        df_Corr.to_csv(orig_csv)
        print(f"原始相关矩阵 Corr 已保存至: {orig_csv}")

    # 可选：返回 df_C2
    return df_C2


def main():
    ap = argparse.ArgumentParser(description="提取指定方法维度的重构相关矩阵")
    ap.add_argument('--results', required=True, help='results.mat 文件路径')
    ap.add_argument('--prepared', required=True, help='prepared.mat 文件路径')
    ap.add_argument('--method', required=True, choices=['Classic', 'Huber', 'IRLS'],
                    help='MDS 方法名称')
    ap.add_argument('--dim', type=int, required=True, help='嵌入维度')
    ap.add_argument('--output', required=True, help='输出 CSV 文件路径')
    ap.add_argument('--save-original', action='store_true',
                    help='同时保存原始相关矩阵')
    args = ap.parse_args()

    extract_reconstructed_corr(
        results_path=args.results,
        prepared_path=args.prepared,
        method=args.method,
        dim=args.dim,
        output_csv=args.output,
        save_original=args.save_original
    )


if __name__ == '__main__':
    main()