#!/usr/bin/env python3
"""
从 results.mat 提取二维嵌入坐标，根据 pt_root_id 匹配细胞类型，
分别按 cell_type、mtype、meso_type 着色，生成三个散点图。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import loadmat
import h5py
import argparse
import os

# ---------- 颜色生成 ----------
def generate_distinct_colors(n, seed=42):
    np.random.seed(seed)
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.75 if i % 2 == 0 else 0.95
        value = 0.85 if i % 3 == 0 else (0.70 if i % 3 == 1 else 0.55)
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        colors.append((*rgb, 1.0))
    return colors

# ---------- .mat 读取 ----------
def load_mat_auto(filepath):
    try:
        data = loadmat(filepath, simplify_cells=False)
        return data
    except NotImplementedError:
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
    v = data[key]
    if isinstance(v, list):
        return v
    v = np.asarray(v, dtype=object)
    return [v.flat[i] for i in range(n)]

# ---------- 核心绘图 ----------
def plot_2d_embedding_by_type(results_path, prepared_path, info_csv,
                              method, output_prefix):
    # 加载数据
    print(f"加载 results: {results_path}")
    res_data = load_mat_auto(results_path)
    print(f"加载 prepared: {prepared_path}")
    prep_data = load_mat_auto(prepared_path)

    # 获取维度 2 的坐标
    dims = np.asarray(prep_data['dims']).flatten().astype(int)
    if 2 not in dims:
        raise ValueError("数据中没有嵌入维度 2")
    idx = np.where(dims == 2)[0][0]

    method_map = {'Classic': 'classic_coords', 'Huber': 'huber_coords', 'IRLS': 'irls_coords'}
    if method not in method_map:
        raise ValueError("method 必须是 Classic, Huber, IRLS 之一")
    coords_key = method_map[method]
    coords_list = get_cell_array(res_data, coords_key, len(dims))
    X = np.asarray(coords_list[idx], dtype=float)

    if X.ndim == 3:
        X = X.squeeze()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] != 2:
        if X.shape[0] == 2:
            X = X.T
        else:
            raise ValueError(f"坐标形状异常: {X.shape}")

    # ---------- 读取 labels（细胞 ID）并转为字符串 ----------
    labels_raw = prep_data.get('labels', None)
    if labels_raw is None:
        raise ValueError("prepared.mat 中未找到 'labels'")

    labels_flat = []
    def flatten_and_decode(item):
        if isinstance(item, (list, np.ndarray)):
            for sub in item:
                flatten_and_decode(sub)
        else:
            if isinstance(item, bytes):
                item = item.decode('utf-8')
            if isinstance(item, np.generic):
                item = str(item)
            labels_flat.append(str(item))
    flatten_and_decode(labels_raw)
    labels = labels_flat

    if len(labels) != X.shape[0]:
        print(f"警告: labels 长度 {len(labels)} 与坐标行数 {X.shape[0]} 不一致，使用默认 ID")
        labels = [f"ID_{i}" for i in range(X.shape[0])]

    # ---------- 读取 CSV，构建 pt_root_id -> 类型映射 ----------
    print(f"读取细胞类型信息: {info_csv}")
    df_info = pd.read_csv(info_csv)

    # 检查必要列
    required_cols = ['pt_root_id', 'cell_type', 'mtype', 'meso_type']
    for col in required_cols:
        if col not in df_info.columns:
            raise ValueError(f"CSV 中缺少列: {col}")

    # 构建映射字典
    id_to_types = {}
    for _, row in df_info.iterrows():
        root_id = str(row['pt_root_id'])
        ct = row['cell_type'] if pd.notna(row['cell_type']) else 'unknown'
        mt = row['mtype'] if pd.notna(row['mtype']) else 'unknown'
        ms = row['meso_type'] if pd.notna(row['meso_type']) else 'unknown'
        id_to_types[root_id] = {'cell_type': str(ct), 'mtype': str(mt), 'meso_type': str(ms)}

    # 为每个 label 分配类型
    type_dicts = []
    for lab in labels:
        type_dicts.append(id_to_types.get(lab, {'cell_type': 'unknown', 'mtype': 'unknown', 'meso_type': 'unknown'}))

    # ---------- 绘制三种分类的图 ----------
    type_keys = ['cell_type', 'mtype', 'meso_type']
    for tkey in type_keys:
        # 收集该分类的所有值
        types = [d[tkey] for d in type_dicts]
        unique_types = list(dict.fromkeys(types))
        # 将 'unknown' 放到最后
        if 'unknown' in unique_types:
            unique_types.remove('unknown')
            unique_types.append('unknown')

        colors = generate_distinct_colors(len(unique_types))
        type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        for t in unique_types:
            mask = [typ == t for typ in types]
            xs = X[mask, 0]
            ys = X[mask, 1]
            ax.scatter(xs, ys, color=type_to_color[t], label=t, s=10, alpha=0.8)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'2D Embedding ({method} Sammon MDS)\nColored by {tkey}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title=tkey)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = f"{output_prefix}_{method}_{tkey}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 图已保存: {out_file}")

# ---------- 命令行 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--method', required=True, choices=['Classic','Huber','IRLS'])
    parser.add_argument('--info-csv', required=True,
                        help='v1718_cell_info.csv 路径')
    parser.add_argument('--output-prefix', default='embedding',
                        help='输出文件前缀，实际文件名为 前缀_方法_分类.png')
    args = parser.parse_args()

    plot_2d_embedding_by_type(args.results, args.prepared, args.info_csv,
                              args.method, args.output_prefix)