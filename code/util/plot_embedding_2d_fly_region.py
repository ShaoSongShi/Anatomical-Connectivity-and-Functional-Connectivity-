#!/usr/bin/env python3
"""
从 results.mat 提取二维嵌入坐标，用 prepared.mat 的 labels 按脑区着色，
颜色生成规则与热图脚本完全一致（黄金比例 HSV）。
无任何对齐检测，直接使用 labels。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import loadmat
import h5py
import argparse
import sys

# ---------- 颜色生成函数（与热图脚本完全相同）----------
def generate_distinct_colors(n, seed=42):
    """生成 n 个在 HSV 空间中均匀分布且饱和度/明度有变化的颜色（带透明度）。"""
    np.random.seed(seed)
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0          # 黄金比例产生均匀色相
        saturation = 0.75 if i % 2 == 0 else 0.95
        value = 0.85 if i % 3 == 0 else (0.70 if i % 3 == 1 else 0.55)
        rgb = mcolors.hsv_to_rgb([hue, saturation, value])
        colors.append((*rgb, 1.0))
    return colors


# ---------- .mat 文件读取（兼容 v7 和 v7.3）----------
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
    """从 .mat 的 cell 数组中提取 Python list。"""
    v = data[key]
    if isinstance(v, list):
        return v
    v = np.asarray(v, dtype=object)
    return [v.flat[i] for i in range(n)]


# ---------- 主绘图函数 ----------
def plot_2d_embedding(results_path, prepared_path, method, output_fig='embedding_2d.png'):
    # 加载数据
    print(f"加载 results: {results_path}")
    res_data = load_mat_auto(results_path)
    print(f"加载 prepared: {prepared_path}")
    prep_data = load_mat_auto(prepared_path)

    # 获取所有嵌入维度
    dims = np.asarray(prep_data['dims']).flatten().astype(int)
    if 2 not in dims:
        raise ValueError("数据中没有嵌入维度 2，无法绘制二维散点图。")
    idx = np.where(dims == 2)[0][0]   # 维度 2 在 dims 中的位置

    # 获取坐标
    method_map = {
        'Classic': 'classic_coords',
        'Huber': 'huber_coords',
        'IRLS': 'irls_coords'
    }
    if method not in method_map:
        raise ValueError(f"method 必须是 Classic, Huber, IRLS 之一")
    coords_field = method_map[method]
    # 读取坐标
    coords_data = res_data[coords_field]
    if isinstance(coords_data, np.ndarray) and coords_data.ndim == 2:
        # 直接使用该矩阵（单个维度的情况）
        X = np.asarray(coords_data, dtype=float)
        # # 确保行数匹配
        # if X.shape[0] != Corr.shape[0]:
        #     X = X.T
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
    # coords_key = method_map[method]
    # coords_list = get_cell_array(res_data, coords_key, len(dims))
    # X = np.asarray(coords_list[idx], dtype=float)

    # # 修正可能的数据形状（转置或挤压）
    # if X.ndim == 3:
    #     X = X.squeeze()
    # if X.ndim == 1:
    #     X = X.reshape(-1, 1)
    # # 确保是 N×2
    # if X.shape[1] != 2:
    #     if X.shape[0] == 2:
    #         X = X.T
    #     else:
    #         raise ValueError(f"坐标矩阵形状为 {X.shape}，无法解析为 N×2")

    # 获取 labels（直接使用，不做任何检测）
    labels = prep_data.get('labels', None)
    if labels is None:
        raise ValueError("prepared.mat 中未找到 'labels' 变量")
    # 将 labels 展平为一维列表，并确保每个元素是字符串（若为 bytes 则解码）
    # 正确提取 labels：若是 cell 数组，每个元素是字符串（或 bytes）
    labels_raw = prep_data['labels']
    if isinstance(labels_raw, np.ndarray):
        # 如果是 ndarray，可能是二维或一维，统一展平
        labels_raw = labels_raw.flatten()
    elif not isinstance(labels_raw, list):
        labels_raw = [labels_raw]

    labels = []
    for item in labels_raw:
        if isinstance(item, bytes):
            labels.append(item.decode('utf-8'))
        elif isinstance(item, str):
            labels.append(item)
        else:
            labels.append(str(item))

    # 生成颜色映射（与热图脚本一致）
    unique_regions = list(dict.fromkeys(labels))   # 保持首次出现顺序
    region_colors = generate_distinct_colors(len(unique_regions))
    region_to_color = {region: region_colors[i] for i, region in enumerate(unique_regions)}

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 按脑区分别绘制，便于添加图例
    for region in unique_regions:
        mask = [l == region for l in labels]
        xs = X[mask, 0]
        ys = X[mask, 1]
        ax.scatter(xs, ys, color=region_to_color[region], label=region, s=10, alpha=0.8)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'2D Embedding ({method} Sammon MDS)\nColored by Brain Region')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Regions')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 二维散点图已保存: {output_fig}")


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取指定方法在维度2的嵌入坐标并绘制散点图")
    parser.add_argument('--results', required=True, help='results.mat 文件路径')
    parser.add_argument('--prepared', required=True, help='prepared.mat 文件路径')
    parser.add_argument('--method', required=True, choices=['Classic','Huber','IRLS'],
                        help='MDS 方法名称')
    parser.add_argument('--output', default='embedding_2d.png',
                        help='输出图片文件名 (默认 embedding_2d.png)')
    args = parser.parse_args()

    plot_2d_embedding(args.results, args.prepared, args.method, args.output)