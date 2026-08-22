#!/usr/bin/env python3
"""
从 results.mat 提取二维嵌入坐标，并按指定的层级标签着色。
颜色生成规则与热图脚本一致（黄金比例 HSV）。
标签来源于 connectivity_matrix 的索引（cell_type），再映射到各层级。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import loadmat
import h5py
import argparse
import os
import sys

# ---------- 颜色生成函数 ----------
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
    v = data[key]
    if isinstance(v, list):
        return v
    v = np.asarray(v, dtype=object)
    return [v.flat[i] for i in range(n)]


def build_celltype_hierarchy_mapping(fw_annot_path, meta_path):
    """
    构建从 cell_type 到各层级标签的映射字典。
    映射逻辑：
      1. 从 meta (Supplemental_file5) 读取 bodyId → type (或 morphology_type)
      2. 从 fw (Supplemental_file1) 读取 hemibrain_type → cell_type 及高级标签
      3. 联合构建 bodyId → cell_type，再汇总为 cell_type → 高级标签（取第一个非空值）
    """
    # 读取数据
    meta = pd.read_csv(meta_path, dtype=str)
    fw = pd.read_csv(fw_annot_path, sep='\t', dtype=str)

    # 去除空格
    for df in (meta, fw):
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

    # 1. 构建 bodyId → type_for_match (优先 morphology_type，回退 type)
    meta['type_for_match'] = meta['morphology_type'].fillna(meta['type']).fillna('')
    meta_filtered = meta[meta['type_for_match'] != '']
    body_to_type = dict(zip(meta_filtered['bodyId'].astype(str), meta_filtered['type_for_match']))

    # 2. 构建 hemibrain_type → cell_type 映射（处理复合键，与之前代码一致）
    fw_mapping = fw[['hemibrain_type', 'cell_type']].dropna()
    fw_mapping = fw_mapping[(fw_mapping['hemibrain_type'] != '') & (fw_mapping['cell_type'] != '')]

    # 拆解逗号分隔和括号分组
    expanded_rows = []
    for _, row in fw_mapping.iterrows():
        raw = row['hemibrain_type']
        cell_type = row['cell_type']
        # 先尝试匹配括号分组加后缀
        import re
        pattern = r'\(([^)]+)\)(\w*)'
        match = re.fullmatch(pattern, raw)
        if match:
            types_str = match.group(1)
            for t in types_str.split(','):
                t = t.strip()
                if t:
                    expanded_rows.append({'hemibrain_type': t, 'cell_type': cell_type})
        else:
            # 普通逗号分隔
            for t in raw.split(','):
                t = t.strip()
                if t:
                    expanded_rows.append({'hemibrain_type': t, 'cell_type': cell_type})
    expanded_mapping = pd.DataFrame(expanded_rows).drop_duplicates('hemibrain_type')
    type_to_cell = dict(zip(expanded_mapping['hemibrain_type'], expanded_mapping['cell_type']))

    # 3. 构建 bodyId → cell_type
    body_to_cell = {}
    for body_id, t in body_to_type.items():
        if t in type_to_cell:
            body_to_cell[body_id] = type_to_cell[t]

    # 4. 从 fw 中提取每个 cell_type 对应的各层级标签（取第一个非空值）
    # 注意：fw 中可能有多个 root_id 对应同一 cell_type，但层级标签应该一致
    hierarchy_fields = ['super_class', 'cell_class', 'cell_sub_class',
                        'ito_lee_hemilineage', 'hartenstein_hemilineage', 'morphology_group']
    # 只保留 fw 中 cell_type 非空且能映射到的行
    fw_valid = fw[fw['cell_type'].isin(set(body_to_cell.values()))].copy()
    # 按 cell_type 分组，取第一个非空的各字段值
    celltype_hierarchy = {}
    for ct, group in fw_valid.groupby('cell_type'):
        row = {}
        for field in hierarchy_fields:
            # 取第一个非空值
            val = group[field].dropna()
            if len(val) > 0:
                row[field] = val.iloc[0]
            else:
                row[field] = 'unknown'
        celltype_hierarchy[ct] = row

    return celltype_hierarchy, hierarchy_fields


# ---------- 主绘图函数 ----------
def plot_2d_embedding(results_path, prepared_path, method, matrix_csv_path,
                      fw_annot_path, meta_path, color_by='cell_class',
                      output_fig='embedding_2d.png'):
    # 加载 results 和 prepared（只用于获取 dims 和坐标）
    print(f"加载 results: {results_path}")
    res_data = load_mat_auto(results_path)
    print(f"加载 prepared: {prepared_path}")
    prep_data = load_mat_auto(prepared_path)

    dims = np.asarray(prep_data['dims']).flatten().astype(int)
    if 2 not in dims:
        raise ValueError("数据中没有嵌入维度 2")
    idx = np.where(dims == 2)[0][0]

    method_map = {
        'Classic': 'classic_coords',
        'Huber': 'huber_coords',
        'IRLS': 'irls_coords'
    }
    if method not in method_map:
        raise ValueError(f"method 必须是 Classic, Huber, IRLS 之一")
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
            raise ValueError(f"坐标矩阵形状为 {X.shape}，无法解析为 N×2")
    n_points = X.shape[0]

    # ---- 1. 读取连接矩阵的索引作为细胞类型列表 ----
    conn_df = pd.read_csv(matrix_csv_path, index_col=0)
    cell_types = conn_df.index.tolist()  # 顺序与坐标行对应
    if len(cell_types) != n_points:
        print(f"警告: 连接矩阵索引数量 ({len(cell_types)}) 与坐标行数 ({n_points}) 不匹配，将使用坐标行数截断。")
        cell_types = cell_types[:n_points]

    # ---- 2. 打印 prepared.mat 中的原始 labels（如果存在）----
    if 'labels' in prep_data:
        raw_labels = prep_data['labels']
        try:
            if isinstance(raw_labels, np.ndarray):
                raw_labels = raw_labels.tolist()
            labels_flat = []
            def flatten_and_decode(item):
                if isinstance(item, (list, np.ndarray)):
                    for sub in item:
                        flatten_and_decode(sub)
                else:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    labels_flat.append(str(item))
            flatten_and_decode(raw_labels)
            print(f"原始 prepared labels 前10个: {labels_flat[:10]}")
        except Exception as e:
            print(f"解析原始 labels 出错: {e}")
    else:
        print("prepared.mat 中没有 'labels' 字段")

    # ---- 3. 构建 cell_type → 层级标签映射 ----
    celltype_hierarchy, hierarchy_fields = build_celltype_hierarchy_mapping(fw_annot_path, meta_path)

    if color_by not in hierarchy_fields:
        raise ValueError(f"color_by 必须是以下之一: {hierarchy_fields}")

    # ---- 4. 为每个细胞类型获取对应层级的标签 ----
    labels = []
    for ct in cell_types:
        if ct in celltype_hierarchy:
            label = celltype_hierarchy[ct].get(color_by, 'unknown')
        else:
            label = 'unknown'
        labels.append(label)

    print(f"新构建的标签（{color_by}）前10个: {labels[:10]}")

    # ---- 5. 绘图 ----
    unique_regions = list(dict.fromkeys(labels))  # 保持顺序
    region_colors = generate_distinct_colors(len(unique_regions))
    region_to_color = {region: region_colors[i] for i, region in enumerate(unique_regions)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for region in unique_regions:
        mask = [l == region for l in labels]
        xs = X[mask, 0]
        ys = X[mask, 1]
        ax.scatter(xs, ys, color=region_to_color[region], label=region, s=10, alpha=0.8)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'2D Embedding ({method} Sammon MDS)\nColored by {color_by}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title=color_by)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 二维散点图已保存: {output_fig}")


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='results.mat 文件路径')
    parser.add_argument('--prepared', required=True, help='prepared.mat 文件路径')
    parser.add_argument('--method', required=True, choices=['Classic','Huber','IRLS'])
    parser.add_argument('--matrix_csv', required=True,
                        help='connectivity_matrix_aggregateed_by_more_cell_type.csv 路径')
    parser.add_argument('--fw_annot', default="/home/wangzezhen/ShenRuihong/corr_principal_validation/flywire_annotations-2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
                        help='FlyWire 注释文件路径')
    parser.add_argument('--meta_path', default="/home/wangzezhen/ShenRuihong/corr_principal_validation/flywire_annotations-2.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv",
                        help='Hemibrain meta 文件路径')
    parser.add_argument('--color_by', default='cell_class',
                        choices=['super_class','cell_class','cell_sub_class',
                                 'ito_lee_hemilineage','hartenstein_hemilineage','morphology_group'],
                        help='用于着色的层级字段')
    parser.add_argument('--output', default='embedding_2d.png', help='输出图片文件名')
    args = parser.parse_args()

    plot_2d_embedding(args.results, args.prepared, args.method, args.matrix_csv,
                      args.fw_annot, args.meta_path, args.color_by, args.output)

'''
python /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code/plot_embedding_2d.py --results /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/fly_aggregateed_by_more_cell_type_nm_2_10_results.mat --prepared /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/fly_aggregateed_by_more_cell_type_nm_2_10_prepared.mat --matrix_csv /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/connectivity_matrix_aggregateed_by_more_cell_type.csv --color_by cell_sub_class --method Classic --output embedding_2d_Classic_cell_sub_class_by_more_ct_nm.png
'''