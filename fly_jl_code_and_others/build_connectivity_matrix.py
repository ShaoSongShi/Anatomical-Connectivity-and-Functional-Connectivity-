'''
1. 加载 traced-neurons.csv 和 Supplemental_file5_hemibrain_meta.csv，通过 bodyId 合并，筛选出 instance 和 type 同时匹配的神经元。
2. 加载 traced-roi-connections.csv，仅保留这些有效神经元之间的连接。
3. 按 bodyId_pre 和 bodyId_post 汇总连接权重（多个 ROI 或重复行会累加weight）。
4. 生成一个方阵 DataFrame，行/列索引为所有有效 bodyId，单元格值为连接权重（缺失为 0）。
'''
import pandas as pd
import numpy as np

traced_neurons_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/data_Janelia_FlyEM_Hemibrain/traced-neurons.csv"
traced_conn_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/data_Janelia_FlyEM_Hemibrain/traced-roi-connections.csv"
meta_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/flywire_annotations-2.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv"  

# 读取数据
traced = pd.read_csv(traced_neurons_path, dtype=str)
meta = pd.read_csv(meta_path, dtype=str)

# 去除所有字符串列的前后空格
for df in (traced, meta):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

# 确保必要的列存在
required_cols = ['bodyId', 'instance', 'type']
for col in required_cols:
    if col not in traced.columns:
        raise ValueError(f"traced-neurons.csv 缺少列: {col}")
    if col not in meta.columns:
        raise ValueError(f"meta 文件缺少列: {col}")

print(f"traced 神经元数: {len(traced)}")
print(f"meta 神经元数: {len(meta)}")

# ===== 2. 合并并筛选同时匹配 instance 和 type 的神经元 =====
merged = traced.merge(meta, on='bodyId', suffixes=('_traced', '_meta'), how='inner')
print(f"通过 bodyId 匹配上的神经元数: {len(merged)}")

# 筛选 instance 和 type 同时匹配
valid = merged[
    (merged['instance_traced'] == merged['instance_meta']) &
    (merged['type_traced'] == merged['type_meta'])
]
valid_bodyIds = set(valid['bodyId'].astype(str))
print(f"instance 和 type 同时匹配的神经元数: {len(valid_bodyIds)}")

if len(valid_bodyIds) == 0:
    print("没有符合条件的神经元，无法构建连接矩阵。")
    exit()

# ===== 3. 读取连接数据 =====
conn = pd.read_csv(traced_conn_path)
# 确保 bodyId 列是字符串，与 valid_bodyIds 类型一致
conn['bodyId_pre'] = conn['bodyId_pre'].astype(str)
conn['bodyId_post'] = conn['bodyId_post'].astype(str)

# 过滤连接：仅保留 pre 和 post 都在有效集合中的行
mask_pre = conn['bodyId_pre'].isin(valid_bodyIds)
mask_post = conn['bodyId_post'].isin(valid_bodyIds)
filtered_conn = conn[mask_pre & mask_post]
print(f"原始连接数: {len(conn)}，过滤后连接数: {len(filtered_conn)}")

if len(filtered_conn) == 0:
    print("过滤后无连接，生成全零矩阵。")
    # 生成全零方阵
    body_ids_sorted = sorted(valid_bodyIds, key=int)
    matrix = pd.DataFrame(0, index=body_ids_sorted, columns=body_ids_sorted)
    print("矩阵形状:", matrix.shape)
    # 保存
    matrix.to_csv("connectivity_matrix.csv")
    print("已保存全零矩阵到 connectivity_matrix.csv")
    exit()

# ===== 4. 汇总连接权重（按 pre 和 post 分组求和） =====
grouped = filtered_conn.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
print(f"去重后的连接对数量: {len(grouped)}") # 合并后的唯一连接对数，通过分组求和，weight 变为该对神经元之间的突触总数

# ===== 5. 构建方阵 DataFrame =====
body_ids_sorted = sorted(valid_bodyIds, key=int)  # 按数字排序
# 创建索引和列
matrix = pd.DataFrame(0, index=body_ids_sorted, columns=body_ids_sorted, dtype=float)

# 填入权重
for _, row in grouped.iterrows():
    pre = row['bodyId_pre']
    post = row['bodyId_post']
    w = row['weight']
    matrix.loc[pre, post] = w

# 可选：如果希望对角线为零（一般连接矩阵对角线无自连，但如果有自连保留）
# 矩阵已是方阵

print("连接矩阵形状:", matrix.shape)
print("非零元素个数:", (matrix > 0).sum().sum())

# ===== 6. 保存为 CSV =====
output_path = "connectivity_matrix.csv"
matrix.to_csv(output_path)
print(f"连接矩阵已保存至: {output_path}")

# 可选：保存有效神经元列表以备后用
valid_neurons = valid[['bodyId', 'instance_traced', 'type_traced']].copy()
valid_neurons.to_csv("valid_neurons.csv", index=False)
print("有效神经元列表已保存至 valid_neurons.csv")