import pandas as pd

# 文件路径（请根据实际位置修改）
traced_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/data_Janelia_FlyEM_Hemibrain/traced-neurons.csv"
meta_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/flywire_annotations-2.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv"  # 若不在当前目录，请提供完整路径

# 读取 CSV，将所有列作为字符串读入，避免类型问题
traced = pd.read_csv(traced_path, dtype=str)
meta = pd.read_csv(meta_path, dtype=str)

print("traced-neurons.csv 行数:", len(traced))
print("meta 文件行数:", len(meta))
print("traced 列名:", traced.columns.tolist())
print("meta 列名:", meta.columns.tolist())

# 检查 bodyId 是否存在
if 'bodyId' not in traced.columns or 'bodyId' not in meta.columns:
    raise ValueError("bodyId 列不存在于其中一个文件")

# 去除可能的前后空格
for col in ['bodyId', 'instance', 'type']:
    if col in traced.columns:
        traced[col] = traced[col].str.strip()
    if col in meta.columns:
        meta[col] = meta[col].str.strip()

# 以 bodyId 为键进行内连接
merged = traced.merge(meta, on='bodyId', how='inner', suffixes=('_traced', '_meta'))
print("\n通过 bodyId 匹配上的神经元数量:", len(merged))

# 比较 instance 和 type 字段
if 'instance_traced' in merged.columns and 'instance_meta' in merged.columns:
    instance_match = (merged['instance_traced'] == merged['instance_meta']).sum()
else:
    instance_match = None

if 'type_traced' in merged.columns and 'type_meta' in merged.columns:
    type_match = (merged['type_traced'] == merged['type_meta']).sum()
else:
    type_match = None

# 两个字段同时匹配
if instance_match is not None and type_match is not None:
    both_match = ((merged['instance_traced'] == merged['instance_meta']) & 
                  (merged['type_traced'] == merged['type_meta'])).sum()
else:
    both_match = None

# 输出统计
print("\n--- 匹配统计 ---")
print(f"bodyId 匹配数量: {len(merged)}")
if instance_match is not None:
    print(f"instance 匹配数量: {instance_match} ({instance_match/len(merged)*100:.2f}%)")
if type_match is not None:
    print(f"type 匹配数量: {type_match} ({type_match/len(merged)*100:.2f}%)")
if both_match is not None:
    print(f"instance 和 type 同时匹配: {both_match} ({both_match/len(merged)*100:.2f}%)")

# 展示几个匹配和不匹配的示例（如果有）
print("\n--- 示例（前5行匹配的数据）---")
print(merged.head(5).to_string())

if len(merged) > 0:
    # 展示 instance 不匹配的示例
    if instance_match is not None and instance_match < len(merged):
        mismatch_inst = merged[merged['instance_traced'] != merged['instance_meta']]
        print("\n--- instance 不匹配的示例（前3行）---")
        print(mismatch_inst[['bodyId', 'instance_traced', 'instance_meta']].head(3).to_string())
    
    # 展示 type 不匹配的示例
    if type_match is not None and type_match < len(merged):
        mismatch_type = merged[merged['type_traced'] != merged['type_meta']]
        print("\n--- type 不匹配的示例（前3行）---")
        print(mismatch_type[['bodyId', 'type_traced', 'type_meta']].head(3).to_string())

# 同时输出未匹配的 bodyId（即只在 traced 或 meta 中出现的）
traced_only = set(traced['bodyId']) - set(meta['bodyId'])
meta_only = set(meta['bodyId']) - set(traced['bodyId'])
print(f"\n仅在 traced 中出现的 bodyId 数量: {len(traced_only)}")
print(f"仅在 meta 中出现的 bodyId 数量: {len(meta_only)}")