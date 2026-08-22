import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. 读取连接数据bodyId_pre,bodyId_post,roi,weight
# ============================================================
conn_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/data_Janelia_FlyEM_Hemibrain/traced-total-connections.csv"
conn = pd.read_csv(conn_path)  # 列: bodyId_pre, bodyId_post, roi, weight

# （可选）只保留目标脑区内的连接，做"某 ROI 内"的类型级矩阵
# target_rois = ["AL(L)", "AL(R)"]
# conn = conn[conn["roi"].isin(target_rois)]

# ============================================================
# 2. 建立 bodyId -> cell type 映射
# ============================================================
# bodyId,instance,type
neuron_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/data_Janelia_FlyEM_Hemibrain/traced-neurons.csv"   # ← 换成你的文件
neurons = pd.read_csv(neuron_path, usecols=["bodyId", "type"])
neurons["type"] = neurons["type"].fillna("").astype(str).str.strip()
id2type = neurons.set_index("bodyId")["type"].to_dict()

# 方式B（需联网 + neuprint-python）：直接从 neuPrint 查询
# from neuprint import Client, fetch_neurons, NeuronCriteria
# c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='YOUR_TOKEN')
# neuron_df, _ = fetch_neurons(NeuronCriteria(status='Traced'))
# id2type = neuron_df.set_index('bodyId')['type'].fillna('').to_dict()

conn["pre_type"] = conn["bodyId_pre"].map(id2type).fillna("")
conn["post_type"] = conn["bodyId_post"].map(id2type).fillna("")

n_before = len(conn)
conn = conn[(conn["pre_type"] != "") & (conn["post_type"] != "")]
print(f"剔除无类型标注的连接: {n_before - len(conn)} / {n_before}")

# ============================================================
# 3. 按 (pre_type, post_type) 聚合：weight 求和
#    （同一神经元对跨多个 roi 的多行在此自然合并）
# ============================================================
edges = (conn.groupby(["pre_type", "post_type"])
             .agg(weight=("weight", "sum"),
                  n_connections=("weight", "size"))   # 底层神经元对数，备用
             .reset_index())

print(f"聚合后类型级边数: {len(edges)}")

# ============================================================
# 4. 透视成 类型 × 类型 矩阵
# ============================================================
type_list = sorted(set(edges["pre_type"]) | set(edges["post_type"]))
type_mat = (edges.pivot(index="pre_type", columns="post_type", values="weight")
                 .reindex(index=type_list, columns=type_list)
                 .fillna(0))

# ============================================================
# 5. 归一化版本：边权 / 下游类型总输入（论文 Fig.4g 的做法）
# ============================================================
total_input = type_mat.sum(axis=0).replace(0, np.nan)   # 每个 post 类型的总输入
type_mat_norm = type_mat.div(total_input, axis=1).fillna(0)

# ============================================================
# 6. （可选）可靠性阈值过滤（论文启发式规则）
# ============================================================
RELIABLE_ABS = 10        # 绝对边权 >= 10 突触 -> 跨半球复现率 >90%
RELIABLE_FRAC = 0.009    # 归一化边权 >= 0.9% -> 复现率 >90%

reliable_mask = (type_mat >= RELIABLE_ABS) | (type_mat_norm >= RELIABLE_FRAC)
type_mat_reliable = type_mat.where(reliable_mask, 0)

print(f"可靠边数量: {(type_mat_reliable > 0).sum().sum()} / {(type_mat > 0).sum().sum()}")
print(f"可靠边包含的突触占比: "
      f"{type_mat_reliable.values.sum() / type_mat.values.sum():.1%}")

# ============================================================
# 7. 保存
# ============================================================
type_mat.to_csv("connectivity_matrix_by_celltype.csv")
type_mat_norm.to_csv("connectivity_matrix_by_celltype_inputNormalized.csv")
type_mat_reliable.to_csv("connectivity_matrix_by_celltype_reliable.csv")
edges.to_csv("celltype_edges_long.csv", index=False)
print("✅different kinds of structual connectivity matrix is saved!")

# ============================================================
# 8. 热图预览（类型太多时标签看不清，可加可靠边筛选后单独画子集）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
sns.heatmap(np.log1p(type_mat.values),   # log(1+w) 压缩色阶
            cmap="viridis", square=True,
            xticklabels=type_list, yticklabels=type_list,
            cbar_kws={"label": "log(1 + synapse count)", "shrink": 0.8}, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
ax.set_xlabel("postsynaptic cell type"); ax.set_ylabel("presynaptic cell type")
plt.tight_layout()
plt.savefig("celltype_connectivity_heatmap.png", dpi=300, bbox_inches="tight")
print("✅saved to celltype_connectivity_heatmap.png!")
# plt.show()