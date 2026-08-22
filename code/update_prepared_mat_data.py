import pandas as pd
import hdf5storage
import numpy as np

# ===== 文件路径（请确认路径正确）=====
prepared_mat_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/fly_aggregateed_by_more_cell_type_nm_2_10_prepared.mat"
conn_matrix_path = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/connectivity_matrix_aggregateed_by_more_cell_type.csv"

# 1. 读取连接矩阵，提取行索引（细胞类型名称）
conn_df = pd.read_csv(conn_matrix_path, index_col=0)
cell_types = conn_df.index.tolist()  # 顺序与矩阵行顺序一致
print(f"✅ 细胞类型数量: {len(cell_types)}")

# 2. 加载原有 prepared.mat（兼容 v7.3 HDF5 格式）
data = hdf5storage.loadmat(prepared_mat_path)
print(f"✅ 加载 prepared.mat 成功，现有变量: {list(data.keys())}, 原 labels 长度: {len(data.get('labels', []))}")

# 3. 更新 labels 字段
data['labels'] = np.array(cell_types, dtype=object)  # 保存为对象数组，对应 MATLAB cell

# 4. 保存回原文件（覆盖）
hdf5storage.savemat(prepared_mat_path, data, format='7.3')
print(f"✅ 已更新 prepared.mat 中的 labels，新长度 {len(cell_types)}")