import requests
import pandas as pd
from caveclient import CAVEclient
# 配置参数
TOKEN = "b809cdd3336677c62d2810645a1348d2"
DATASET = "minnie65_public"
TABLE_NAME = "allen_v1_column_types_slanted_ref"
# API请求头
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# ========================
# 连接 MICrONS 数据集
# ========================
client = CAVEclient("minnie65_public")
client.materialize.version = 1718  # 固定版本，不飘


df = client.materialize.query_table(TABLE_NAME)
# https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/nucleus_neuron_classification/nucleus_neuron_svm.csv
# ========================
# 保留你需要的列
# ========================
df = df[["id", "classification_system", "cell_type"]]

# ========================
# 导出CSV
# ========================
df.to_csv("/home/user/ShenRuihong/bishe/corr_principal_validation/allen_v1_column_types_slanted_ref.csv", index=False)

print("✅ 成功导出官方细胞类型表！")
print("文件名：allen_v1_column_types_slanted_ref.csv")
print(f"共 {len(df)} 个细胞标注")
print("\n前5行：")
print(df.head())