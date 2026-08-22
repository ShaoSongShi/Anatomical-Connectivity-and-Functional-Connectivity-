import pandas as pd

# 读取 CSV：第一列作为行索引，第一行作为列名
df = pd.read_csv('/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_tbar_count_Branson_output_mode_corr/StructuralMatrix_branson_tbar.csv', index_col=0)

# 转置矩阵（行标签与列标签互换）
df_transposed = df.T

# 保存为 CSV，保留索引和列名（第一行第一列自动为空）
df_transposed.to_csv('/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_tbar_count_Branson/StructuralMatrix_branson_tbar.csv')
print("\n✅ 转置矩阵已保存: /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_tbar_count_Branson/StructuralMatrix_branson_tbar.csv")