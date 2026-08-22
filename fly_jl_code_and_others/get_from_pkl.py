import pickle
import pandas as pd
import numpy as np
import os

# ========== 1. 读取 pkl 文件 ==========
pkl_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/Turner2021_connectome_connectivity/WeightedSynapseNumber_computed_20210114.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# ========== 2. 预览数据结构 ==========
print(f"数据类型: {type(data)}")
print(f"数据形状/长度: {getattr(data, 'shape', getattr(data, '__len__', lambda: 'N/A')())}")

# 如果是 DataFrame，直接预览
if isinstance(data, pd.DataFrame):
    print("\nDataFrame 前5行:")
    print(data.head())
    print(f"\n行列数: {data.shape}")
    print(f"列名: {list(data.columns[:10])}...")  # 只显示前10个列名
    
    # 直接保存为 CSV
    csv_path = pkl_path.replace('.pkl', '.csv')
    data.to_csv(csv_path)
    print(f"\n已保存为 CSV: {csv_path}")

# 如果是 numpy ndarray / scipy sparse matrix
elif isinstance(data, np.ndarray):
    print(f"\n数组形状: {data.shape}")
    print(f"数组 dtype: {data.dtype}")
    print(f"前5行5列:\n{data[:5, :5] if data.ndim >= 2 else data[:5]}")
    
    # 保存为 CSV
    csv_path = pkl_path.replace('.pkl', '.csv')
    np.savetxt(csv_path, data, delimiter=',')
    print(f"\n已保存为 CSV: {csv_path}")

# 如果是 scipy sparse matrix
elif hasattr(data, 'toarray'):
    print(f"\n稀疏矩阵形状: {data.shape}")
    dense = data.toarray()
    csv_path = pkl_path.replace('.pkl', '.csv')
    np.savetxt(csv_path, dense, delimiter=',')
    print(f"\n稀疏矩阵已转稠密并保存为 CSV: {csv_path}")

# 如果是 dict（可能包含多个矩阵）
elif isinstance(data, dict):
    print(f"\n字典包含的键: {list(data.keys())}")
    for key, value in data.items():
        print(f"\n键 '{key}' 的类型: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"  形状: {value.shape}")
        # 对每个值分别保存
        if isinstance(value, (np.ndarray, pd.DataFrame)) or hasattr(value, 'toarray'):
            csv_path = pkl_path.replace('.pkl', f'_{key}.csv')
            if isinstance(value, pd.DataFrame):
                value.to_csv(csv_path)
            elif hasattr(value, 'toarray'):
                np.savetxt(csv_path, value.toarray(), delimiter=',')
            else:
                np.savetxt(csv_path, value, delimiter=',')
            print(f"  已保存: {csv_path}")

# 如果是 list
elif isinstance(data, list):
    print(f"\n列表长度: {len(data)}")
    print(f"前3个元素类型: {[type(x) for x in data[:3]]}")
    # 尝试转为 DataFrame 保存
    try:
        df = pd.DataFrame(data)
        csv_path = pkl_path.replace('.pkl', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n列表已转 DataFrame 保存为 CSV: {csv_path}")
    except Exception as e:
        print(f"列表转 CSV 失败: {e}")

else:
    print(f"\n未处理的类型: {type(data)}")
    print("请根据上述输出手动处理。")