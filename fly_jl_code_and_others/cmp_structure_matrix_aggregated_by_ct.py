import numpy as np
import pandas as pd

def load_connectivity_matrix(filepath):
    df = pd.read_csv(filepath, index_col=0)
    # 确保行索引和列名相同（应该相同）
    if not df.index.equals(df.columns):
        raise ValueError(f"文件 {filepath} 的行标签和列标签不一致")
    cell_types = df.index.tolist()
    matrix = df.values.astype(float)
    return cell_types, matrix

def find_most_different_cell_type(matrix1, matrix2, cell_types):
    """
    比较两个结构连接矩阵，找出差异最大的行或列对应的细胞类型。

    参数:
        matrix1, matrix2: 二维列表或numpy数组，形状 (n, n)
        cell_types: 列表，长度为 n，按字母顺序排列的细胞类型名称

    返回:
        tuple: (细胞类型名称, 'row' 或 'col', 差异值)
    """
    # 转换为numpy数组以便计算
    m1 = np.array(matrix1, dtype=float)
    m2 = np.array(matrix2, dtype=float)

    if m1.shape != m2.shape:
        raise ValueError("两个矩阵的维度不匹配")
    if len(cell_types) != m1.shape[0]:
        raise ValueError("细胞类型列表长度与矩阵维度不一致")

    # 计算每个行向量的L1差异（绝对值之和）
    row_diffs = np.sum(np.abs(m1 - m2), axis=1)
    # 计算每个列向量的L1差异
    col_diffs = np.sum(np.abs(m1 - m2), axis=0)

    # 找出最大差异值所在的行和列
    max_row_idx = np.argmax(row_diffs)
    max_col_idx = np.argmax(col_diffs)
    max_row_val = row_diffs[max_row_idx]
    max_col_val = col_diffs[max_col_idx]

    # 比较行和列的最大差异，输出较大者
    if max_row_val >= max_col_val:
        return cell_types[max_row_idx], 'row', max_row_val
    else:
        return cell_types[max_col_idx], 'col', max_col_val

if __name__ == "__main__":
    file1 = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_method_traced_total/connectivity_matrix_by_celltype.csv"
    file2 = "/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_coupling_correlation/fly_connectivity_matrix_aggregated_by_type.csv"
    
    types1, mat1 = load_connectivity_matrix(file1)
    types2, mat2 = load_connectivity_matrix(file2)
    
    # 检查两个文件的细胞类型是否一致，若不一致，按第一个重排第二个
    if types1 != types2:
        # 确保所有类型都在第二个中，否则报错
        # 按照types1的顺序重排第二个矩阵
        idx = [types1.index(t) for t in types1]
        mat2 = mat2[np.ix_(idx, idx)]
        types2 = types1  # 更新为一致
        # 但要注意可能第二个有额外类型，这里仅使用共有的？但简单起见我们假设两者完全一致，但若顺序不同就重排。
    # 如果长度不同，则报错或取交集？暂且认为一致，因为都是字母顺序。
    
    result_type, axis, diff = find_most_different_cell_type(mat1, mat2, types1)
    print(f"差异最大的{axis}对应的细胞类型: {result_type}, 差异值: {diff}")