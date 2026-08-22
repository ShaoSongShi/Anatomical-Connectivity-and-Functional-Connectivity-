#!/usr/bin/env python3
"""
prepare_distance_matrices.py

从相关性矩阵出发，按 ERM 模型为每个嵌入维度 d 生成距离矩阵 D_d。

流程（对应 Julia 代码逻辑）：
  1. 读取相关性矩阵 CSV（行名/列名为标签，允许重复标签，假定行列同序）；
  2. 对上三角相关值做直方图（0.1:0.01:0.5，归一化为 pdf），
     在 log-log 下做最小二乘拟合 ln(p) = b + k * ln(R)，得到截距 b 与斜率 k；
  3. 对每个嵌入维度 d：
       mu  = | -d / (k + 1) |
       S_d = 2 * pi^(d/2) / Gamma(d/2)
       rho = | N * mu * exp(b) / (S_d * eps^d) |
       L   = (N / rho)^(1/d)
  4. 由相关性矩阵计算距离矩阵（find_D）：
       D = eps * sqrt(| C^(-2/mu) - 1 |)
       D[D > L] = L * log(D[D > L] / L) + L      # 对数截断
  5. 保存 prepared.mat：dims / D_cells / Corr / mu_vec / rho_vec / L_vec /
     epsilon / slope_k / intercept_b。

用法：
  python3 prepare_distance_matrices.py corr.csv prepared.mat \
      --dims 2:10 --epsilon 0.03125
"""
import hdf5storage
import argparse
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.special import gamma


def load_correlation_csv(csv_path):
    """读取相关性矩阵 CSV。行名列名允许重复，假定行与列同序对应。"""
    df = pd.read_csv(csv_path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"相关性矩阵不是方阵: {df.shape}。请确认第一行/第一列为标签。"
        )
    labels = list(df.index)
    C = df.to_numpy(dtype=float)

    # 对角线缺失（空值）按 1 填充；非对角 NaN 按 0（最弱连接）处理
    nan_diag = np.isnan(np.diag(C))
    np.fill_diagonal(C, np.where(nan_diag, 1.0, np.diag(C)))
    n_nan = int(np.isnan(C).sum())
    if n_nan > 0:
        print(f"WARNING: 非对角存在 {n_nan} 个 NaN，已按 0 填充。")
        C = np.nan_to_num(C, nan=0.0)

    # 对称化
    C = (C + C.T) / 2.0
    return C, labels


def fit_loglog_slope(C, bin_start=0.1, bin_width=0.01, bin_end=0.5,
                     trim_head=13, trim_tail=7):
    """
    复现 Julia 代码的分布拟合：
      上三角相关值 -> max(0, R) -> 直方图(归一化 pdf) -> 非零 bin 中心 ->
      去掉前 trim_head 个与后 trim_tail 个点 -> log-log 最小二乘。
    返回 (intercept_b, slope_k)。
    """
    N = C.shape[0]
    iu = np.triu_indices(N, 1)
    R = C[iu]
    cd = np.maximum(R, 0.0)

    edges = np.arange(bin_start, bin_end + bin_width / 2.0, bin_width)
    weights, edges = np.histogram(cd, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    mask = weights > 0
    cor_vals = centers[mask]
    p_vals = weights[mask]
    print(f"直方图非零 bin 数: {len(cor_vals)}")

    lo = trim_head - 1                      # Julia 15 -> python index 14
    hi = len(cor_vals) - trim_tail          # Julia length-10
    if hi <= lo + 1:
        print("WARNING: 截断后剩余点不足，改用全部非零 bin 拟合。")
        lo, hi = 0, len(cor_vals)
    x = np.log(cor_vals[lo:hi])
    y = np.log(p_vals[lo:hi])

    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    b, k = float(coef[0]), float(coef[1])

    print(f"斜率 k = {k:.6f}")
    print(f"截距 b = {b:.6f}")
    return b, k


def erm_params_per_dim(N, b, k, d, epsilon):
    """按 Julia 代码，对每个维度 d 计算 mu, rho, L。"""
    mu = abs(-d / (k + 1.0))
    S_d = 2.0 * np.pi ** (d / 2.0) / gamma(d / 2.0)
    rho = abs((N * mu * np.exp(b)) / (S_d * epsilon ** d))
    L = (N / rho) ** (1.0 / d)
    return mu, rho, L


def find_D(C, mu, epsilon, L):
    """find_D：D = eps*sqrt(|C^(-2/mu) - 1|)，超过 L 的部分对数截断。"""
    C_abs = np.maximum(np.abs(C), 1e-10)      # 防止 C=0 导致 Inf
    D = epsilon * np.sqrt(np.abs(np.power(C_abs, -2.0 / mu) - 1.0))
    over = D > L
    D[over] = L * np.log(D[over] / L) + L
    np.fill_diagonal(D, 0.0)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corr_csv", help="相关性矩阵 CSV（第一行/第一列为标签）")
    ap.add_argument("output_mat", help="输出 .mat 文件路径")
    ap.add_argument("--dims", default="2:10",
                    help="嵌入维度范围，格式 start:end（含端点），默认 2:10")
    ap.add_argument("--epsilon", type=float, default=0.03125, help="ERM 参数 eps")
    ap.add_argument("--trim-head", type=int, default=13, help="log-log 拟合去掉前几个点")
    ap.add_argument("--trim-tail", type=int, default=7, help="log-log 拟合去掉后几个点")
    args = ap.parse_args()

    s, e = args.dims.split(":")
    dims = list(range(int(s), int(e) + 1))

    print(f"读取相关性矩阵: {args.corr_csv}")
    C, labels = load_correlation_csv(args.corr_csv)
    N = C.shape[0]
    print(f"矩阵大小: {N} x {N}")

    b, k = fit_loglog_slope(C, trim_head=args.trim_head, trim_tail=args.trim_tail)

    # D_cells = np.empty((len(dims), 1), dtype=object)
    mu_vec, rho_vec, L_vec = [], [], []
    for i, d in enumerate(dims):
        mu, rho, L = erm_params_per_dim(N, b, k, d, args.epsilon)
        # D = find_D(C, mu, args.epsilon, L)
        # D_cells[i, 0] = D
        mu_vec.append(mu)
        rho_vec.append(rho)
        L_vec.append(L)
        # print(f"d={d:2d}: mu={mu:.6f}, rho={rho:.6e}, L={L:.6f}, "
        #       f"D range=[{D[np.triu_indices(N,1)].min():.3e}, {D.max():.6f}]")

   
    # 替换 savemat 为：
    hdf5storage.savemat(args.output_mat, {
        "dims": np.array(dims, dtype=np.float64),
        # "D_cells": D_cells,          # 注意：hdf5storage 会正确处理 object 数组
        "Corr": C,
        "mu_vec": np.array(mu_vec),
        "rho_vec": np.array(rho_vec),
        "L_vec": np.array(L_vec),
        "epsilon": np.array([args.epsilon]),
        "slope_k": np.array([k]),
        "intercept_b": np.array([b]),
        "labels": np.array(labels, dtype=object),
    }, format='7.3')


if __name__ == "__main__":
    main()
