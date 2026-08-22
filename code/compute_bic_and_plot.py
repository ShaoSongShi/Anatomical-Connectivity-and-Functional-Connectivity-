#!/usr/bin/env python3
"""
compute_bic_and_plot.py

读取 MATLAB 批量 Sammon MDS 结果（results.mat）与 ERM 准备数据（prepared.mat），
对每个方法（Classic / Huber / IRLS Sammon MDS）和每个嵌入维度：
  1. 由嵌入坐标 X 重构欧氏距离矩阵 D2；
  2. 按 ERM 反解重构相关性矩阵 C2 = eps^mu * (D2^2 + eps^2)^(-mu/2)；
  3. 在相关性域比较原始相关性矩阵 Corr 与重构矩阵 C2：
       - NCC / MSE / MAE（上三角非对角元素）；
       - BIC：理想情况下重构前后相关性矩阵的残差为高斯噪声，
         r_ij = C_ij - C2_ij ~ N(0, sigma^2)，轮廓化 sigma^2 并丢弃常数项后
             BIC(d) = n * ln(RSS/n) + k_d * ln(n),
             RSS = sum_{i<j} r_ij^2,  n = N(N-1)/2,
             k_d = N*d - d(d+1)/2 + 1
         （k_d 扣除 d 维欧氏嵌入的旋转/平移/反射不可识别自由度，
           +1 为噪声尺度参数）。
       三种 MDS 方法共用同一似然族与 k/n 口径，BIC 曲线可横向比较。
       加 --fisher-z 时在 atanh(C) 域计算残差（相关性估计噪声经
       Fisher z 变换后方差近似恒定为 1/(T-3)，更接近同方差高斯）。

用法：
  python3 compute_bic_and_plot.py <results.mat> <prepared.mat> [output_prefix] [--fisher-z]
"""

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# .mat 读取（v7 / v7.3 HDF5 自适应）
# ---------------------------------------------------------------------------
def load_mat_auto(filepath):
    try:
        from scipy.io import loadmat
        return loadmat(filepath, simplify_cells=False)
    except NotImplementedError:
        import h5py
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
    """把 .mat 中的 cell 统一成 python list。"""
    v = data[key]
    if isinstance(v, list):
        return v
    v = np.asarray(v, dtype=object)
    return [v.flat[i] for i in range(n)]


def extract_scalar(data, key, idx):
    val = np.asarray(data[key], dtype=float)
    return float(val.flat[idx])


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def corr_metrics(C_orig, C_recon, iu):
    """在相关性域比较（仅上三角非对角）。"""
    a = C_orig[iu]
    b = C_recon[iu]
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    x = a - a.mean()
    y = b - b.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    ncc = float(np.sum(x * y) / denom) if denom > 0 else np.nan
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    return ncc, mse, mae


def n_free_params(N, d):
    """k_d = N*d - d(d+1)/2 + 1（扣除旋转/平移/反射自由度，+1 为噪声尺度）。"""
    return N * d - d * (d + 1) // 2 + 1


def bic_gaussian(r, n_pairs, k_d):
    """相关性域高斯 BIC：BIC = n*ln(RSS/n) + k_d*ln(n)。
    三种 MDS 方法共用同一似然族与口径，曲线可横向比较。"""
    rss = max(np.sum(r**2), np.finfo(float).tiny)
    return n_pairs * np.log(rss / n_pairs) + k_d * np.log(n_pairs)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_metric(dims, results, metric_name, ylabel, filename, lower_is_better=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        'Classic Sammon MDS': '#2E86AB',
        'Huber Sammon MDS':   '#A23B72',
        'IRLS Sammon MDS':    '#F18F01'
    }
    markers = {
        'Classic Sammon MDS': 'o',
        'Huber Sammon MDS':   's',
        'IRLS Sammon MDS':    '^'
    }
    linestyles = {
        'Classic Sammon MDS': '-',
        'Huber Sammon MDS':   '--',
        'IRLS Sammon MDS':    '-.'
    }
    for name, res in results.items():
        vals = res[metric_name]
        arr = np.asarray(vals, dtype=float)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad > 0:
            print(f"  WARNING: {name} 的 {metric_name} 有 {n_bad} 个 NaN/Inf "
                  f"(dims={dims[~np.isfinite(arr)].tolist()})，对应线段不会出现在图中。")
        ax.plot(dims, vals, marker=markers[name], color=colors[name],
                linestyle=linestyles[name],
                linewidth=2.5, markersize=8, label=name,
                markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        idx_local = np.argmin(np.where(finite, arr, np.inf)) if lower_is_better \
            else np.argmax(np.where(finite, arr, -np.inf))
        ax.scatter([dims[idx_local]], [arr[idx_local]], s=350, c=colors[name],
                   marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax.set_xlabel('Embedding Dimension', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{metric_name.upper()} vs. Dimension: Sammon MDS Methods', fontsize=15)
    ax.set_xticks(dims)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(results_path, prepared_path, output_prefix, fisher_z=False):
    print(f"Loading results : {results_path}")
    print(f"Loading prepared: {prepared_path}")
    if fisher_z:
        print("BIC 将在 Fisher z (atanh) 域计算")
    res_data = load_mat_auto(results_path)
    prep = load_mat_auto(prepared_path)

    dims = np.asarray(prep['dims']).flatten().astype(int)
    n_dims = len(dims)
    Corr = np.asarray(prep['Corr'], dtype=float)
    N = Corr.shape[0]
    mu_vec = np.asarray(prep['mu_vec']).flatten()
    epsilon = float(np.asarray(prep['epsilon']).flatten()[0])
    print(f"N = {N}, dims = {list(dims)}, epsilon = {epsilon}")

    iu = np.triu_indices(N, 1)
    n_pairs = N * (N - 1) // 2

    methods = {
        'Classic Sammon MDS': {'coords': 'classic_coords', 'stress': 'classic_stress'},
        'Huber Sammon MDS':   {'coords': 'huber_coords',   'stress': 'huber_stress'},
        'IRLS Sammon MDS':    {'coords': 'irls_coords',    'stress': 'irls_stress'},
    }

    results = {}
    for name, keys in methods.items():
        coords_list = get_cell_array(res_data, keys['coords'], n_dims)
        bics, stresses, nccs, mses, maes = [], [], [], [], []

        for i, d in enumerate(dims):
            X = np.asarray(coords_list[i], dtype=float)
            if X.ndim == 3:
                X = X.squeeze()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != N:          # v7.3 转置方向兜底
                X = X.T

            # 1. 嵌入坐标 -> 欧氏距离矩阵
            D2 = squareform(pdist(X, metric='euclidean'))
            np.fill_diagonal(D2, 0.0)

            # 2. ERM 反解重构相关性矩阵 C2 = eps^mu (D2^2 + eps^2)^(-mu/2)
            mu = float(mu_vec[i])
            C2 = epsilon**mu * np.power(D2**2 + epsilon**2, -mu / 2.0)

            # 3. 相关性域指标（Corr vs C2）
            ncc, mse, mae = corr_metrics(Corr, C2, iu)
            nccs.append(ncc); mses.append(mse); maes.append(mae)

            # 4. 相关性域高斯 BIC（三种方法同口径，可横向比较）
            k_d = n_free_params(N, int(d))
            if fisher_z:
                clip = 1.0 - 1e-10
                rc = (np.arctanh(np.clip(Corr[iu], -clip, clip))
                      - np.arctanh(np.clip(C2[iu], -clip, clip)))
            else:
                rc = Corr[iu] - C2[iu]
            bic = bic_gaussian(rc, n_pairs, k_d)
            bics.append(bic)
            stresses.append(extract_scalar(res_data, keys['stress'], i))

            print(f"  {name:22s} d={d:2d}: BIC={bic:14.2f} NCC={ncc:.4f} "
                  f"MSE={mse:.6f} MAE={mae:.6f}")

        results[name] = {'dims': dims, 'bics': bics, 'stresses': stresses,
                         'ncc': nccs, 'mse': mses, 'mae': maes}

    # ---- 导出全部指标到 CSV，便于检查缺失/重合 ----
    rows = []
    for name, res in results.items():
        for i, d in enumerate(dims):
            rows.append({
                'method': name, 'dim': int(d),
                'bic': res['bics'][i],
                'ncc': res['ncc'][i], 'mse': res['mse'][i],
                'mae': res['mae'][i], 'stress': res['stresses'][i],
            })
    csv_file = f'{output_prefix}_metrics.csv'
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    print(f"\nAll metrics saved to: {csv_file}")

    # ---- 曲线重合检测：两两比较各指标序列 ----
    names = list(results.keys())
    for metric in ['ncc', 'mse', 'mae', 'bics']:
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                va = np.asarray(results[names[a]][metric], dtype=float)
                vb = np.asarray(results[names[b]][metric], dtype=float)
                both = np.isfinite(va) & np.isfinite(vb)
                if both.any():
                    diff = np.nanmax(np.abs(va[both] - vb[both]))
                    scale = max(np.nanmax(np.abs(va[both])), 1e-12)
                    if diff / scale < 1e-3:
                        print(f"  NOTE: {metric} 序列 {names[a]} 与 {names[b]} "
                              f"几乎重合（最大相对差 {diff/scale:.2e}），"
                              f"图中先画的线会被盖住。")

    # ---- 绘图 ----
    print("\nGenerating plots...")
    ztag = ' (Fisher z domain)' if fisher_z else ''
    plot_metric(dims, results, 'bics',
                f'BIC on correlation residuals{ztag}  (lower is better)',
                f'{output_prefix}_bic.png', lower_is_better=True)
    plot_metric(dims, results, 'ncc',  'NCC (Corr vs reconstructed, higher is better)',
                f'{output_prefix}_ncc.png', lower_is_better=False)
    plot_metric(dims, results, 'mse',  'MSE (Corr vs reconstructed, lower is better)',
                f'{output_prefix}_mse.png', lower_is_better=True)
    plot_metric(dims, results, 'mae',  'MAE (Corr vs reconstructed, lower is better)',
                f'{output_prefix}_mae.png', lower_is_better=True)

    # ---- 文本摘要 ----
    summary_file = f'{output_prefix}_summary.txt'
    lines = []
    lines.append("=" * 70)
    lines.append("Sammon MDS Batch Results Summary")
    lines.append("BIC: Gaussian likelihood on correlation residuals (C - C2)"
                 + (" (Fisher z domain)" if fisher_z else ""))
    lines.append("  BIC(d) = n*ln(RSS/n) + k_d*ln(n), "
                 "k_d = N*d - d(d+1)/2 + 1, n = N(N-1)/2")
    lines.append("  Same likelihood family & counting for all methods -> cross-comparable")
    lines.append("=" * 70)
    for metric, label, lower in [('bics', 'BIC', True), ('ncc', 'NCC', False),
                                 ('mse', 'MSE', True), ('mae', 'MAE', True)]:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"{label} Summary ({'lower' if lower else 'higher'} is better)")
        lines.append(f"{'Method':<25} {'Best Dim':>10} {label:>16} {'Stress':>12}")
        lines.append("-" * 70)
        for name, res in results.items():
            vals = np.array(res[metric], dtype=float)
            finite = np.isfinite(vals)
            idx = int(np.argmin(np.where(finite, vals, np.inf))) if lower \
                else int(np.argmax(np.where(finite, vals, -np.inf)))
            lines.append(f"{name:<25} {dims[idx]:>10} {vals[idx]:>16.4f} "
                         f"{res['stresses'][idx]:>12.6f}")
    text = "\n".join(lines)
    with open(summary_file, 'w') as f:
        f.write(text + "\n")
    print(f"\n{text}")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("results_mat")
    ap.add_argument("prepared_mat")
    ap.add_argument("output_prefix", nargs="?", default="sammon")
    ap.add_argument("--fisher-z", action="store_true",
                    help="在 atanh(C) 域计算相关性残差的 BIC")
    args = ap.parse_args()
    main(args.results_mat, args.prepared_mat, args.output_prefix, args.fisher_z)
