#!/bin/bash
# ============================================================================
# run_sammon_pipeline.sh
#
# 一键运行：
#   Step 0 (Python) : 相关性矩阵 -> ERM 拟合斜率 -> 逐维度生成距离矩阵 D_d
#   Step 1 (MATLAB) : 每个 D_d 上批量运行三种 Sammon MDS
#   Step 2 (Python) : 重构 C2，相关性域 NCC/MSE/MAE + 分方法 BIC，绘图
#
# 用法：
#   ./run_sammon_pipeline.sh <corr.csv> [dims] [output_prefix] [epsilon]
# 示例：
#   ./run_sammon_pipeline.sh corr.csv 2:10 sammon 0.03125
# prefix命名：物种_连接权重定义方法或者连接矩阵处理方法_降维方法_嵌入维度
# ============================================================================
set -e

MATLAB_BIN="$HOME/MATLAB/R2023b/bin/matlab"

CORR_CSV="${1:-corr.csv}"
DIMS="${2:-2:10}"
PREFIX="${3:-sammon}"
EPSILON="${4:-0.03125}"

PREPARED_MAT="${PREFIX}_prepared.mat"
RESULTS_BASE="${PREFIX}_results_dim_"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "$CORR_CSV" ]; then
    echo "ERROR: Input file '$CORR_CSV' not found."
    exit 1
fi

if [ ! -f "$MATLAB_BIN" ]; then
    echo "Trying fallback: /Applications/MATLAB_R2023b.app/bin/matlab"
    MATLAB_BIN="/Applications/MATLAB_R2023b.app/bin/matlab"
    if [ ! -f "$MATLAB_BIN" ]; then
        echo "ERROR: MATLAB binary not found. Please check your MATLAB installation."
        exit 1
    fi
    export MATLAB=/Applications/MATLAB_R2023b.app
    export MATLAB_ARCH=maca64
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found in PATH."
    exit 1
fi

python3 -c "import scipy, numpy, pandas, matplotlib" 2>/dev/null || {
    echo "ERROR: Python dependencies missing. Run: pip3 install scipy numpy pandas matplotlib"
    exit 1
}

for f in prepare_distance_matrices.py run_batch_sammon.m compute_bic_and_plot.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: '$f' not found in $SCRIPT_DIR"
        exit 1
    fi
done

# ==============================
# Step 0: 生成 prepared.mat
# ==============================
python3 prepare_distance_matrices.py "$CORR_CSV" "$PREPARED_MAT" \
    --dims "$DIMS" --epsilon "$EPSILON"

# ==============================
# 检测是否已有完整的分批结果文件
# ==============================
# 解析维度列表 (支持 start:end 或 start:step:end)
dim_list=$(python3 -c "
import sys
dims_str = sys.argv[1]
parts = dims_str.split(':')
if len(parts) == 1:
    dims = [int(parts[0])]
elif len(parts) == 2:
    start, end = map(int, parts)
    dims = list(range(start, end+1))
elif len(parts) == 3:
    start, step, end = map(int, parts)
    dims = list(range(start, end+1, step))
else:
    raise ValueError('Invalid dims format')
print(' '.join(map(str, dims)))
" "$DIMS")

ABS_BASE="$(cd "$SCRIPT_DIR" && pwd)/$RESULTS_BASE"
all_exist=true
for dim in $dim_list; do
    result_file="${ABS_BASE}${dim}.mat"
    if [ ! -f "$result_file" ]; then
        all_exist=false
        break
    fi
done

# ==============================
# Step 1: MATLAB 批量 Sammon MDS
# ==============================
if $all_exist; then
    echo "All result files ($(echo $dim_list | tr ' ' ',')) already exist, skipping MATLAB step."
else
    echo "Some result files missing, running MATLAB batch Sammon..."
    ABS_INPUT="$(cd "$(dirname "$PREPARED_MAT")" && pwd)/$(basename "$PREPARED_MAT")"
    "$MATLAB_BIN" -batch "run_batch_sammon('$ABS_INPUT', '$ABS_BASE');" 2>&1 | tee matlab_log.txt
fi

# ==============================
# Step 2: Python 汇总与绘图
# ==============================
python3 compute_bic_and_plot.py "$RESULTS_BASE" "$PREPARED_MAT" "$PREFIX"