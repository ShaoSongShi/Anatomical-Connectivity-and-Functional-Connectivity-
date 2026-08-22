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
# cd /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code
# ============================================================================

set -e

MATLAB_BIN="$HOME/MATLAB/R2023b/bin/matlab"

# 参数解析
CORR_CSV="${1:-corr.csv}"
DIMS="${2:-2:10}"
PREFIX="${3:-sammon}"
EPSILON="${4:-0.03125}"

PREPARED_MAT="${PREFIX}_prepared.mat"
OUTPUT_MAT="${PREFIX}_results.mat"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Sammon MDS Pipeline (ERM correlation version)"
echo "=========================================="
echo "Input correlation CSV : $CORR_CSV"
echo "Dimensions            : $DIMS"
echo "Epsilon               : $EPSILON"
echo "Prepared data         : $PREPARED_MAT"
echo "Output results        : $OUTPUT_MAT"
echo "Output prefix         : $PREFIX"
echo "=========================================="

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

# ============================================================================
# Step 0: Python 生成逐维度距离矩阵
# ============================================================================
echo ""
echo "=========================================="
echo "Step 0: ERM fitting & distance matrices"
echo "=========================================="

python3 /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code/prepare_distance_matrices.py "$CORR_CSV" "$PREPARED_MAT" \
    --dims "$DIMS" --epsilon "$EPSILON"

# ============================================================================
# Step 1: 逐维度运行 MATLAB Sammon MDS
# ============================================================================
echo ""
echo "=========================================="
echo "Step 1: Running MATLAB Sammon MDS per dimension"
echo "=========================================="
echo "Methods: Classic, Huber (c=0.05), IRLS (c=4.0)"
echo ""

ABS_INPUT="$(cd "$(dirname "$PREPARED_MAT")" && pwd)/$(basename "$PREPARED_MAT")"
ABS_OUTPUT="$(cd "$SCRIPT_DIR" && pwd)/$OUTPUT_MAT"

# 解析维度范围（如 "2:10"）
DIM_START=$(echo $DIMS | cut -d':' -f1)
DIM_END=$(echo $DIMS | cut -d':' -f2)

# 删除可能存在的旧结果文件（避免残留干扰）
rm -f "$ABS_OUTPUT"

for dim in $(seq $DIM_START $DIM_END); do
    echo ""
    echo ">>> Processing dimension $dim ..."
    # 运行单维度处理，输出追加到日志
    "$MATLAB_BIN" -batch "run_single_dim('$ABS_INPUT', '$ABS_OUTPUT', $dim);" \
        2>&1 | tee -a matlab_log.txt

    # 检查该维度是否成功写入（可选）
    if [ ! -f "$ABS_OUTPUT" ]; then
        echo "ERROR: $ABS_OUTPUT not created for dimension $dim"
        exit 1
    fi
done

echo ""
echo "All dimensions processed. Results saved to: $ABS_OUTPUT"

# # ============================================================================
# # Step 1: MATLAB 批量 Sammon MDS
# # ============================================================================
# echo ""
# echo "=========================================="
# echo "Step 1: Running MATLAB batch Sammon MDS"
# echo "=========================================="
# echo "Methods: Classic, Huber (c=0.05), IRLS (c=4.0)"
# echo ""

# ABS_INPUT="$(cd "$(dirname "$PREPARED_MAT")" && pwd)/$(basename "$PREPARED_MAT")"
# ABS_OUTPUT="$(cd "$SCRIPT_DIR" && pwd)/$OUTPUT_MAT"

# "$MATLAB_BIN" -batch "run_batch_sammon('$ABS_INPUT', '$ABS_OUTPUT');" 2>&1 | tee matlab_log.txt

# if [ ! -f "$OUTPUT_MAT" ]; then
#     echo "ERROR: MATLAB did not produce output file '$OUTPUT_MAT'."
#     echo "Check matlab_log.txt for details."
#     exit 1
# fi

# echo ""
# echo "MATLAB results saved to: $OUTPUT_MAT"

# ============================================================================
# Step 2: Python 重构相关性矩阵、计算指标与 BIC、绘图
# ============================================================================
echo ""
echo "=========================================="
echo "Step 2: Reconstruction, metrics & BIC"
echo "=========================================="

python3 /home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code/compute_bic_and_plot.py "$OUTPUT_MAT" "$PREPARED_MAT" "$PREFIX"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Prepared: $PREPARED_MAT"
echo "Results:  $OUTPUT_MAT"
echo "Plots:    ${PREFIX}_{bic,ncc,mse,mae}.png"
echo "Summary:  ${PREFIX}_summary.txt"
echo "Log:      matlab_log.txt"
echo "=========================================="
