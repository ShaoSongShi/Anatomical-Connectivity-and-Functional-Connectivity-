#!/bin/bash
set -e  # 遇错即停

TMP_DIR="/home/wangzezhen/tmp_matlab"
mkdir -p "$TMP_DIR"
export TMPDIR="$TMP_DIR"
export MATLAB_PREFDIR="$TMP_DIR/.matlab"

PREPARED="/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code/fly_subset_sammon_2_10_prepared.mat"
RESULTS="/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code/fly_subset_sammon_2_10_results.mat"

echo "检查 prepared 文件是否存在..."
if [ ! -f "$PREPARED" ]; then
    echo "错误: $PREPARED 不存在！请先生成。"
    exit 1
fi

echo "开始补跑维度 ..."

for dim in 4 5 6 7 8 9 10; do
    echo ">>> 正在处理维度 $dim ..."
    matlab -Djava.io.tmpdir="$TMP_DIR" -batch "run_single_dim('$PREPARED', '$RESULTS', $dim);" 2>&1 | tee -a matlab_log.txt
    if [ $? -eq 0 ]; then
        echo "维度 $dim 成功完成"
    else
        echo "维度 $dim 失败，退出"
        exit 1
    fi
done

echo "所有指定维度处理完毕。"