#!/bin/bash
# run_all_plots.sh
# 批量生成不同方法和层级的二维嵌入散点图

# 基础路径和固定参数
RESULTS="/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/fly_aggregateed_by_more_cell_type_nm_2_10_results.mat"
PREPARED="/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/fly_aggregateed_by_more_cell_type_nm_2_10_prepared.mat"
MATRIX_CSV="/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/results_aggregated_by_cell_type_nature_data_hemibrain/connectivity_matrix_aggregateed_by_more_cell_type.csv"
PY_SCRIPT="/home/wangzezhen/ShenRuihong/corr_principal_validation/BIC_pipeline/code/plot_embedding_2d.py"

# 方法列表和层级列表
METHODS=("Classic" "Huber" "IRLS")
COLOR_BY_LIST=("super_class" "cell_class" "cell_sub_class" "ito_lee_hemilineage" "hartenstein_hemilineage" "morphology_group")

# 遍历方法
for method in "${METHODS[@]}"; do
    # 遍历层级
    for color_by in "${COLOR_BY_LIST[@]}"; do
        # 构建输出文件名
        output="embedding_2d_${method}_${color_by}_by_more_ct_nm.png"
        echo "正在生成: ${output}"
        # 执行 Python 脚本
        python "${PY_SCRIPT}" \
            --results "${RESULTS}" \
            --prepared "${PREPARED}" \
            --matrix_csv "${MATRIX_CSV}" \
            --method "${method}" \
            --color_by "${color_by}" \
            --output "${output}"
        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "✅ 成功生成: ${output}"
        else
            echo "❌ 生成失败: ${output}"
        fi
        echo "----------------------------------------"
    done
done

echo "所有图片生成完毕。"