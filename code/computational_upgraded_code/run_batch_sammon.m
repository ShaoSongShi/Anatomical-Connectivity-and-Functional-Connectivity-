% 循环所有维度，逐个调用 run_single_dim。

function run_batch_sammon(input_path, output_base)
    % input_path  : prepared.mat 路径
    % output_base : 输出文件基础名（不含维度后缀和扩展名）
    %              例如 'sammon_results_dim_'，实际生成 'sammon_results_dim_2.mat' 等

    m = matfile(input_path, 'Writable', false);
    dims = m.dims(1, :);
    for i = 1:length(dims)
        d = dims(i);
        output_path = sprintf('%s%d.mat', output_base, d);
        run_single_dim(input_path, output_path, d);
    end
end