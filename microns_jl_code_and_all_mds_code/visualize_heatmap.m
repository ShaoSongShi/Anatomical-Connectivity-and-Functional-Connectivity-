%% ================== 距离矩阵可视化（极简版） ==================
clear; clc; close all;

%%%%%%%%%%%% 1. 加载与预处理 %%%%%%%%%%%%
load('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/distance_matrix.mat');

D = (D + D') / 2;
n_obs = size(D, 1);
D(1:n_obs+1:end) = 0;

mask = ~eye(size(D));
D(mask & D == 0) = 1e-6;
D(D < 0) = 1e-6;
D(D > 1) = 1;

d_vec = D(mask);
max_dist = max(d_vec);

fprintf('矩阵维度: %d × %d\n', n_obs, n_obs);
fprintf('最大距离: %.10f\n', max_dist);
fprintf('中位数: %.6f | 均值: %.6f\n', median(d_vec), mean(d_vec));

%%%%%%%%%%%% 输出路径 %%%%%%%%%%%%
out_dir = '/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%%%%%%%%%%%% 2. 热图（颜色反转）%%%%%%%%%%%%
figure('Color', 'w');
imagesc(D);
axis square tight;
colormap(flipud(jet));   % 颜色反转：原jet中蓝色(低)↔红色(高)互换
colorbar;
caxis([0 1]);
title(sprintf('Distance Matrix (N = %d)', n_obs), 'FontSize', 13);
xlabel('Neuron Index'); ylabel('Neuron Index');
set(gca, 'FontSize', 11, 'TickDir', 'out', 'Box', 'off');

print(gcf, fullfile(out_dir, 'distance_matrix_heatmap'), '-dpng', '-r300');

%%%%%%%%%%%% 3. 非对角元素分布（纯柱状图）%%%%%%%%%%%%
figure('Color', 'w');
histogram(d_vec, 80, 'FaceColor', [0.35 0.25 0.65], 'EdgeColor', 'k', 'LineWidth', 0.5);

set(gca, 'YScale', 'log');  % <-- y 轴对数刻度

xlabel('Distance'); 
ylabel('Count (log scale)');
title(sprintf('Distance Distribution (N = %d)', numel(d_vec)), 'FontSize', 13);
set(gca, 'FontSize', 11, 'TickDir', 'out', 'Box', 'off');

print(gcf, fullfile(out_dir, 'distance_distribution'), '-dpng', '-r300');

fprintf('✅ 图片已保存至: %s\n', out_dir);