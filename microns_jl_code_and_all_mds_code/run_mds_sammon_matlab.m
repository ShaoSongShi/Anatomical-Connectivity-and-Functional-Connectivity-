%% ================== Sammon MDS 计算脚本 ==================
% 功能：读取距离矩阵 D → 运行 mdscale → 保存降维结果 X 和 stress
% 完全等价于：X, stress = mdscale(D, n, 'criterion','sammon');
%% ==========================================================

clear; clc; close all;

%%%%%%%%%%%% 1. 加载你的距离矩阵 D %%%%%%%%%%%%
% 替换成你 D 的路径（.mat 文件）
load('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/distance_matrix.mat');  % 必须包含变量：D (N×N 距离矩阵)
%%%%%%%%%%%% 2. 【唯一正确的修复方式】%%%%%%%%%%%%
% 1. 强制对称
D = (D + D') / 2;

% 2. ✅ 对角线必须 = 0 （核心！！！）
n_obs = size(D,1);
D(1:n_obs+1:end) = 0;  

% 3. ✅ 只修复【非对角线】的 0 → 极小值
% 生成非对角线掩码
mask = ~eye(size(D));
D(mask & D == 0) = 1e-6;
D(D < 0) = 1e-6;

% 先缩放
% D = D / max(D(:));

% 再限制范围
D(D > 1) = 1;
% 检查
fprintf('修复后 D 最小值: %.10f\n', min(D(:)));
fprintf('修复后 D 最大值: %.10f\n', max(D(:)));
fprintf('是否对称: %d\n', issymmetric(D));
fprintf('对角线值: %.10f\n', D(1,1));  % 必须输出 0
fprintf('数据检查完成\n');


%%%%%%%%%%%% 3. 运行 Sammon MDS（核心代码）%%%%%%%%%%%%
n = 5;
fprintf('正在运行 Sammon MDS...\n');
opts = statset('Display','final');  % 显示迭代信息
[X, stress] = mdscale(D, n, ...
    'Criterion','sammon', ...
    'Start','random', ...
    'Options',opts);

fprintf('✅ Sammon 完成，Stress = %.6f\n', stress);

%%%%%%%%%%%% 4. 保存结果给 Julia 使用 %%%%%%%%%%%%
save('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/mds_result_d5.mat', 'X', 'stress');
fprintf('✅ 结果已保存：/home/user/ShenRuihong/bishe/corr_principal_validation/fly/results_subset/mds_result_d5.mat\n');