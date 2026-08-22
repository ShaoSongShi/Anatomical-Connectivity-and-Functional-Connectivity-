%% ================== 鲁棒 Sammon MDS 计算脚本 ==================
% 功能：读取距离矩阵 D → 运行经典 / Huber / IRLS Sammon MDS → 保存结果
% 关键修改：
%   1. 排除距离矩阵全局最大值对应的点对（原始相关性 = 1e-6）
%   2. 新增 classic_sammon_mds 函数
%% ==========================================================

clear; clc; close all;

%%%%%%%%%%%% 1. 加载你的距离矩阵 D %%%%%%%%%%%%
load('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/distance_matrix.mat');

%%%%%%%%%%%% 2. 【数据预处理】%%%%%%%%%%%%
D = (D + D') / 2;
n_obs = size(D,1);
D(1:n_obs+1:end) = 0;

mask = ~eye(size(D));
D(mask & D == 0) = 1e-6;
D(D < 0) = 1e-6;
D(D > 1) = 1;

fprintf('修复后 D 最小值: %.10f\n', min(D(:)));
fprintf('修复后 D 最大值: %.10f\n', max(D(:)));
fprintf('是否对称: %d\n', issymmetric(D));
fprintf('对角线值: %.10f\n', D(1,1));
fprintf('数据检查完成\n\n');

%%%%%%%%%%%% 2.5 排除全局最大值（原始相关性 = 1e-6）%%%%%%%%%%%%
max_dist = max(D(:));
% 精确匹配最大值，允许极小浮点容差
exclude_mask = abs(D - max_dist) < 1e-12;
% 保险：对角线永不排除
exclude_mask(1:n_obs+1:end) = false;

fprintf('全局最大距离: %.10f\n', max_dist);
fprintf('排除的最大距离点对数量: %d (占总非对角点对 %.2f%%)\n', ...
    sum(exclude_mask(:)), sum(exclude_mask(:)) / sum(mask(:)) * 100);

%%%%%%%%%%%% 3. 运行三种 Sammon MDS %%%%%%%%%%%%
n = 5;
c_huber = 0.05;
c_irls = 4.0;
max_iter = 600;
tol = 1e-6;
inner_max_iter = 20;
out_dir = '/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/noMaxD';
% --- 3.1 经典 Sammon MDS ---
fprintf('\n========== 正在运行 经典 Sammon MDS（排除最大值） ==========\n');
[X_classic, stress_classic] = classic_sammon_mds(D, n, 400, tol, exclude_mask);
fprintf('✅ 经典 Sammon 完成，Stress = %.6f\n\n', stress_classic);
save('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/noMaxDmds_result_noMaxD_classic_d5.mat', 'X_classic', 'stress_classic');
%save([out_dir 'mds_result_noMaxD_classic_d5.mat'], 'X_classic', 'stress_classic');
% --- 3.2 Huber Sammon MDS ---
fprintf('========== 正在运行 Huber Sammon MDS (c=%.3f, 排除最大值) ==========\n', c_huber);
[X_huber, stress_huber] = huber_sammon_mds(D, n, c_huber, max_iter, tol, exclude_mask);
fprintf('✅ Huber Sammon 完成，Stress = %.6f\n\n', stress_huber);
save('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/noMaxD/mds_result_noMaxD_huber_d5.mat', 'X_huber', 'stress_huber');
%save([out_dir 'mds_result_noMaxD_huber_d5.mat'], 'X_huber', 'stress_huber');
% --- 3.3 IRLS 自适应 Sammon MDS ---
fprintf('========== 正在运行 IRLS 自适应 Sammon MDS (c=%.1f, 排除最大值) ==========\n', c_irls);
[X_irls, stress_irls] = irls_sammon_mds(D, n, c_irls, 50, tol, inner_max_iter, exclude_mask);
fprintf('✅ IRLS Sammon 完成，Stress = %.6f\n\n', stress_irls);
save('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/noMaxD/mds_result_noMaxD_irls_d5.mat', 'X_irls', 'stress_irls');
%save([out_dir 'mds_result_noMaxD_irls_d5.mat'], 'X_irls', 'stress_irls');
%%%%%%%%%%%% 4. 保存结果 %%%%%%%%%%%%

fprintf('✅ 三种 MDS 结果已保存至 %s\n', out_dir);

%%%%%%%%%%%% 5. 辅助函数定义 %%%%%%%%%%%%

function [X, stress] = classic_sammon_mds(D, n, max_iter, tol, exclude_mask)
%CLASSIC_SAMMON_MDS 标准 Sammon MDS（支持排除指定距离点对）
    
    N = size(D, 1);
    if nargin < 5 || isempty(exclude_mask)
        exclude_mask = false(size(D));
    end
    rng(42);
    % ===== 经典 MDS 初始化（替代 randn）=====
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
            % 若有效维度不足 n，补零列
            X = [X_init, zeros(N, n - size(X_init, 2))];
        else
            X = X_init(:, 1:n);
        end
    catch ME
        warning('classic_sammon_mds:CMDSCALEFailed', ...
                'CMDSCALE failed (%s), falling back to random init', ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end
    
    mask = ~eye(N);
    valid_mask = mask & ~exclude_mask;
    
    % Sammon 基础权重，被排除点为 0
    W = zeros(N);
    W(valid_mask) = 1 ./ D(valid_mask);
    W(isinf(W) | isnan(W)) = 0;
    
    fprintf('迭代\t相对变化\n');
    for iter = 1:max_iter
        d = squareform(pdist(X));
        d(d == 0) = eps;
        
        % SMACOF 更新
        V = diag(sum(W, 2)) - W;
        B = zeros(N);
        B(mask) = -W(mask) .* D(mask) ./ d(mask);
        B(1:N+1:end) = -sum(B, 2);
        
        lambda_reg = 1e-6;
        X_new = (V + lambda_reg * eye(N)) \ (B * X);
        if any(isnan(X_new(:))) || any(isinf(X_new(:)))
            warning('MDS:Singular', '矩阵奇异，回退至伪逆');
            X_new = pinv(V + lambda_reg * eye(N)) * (B * X);
        end
        X_new = X_new - mean(X_new, 1);
        
        change = norm(X_new - X, 'fro') / (norm(X, 'fro') + eps);
        X = X_new;
        
        if mod(iter, 100) == 0 || iter == 1 || change < tol
            fprintf('%d\t%.6f\n', iter, change);
        end
        if change < tol
            fprintf('经典 Sammon MDS 收敛于第 %d 次迭代\n', iter);
            break;
        end
    end
    
    % 应力仅计算有效点对
    d_final = squareform(pdist(X));
    stress = sum((D(valid_mask) - d_final(valid_mask)).^2 ./ D(valid_mask)) ...
             / sum(D(valid_mask));
end

function [X, stress] = huber_sammon_mds(D, n, c, max_iter, tol, exclude_mask)
%HUBER_SAMMON_MDS 基于 Huber 损失的鲁棒 Sammon MDS（支持排除指定距离点对）
    
    N = size(D, 1);
    if nargin < 6 || isempty(exclude_mask)
        exclude_mask = false(size(D));
    end
    rng(42);
    % ===== 经典 MDS 初始化（替代 randn）=====
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
            % 若有效维度不足 n，补零列
            X = [X_init, zeros(N, n - size(X_init, 2))];
        else
            X = X_init(:, 1:n);
        end
    catch ME
        warning('classic_sammon_mds:CMDSCALEFailed', ...
                'CMDSCALE failed (%s), falling back to random init', ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end
    
    mask = ~eye(N);
    valid_mask = mask & ~exclude_mask;
    
    % Sammon 基础权重，被排除点为 0
    W_base = zeros(N);
    W_base(valid_mask) = 1 ./ D(valid_mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;
    
    fprintf('迭代\t相对变化\tHuber截断比例(仅有效点)\n');
    
    for iter = 1:max_iter
        d = squareform(pdist(X));
        d(d == 0) = eps;
        
        abs_resid = zeros(N);
        abs_resid(valid_mask) = abs(D(valid_mask) - d(valid_mask));
        
        % Huber 权重：软截断
        W_huber = ones(N);
        W_huber(valid_mask) = min(1, c ./ abs_resid(valid_mask));
        
        W = W_base .* W_huber;
        W(1:N+1:end) = 0;
        
        % SMACOF 更新
        V = diag(sum(W, 2)) - W;
        B = zeros(N);
        B(mask) = -W(mask) .* D(mask) ./ d(mask);
        B(1:N+1:end) = -sum(B, 2);
        
        lambda_reg = 1e-6;
        X_new = (V + lambda_reg * eye(N)) \ (B * X);
        if any(isnan(X_new(:))) || any(isinf(X_new(:)))
            warning('MDS:Singular', '矩阵奇异，回退至伪逆');
            X_new = pinv(V + lambda_reg * eye(N)) * (B * X);
        end
        X_new = X_new - mean(X_new, 1);
        
        change = norm(X_new - X, 'fro') / (norm(X, 'fro') + eps);
        X = X_new;
        
        if mod(iter, 100) == 0 || iter == 1 || change < tol
            trunc_ratio = sum(W_huber(valid_mask) < 1) / sum(valid_mask(:));
            fprintf('%d\t%.6f\t%.2f%%\n', iter, change, trunc_ratio*100);
        end
        if change < tol
            fprintf('Huber Sammon MDS 收敛于第 %d 次迭代\n', iter);
            break;
        end
    end
    
    d_final = squareform(pdist(X));
    stress = sum((D(valid_mask) - d_final(valid_mask)).^2 ./ D(valid_mask)) ...
             / sum(D(valid_mask));
end

function [X, stress] = irls_sammon_mds(D, n, c, max_iter, tol, inner_max_iter, exclude_mask)
%IRLS_SAMMON_MDS 基于 IRLS 自适应重加权的鲁棒 Sammon MDS（支持排除指定距离点对）
    
    N = size(D, 1);
    if nargin < 7 || isempty(exclude_mask)
        exclude_mask = false(size(D));
    end
    rng(42);
    % ===== 经典 MDS 初始化（替代 randn）=====
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
            % 若有效维度不足 n，补零列
            X = [X_init, zeros(N, n - size(X_init, 2))];
        else
            X = X_init(:, 1:n);
        end
    catch ME
        warning('classic_sammon_mds:CMDSCALEFailed', ...
                'CMDSCALE failed (%s), falling back to random init', ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end
    
    mask = ~eye(N);
    valid_mask = mask & ~exclude_mask;
    
    % Sammon 基础权重，被排除点为 0
    W_base = zeros(N);
    W_base(valid_mask) = 1 ./ D(valid_mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;
    
    fprintf('外层\t内层\t相对变化\tMAD阈值\t截断比例(仅有效点)\n');
    
    for outer = 1:max_iter
        d = squareform(pdist(X));
        d(d == 0) = eps;
        
        abs_resid = zeros(N);
        abs_resid(valid_mask) = abs(D(valid_mask) - d(valid_mask));
        
        % 自适应阈值：基于有效点残差的 MAD
        mad_val = median(abs_resid(valid_mask));
        if mad_val > 0
            threshold = c * mad_val / 0.6745;
        else
            threshold = c * mean(abs_resid(valid_mask));
        end
        
        % IRLS 权重：Talwar 硬截断
        W_irls = zeros(N);
        W_irls(valid_mask) = double(abs_resid(valid_mask) <= threshold);
        
        W = W_base .* W_irls;
        W(1:N+1:end) = 0;
        
        if sum(W(:)) == 0
            error('IRLS：所有有效权重被截断为零，请减小 c 值或检查数据');
        end
        
        % 内层：固定权重 W 运行 SMACOF
        X_inner = X;
        inner_iter_actual = inner_max_iter;
        
        for inner = 1:inner_max_iter
            d_inner = squareform(pdist(X_inner));
            d_inner(d_inner == 0) = eps;
            
            V = diag(sum(W, 2)) - W;
            B = zeros(N);
            B(mask) = -W(mask) .* D(mask) ./ d_inner(mask);
            B(1:N+1:end) = -sum(B, 2);
            
            lambda_reg = 1e-6;
            X_new = (V + lambda_reg * eye(N)) \ (B * X_inner);
            if any(isnan(X_new(:))) || any(isinf(X_new(:)))
                warning('MDS:Singular', '矩阵奇异，回退至伪逆');
                X_new = pinv(V + lambda_reg * eye(N)) * (B * X_inner);
            end
            X_new = X_new - mean(X_new, 1);
            
            inner_change = norm(X_new - X_inner, 'fro') / (norm(X_inner, 'fro') + eps);
            X_inner = X_new;
            
            if inner_change < tol
                inner_iter_actual = inner;
                break;
            end
        end
        
        outer_change = norm(X_inner - X, 'fro') / (norm(X, 'fro') + eps);
        X = X_inner;
        
        trunc_ratio = sum(W_irls(valid_mask) < 1) / sum(valid_mask(:));
        fprintf('%d\t%d\t%.6f\t%.6f\t%.2f%%\n', ...
            outer, inner_iter_actual, outer_change, threshold, trunc_ratio*100);
        
        if outer_change < tol
            fprintf('IRLS Sammon MDS 收敛于第 %d 次外层迭代\n', outer);
            break;
        end
    end
    
    d_final = squareform(pdist(X));
    stress = sum((D(valid_mask) - d_final(valid_mask)).^2 ./ D(valid_mask)) ...
             / sum(D(valid_mask));
end