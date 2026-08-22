%% ================== 鲁棒 Sammon MDS 计算脚本 ==================
% 功能：读取距离矩阵 D → 运行鲁棒 Sammon MDS → 保存降维结果 X 和 stress
% 
% 原 mdscale(D, n, 'Criterion','sammon') 的鲁棒化实现
% 支持两种异常值处理策略：
%   1. Huber/L1 损失：在 SMACOF 迭代中动态施加 Huber 权重，软截断大残差
%   2. IRLS 自适应重加权：外层自适应估计异常值，内层 SMACOF 固定权重优化
% 【更新】IRLS 策略现使用 Huber Sammon MDS 的结果作为初始值
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

%%%%%%%%%%%% 3. 运行鲁棒 Sammon MDS %%%%%%%%%%%%
n = 5;

c_huber = 0.05;
c_irls = 4.0;
max_iter = 600;
tol = 1e-6;

% --- 3.1 Huber Sammon MDS ---
fprintf('========== 正在运行 Huber Sammon MDS (c=%.3f) ==========\n', c_huber);
[X_huber, stress_huber] = huber_sammon_mds(D, n, c_huber, max_iter, tol);
fprintf('✅ Huber Sammon 完成，Stress = %.6f\n\n', stress_huber);
save('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/mds_result_huber_d5.mat', 'X_huber', 'stress_huber');

% --- 3.2 IRLS 自适应 Sammon MDS（使用 Huber 结果初始化）---
fprintf('========== 正在运行 IRLS 自适应 Sammon MDS (c=%.1f) ==========\n', c_irls);
fprintf('>>> 使用 Huber Sammon MDS 结果作为初始值\n');
inner_max_iter = 20;
[X_irls, stress_irls] = irls_sammon_mds(D, n, c_irls, 50, tol, inner_max_iter, X_huber);
fprintf('✅ IRLS Sammon 完成，Stress = %.6f\n\n', stress_irls);
save('/home/user/ShenRuihong/bishe/corr_principal_validation/microns_syn_mat_ct/huber_irls_noMaxD/d5/mds_result_irls_d5.mat', 'X_irls', 'stress_irls');

%%%%%%%%%%%% 4. 保存结果 %%%%%%%%%%%%
fprintf('✅ 结果已保存\n');

%%%%%%%%%%%% 5. 辅助函数定义 %%%%%%%%%%%%

function [X, stress] = huber_sammon_mds(D, n, c, max_iter, tol)
%HUBER_SAMMON_MDS 基于 Huber 损失的鲁棒 Sammon MDS
    
    N = size(D, 1);
    rng(42);
    % ===== 经典 MDS 初始化（替代 randn）=====
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
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
    
    W_base = zeros(N);
    W_base(mask) = 1 ./ D(mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;
    
    fprintf('迭代\t相对变化\tHuber截断比例\n');
    
    for iter = 1:max_iter
        d = squareform(pdist(X));
        d(d == 0) = eps;
        
        abs_resid = zeros(N);
        abs_resid(mask) = abs(D(mask) - d(mask));
        
        W_huber = ones(N);
        W_huber(mask) = min(1, c ./ abs_resid(mask));
        
        W = W_base .* W_huber;
        W(1:N+1:end) = 0;
        
        % SMACOF 更新
        V = diag(sum(W, 2)) - W;
        B = zeros(N);
        B(mask) = -W(mask) .* D(mask) ./ d(mask);
        B(1:N+1:end) = -sum(B, 2);
        
        % ===== 数值稳定化求解 =====
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
            trunc_ratio = sum(W_huber(mask) < 1) / sum(mask(:));
            fprintf('%d\t%.6f\t%.2f%%\n', iter, change, trunc_ratio*100);
        end
        
        if change < tol
            fprintf('Huber Sammon MDS 收敛于第 %d 次迭代\n', iter);
            break;
        end
    end
    
    d_final = squareform(pdist(X));
    stress = sum((D(mask) - d_final(mask)).^2 ./ D(mask)) / sum(D(mask));
end

function [X, stress] = irls_sammon_mds(D, n, c, max_iter, tol, inner_max_iter, X_init)
%IRLS_SAMMON_MDS 基于 IRLS 自适应重加权的鲁棒 Sammon MDS
%   X_init - (可选) 外部传入的初始嵌入坐标，例如 Huber Sammon MDS 的输出
    
    N = size(D, 1);
    rng(42);
    
    % ===== 初始化：优先使用传入的 X_init，否则回退到经典 MDS =====
    if nargin >= 7 && ~isempty(X_init)
        % 检查并适配维度
        if size(X_init, 1) ~= N
            error('IRLS:DimensionMismatch', ...
                  '传入的 X_init 行数 (%d) 与距离矩阵维度 (%d) 不匹配', size(X_init, 1), N);
        end
        if size(X_init, 2) < n
            X = [X_init, zeros(N, n - size(X_init, 2))];
            fprintf('>>> X_init 维度不足 %d，已补零列至 %d 维\n', n, n);
        elseif size(X_init, 2) > n
            X = X_init(:, 1:n);
            fprintf('>>> X_init 维度超过 %d，已截取前 %d 列\n', n, n);
        else
            X = X_init;
            fprintf('>>> 使用传入 X_init 直接初始化 (%d x %d)\n', N, n);
        end
    else
        % 回退：经典 MDS 初始化
        try
            [X_init_cmd, ~] = cmdscale(D, n);
            if size(X_init_cmd, 2) < n
                X = [X_init_cmd, zeros(N, n - size(X_init_cmd, 2))];
            else
                X = X_init_cmd(:, 1:n);
            end
        catch ME
            warning('irls_sammon_mds:CMDSCALEFailed', ...
                    'CMDSCALE failed (%s), falling back to random init', ME.message);
            X = randn(N, n) * 0.01;
        end
    end
    
    mask = ~eye(N);
    
    W_base = zeros(N);
    W_base(mask) = 1 ./ D(mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;
    
    fprintf('外层\t内层\t相对变化\tMAD阈值\t截断比例\n');
    
    for outer = 1:max_iter
        % 基于当前 X 计算残差分布（外层权重更新）
        d = squareform(pdist(X));
        d(d == 0) = eps;
        
        abs_resid = zeros(N);
        abs_resid(mask) = abs(D(mask) - d(mask));
        
        mad_val = median(abs_resid(mask));
        if mad_val > 0
            threshold = c * mad_val / 0.6745;
        else
            threshold = c * mean(abs_resid(mask));
        end
        
        W_irls = zeros(N);
        W_irls(mask) = double(abs_resid(mask) <= threshold);
        
        W = W_base .* W_irls;
        W(1:N+1:end) = 0;
        
        if sum(W(:)) == 0
            error('IRLS：所有权重被截断为零，请减小 c 值或检查数据');
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
        
        trunc_ratio = sum(W_irls(mask) < 1) / sum(mask(:));
        fprintf('%d\t%d\t%.6f\t%.6f\t%.2f%%\n', ...
            outer, inner_iter_actual, outer_change, threshold, trunc_ratio*100);
        
        if outer_change < tol
            fprintf('IRLS Sammon MDS 收敛于第 %d 次外层迭代\n', outer);
            break;
        end
    end
    
    d_final = squareform(pdist(X));
    stress = sum((D(mask) - d_final(mask)).^2 ./ D(mask)) / sum(D(mask));
end