function run_single_dim(input_path, output_path, dim_val)
    % input_path  : prepared.mat 路径
    % output_path : 该维度的输出 .mat 文件（如 'results_dim_2.mat'）
    % dim_val     : 当前处理维度

    m = matfile(input_path, 'Writable', false);
    dims = m.dims(1, :);
    idx = find(dims == dim_val, 1);
    if isempty(idx)
        error('维度 %d 不在 dims 列表中', dim_val);
    end

    % 读取 ERM 参数及原始相关矩阵
    Corr = m.Corr;
    mu = m.mu_vec(1, idx);
    L = m.L_vec(1, idx);
    epsilon = m.epsilon(1, 1);

    % 计算该维度的距离矩阵（find_D 函数）
    D = compute_D(Corr, mu, epsilon, L);
    n_obs = size(D,1);
    D(1:n_obs+1:end) = 0;          % 对角线归零
    mask = ~eye(n_obs);
    D(mask & D <= 0) = 1e-6;       % 防止零距离

    fprintf('正在处理维度 %d (%d/%d)\n', dim_val, idx, length(dims));

    % 参数设置
    c_huber = 0.05;
    c_irls = 4.0;
    max_iter = 600;
    tol = 1e-6;
    inner_max_iter = 20;

    % 执行三种 MDS
    [X_classic, s_classic] = classic_sammon_mds(D, dim_val, max_iter, tol);
    [X_huber, s_huber] = huber_sammon_mds(D, dim_val, c_huber, max_iter, tol);
    [X_irls, s_irls, keep_mask] = irls_sammon_mds(D, dim_val, c_irls, 50, tol, inner_max_iter);

    fprintf('维度 %d 完成：Classic stress=%.6f, Huber stress=%.6f, IRLS stress=%.6f\n', ...
            dim_val, s_classic, s_huber, s_irls);

    % 保存当前维度结果（覆盖）
    result = struct();
    result.dim_val = dim_val;
    result.classic_coords = X_classic;
    result.huber_coords   = X_huber;
    result.irls_coords    = X_irls;
    result.classic_stress = s_classic;
    result.huber_stress   = s_huber;
    result.irls_stress    = s_irls;
    result.irls_mask      = keep_mask;

    save(output_path, '-struct', 'result', '-v7.3');
    fprintf('结果已保存到 %s\n', output_path);
end

% ---- 辅助函数：计算距离矩阵 ----
function D = compute_D(C, mu, epsilon, L)
    C_abs = max(abs(C), 1e-10);
    D = epsilon * sqrt(abs(C_abs.^(-2/mu) - 1));
    over = D > L;
    D(over) = L * log(D(over) / L) + L;
    D(1:size(D,1)+1:end) = 0;
end

% 以下 classic_sammon_mds, huber_sammon_mds, irls_sammon_mds 函数保持不变

%% ========================================================================
%% Classic Sammon MDS
%% ========================================================================
function [X, stress] = classic_sammon_mds(D, n, max_iter, tol)
    N = size(D, 1);
    rng(42);
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
            X = [X_init, zeros(N, n - size(X_init, 2))];
        else
            X = X_init(:, 1:n);
        end
    catch ME
        % FIX: 修正 warning 调用，仅传入两个参数
        warning(ME.identifier, '%s', ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end

    mask = ~eye(N);
    W_base = zeros(N);
    W_base(mask) = 1 ./ D(mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;

    fprintf('Iter\tRelChange\tStress\n');

    for iter = 1:max_iter
        d = squareform(pdist(X));
        d(d == 0) = eps;

        W = W_base;
        W(1:N+1:end) = 0;

        V = diag(sum(W, 2)) - W;
        B = zeros(N);
        B(mask) = -W(mask) .* D(mask) ./ d(mask);
        B(1:N+1:end) = -sum(B, 2);

        lambda_reg = 1e-6;
        X_new = (V + lambda_reg * eye(N)) \ (B * X);

        if any(isnan(X_new(:))) || any(isinf(X_new(:)))
            X_new = pinv(V + lambda_reg * eye(N)) * (B * X);
        end

        X_new = X_new - mean(X_new, 1);
        change = norm(X_new - X, 'fro') / (norm(X, 'fro') + eps);
        X = X_new;

        if mod(iter, 100) == 0 || iter == 1 || change < tol
            d_cur = squareform(pdist(X));
            stress_cur = sum((D(mask) - d_cur(mask)).^2 ./ D(mask)) / sum(D(mask));
            fprintf('%d\t%.6f\t%.6f\n', iter, change, stress_cur);
        end

        if change < tol
            fprintf('Classic Sammon MDS converged at iteration %d\n', iter);
            break;
        end
    end

    d_final = squareform(pdist(X));
    stress = sum((D(mask) - d_final(mask)).^2 ./ D(mask)) / sum(D(mask));
end

%% ========================================================================
%% Huber Sammon MDS
%% ========================================================================
function [X, stress] = huber_sammon_mds(D, n, c, max_iter, tol)
    N = size(D, 1);
    rng(42);
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
            X = [X_init, zeros(N, n - size(X_init, 2))];
        else
            X = X_init(:, 1:n);
        end
    catch ME
        % FIX: 修正 warning 调用
        warning(ME.identifier, '%s', ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end

    mask = ~eye(N);
    W_base = zeros(N);
    W_base(mask) = 1 ./ D(mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;

    fprintf('Iter\tRelChange\tTrunc%%\n');

    for iter = 1:max_iter
        d = squareform(pdist(X));
        d(d == 0) = eps;

        abs_resid = zeros(N);
        abs_resid(mask) = abs(D(mask) - d(mask));

        W_huber = ones(N);
        W_huber(mask) = min(1, c ./ abs_resid(mask));

        W = W_base .* W_huber;
        W(1:N+1:end) = 0;

        V = diag(sum(W, 2)) - W;
        B = zeros(N);
        B(mask) = -W(mask) .* D(mask) ./ d(mask);
        B(1:N+1:end) = -sum(B, 2);

        lambda_reg = 1e-6;
        X_new = (V + lambda_reg * eye(N)) \ (B * X);

        if any(isnan(X_new(:))) || any(isinf(X_new(:)))
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
            fprintf('Huber Sammon MDS converged at iteration %d\n', iter);
            break;
        end
    end

    d_final = squareform(pdist(X));
    stress = sum((D(mask) - d_final(mask)).^2 ./ D(mask)) / sum(D(mask));
end

%% ========================================================================
%% IRLS Sammon MDS（额外返回收敛时的保留掩码，供 quasi-BIC 计算）
%% ========================================================================
function [X, stress, keep_mask] = irls_sammon_mds(D, n, c, max_iter, tol, inner_max_iter)
    N = size(D, 1);
    rng(42);
    try
        [X_init, ~] = cmdscale(D, n);
        if size(X_init, 2) < n
            X = [X_init, zeros(N, n - size(X_init, 2))];
        else
            X = X_init(:, 1:n);
        end
    catch ME
        % FIX: 修正 warning 调用
        warning(ME.identifier, '%s', ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end

    mask = ~eye(N);
    W_base = zeros(N);
    W_base(mask) = 1 ./ D(mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;

    fprintf('Outer\tInner\tRelChange\tThreshold\tTrunc%%\n');

    W_irls = ones(N);
    for outer = 1:max_iter
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
        % FIX: 防止 threshold 为 0 导致所有权重被截断
        if threshold <= 0
            threshold = 1e-12;
        end

        W_irls = zeros(N);
        W_irls(mask) = double(abs_resid(mask) <= threshold);

        W = W_base .* W_irls;
        W(1:N+1:end) = 0;

        if sum(W(:)) == 0
            error('IRLS: all weights truncated to zero, reduce c or check data');
        end

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
            fprintf('IRLS Sammon MDS converged at outer iteration %d\n', outer);
            break;
        end
    end

    keep_mask = W_irls > 0;

    d_final = squareform(pdist(X));
    stress = sum((D(mask) - d_final(mask)).^2 ./ D(mask)) / sum(D(mask));
end