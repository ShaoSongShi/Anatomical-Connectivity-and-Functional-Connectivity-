function run_batch_sammon(input_path, output_path)
%RUN_BATCH_SAMMON 批量运行三种 Sammon MDS，每个嵌入维度使用各自的距离矩阵 D_d
% 输入 prepared.mat 需包含：
%   dims    : 1 x n_dims  嵌入维度列表
%   D_cells : n_dims x 1 cell，每个元素为该维度对应的距离矩阵 D_d
%   Corr    : N x N 原始相关性矩阵（仅透传保存，Python 端使用）
% 输出 results.mat：坐标、stress、IRLS 最终保留掩码（供 BIC 计算）

    fprintf('Loading prepared data from: %s\n', input_path);
    data = load(input_path);
    dims = data.dims(:)';
    D_cells = data.D_cells;
    n_dims = length(dims);

    sampleD = D_cells{1};
    n_obs = size(sampleD, 1);
    fprintf('Data size: %d x %d, dimensions: ', n_obs, n_obs);
    fprintf('%d ', dims); fprintf('\n');

    % 预分配参数
    c_huber = 0.05;
    c_irls = 4.0;
    max_iter = 600;
    tol = 1e-6;
    inner_max_iter = 20;

    % 预分配 cell 和矩阵
    classic_coords = cell(n_dims, 1);
    classic_stress = zeros(n_dims, 1);

    huber_coords = cell(n_dims, 1);
    huber_stress = zeros(n_dims, 1);

    irls_coords = cell(n_dims, 1);
    irls_stress = zeros(n_dims, 1);
    irls_masks  = cell(n_dims, 1);   % IRLS 收敛时的保留掩码（BIC 用）

    for i = 1:n_dims
        n = dims(i);
        fprintf('\n========================================\n');
        fprintf('Dimension = %d (%d/%d)\n', n, i, n_dims);
        fprintf('========================================\n');

        % 每个维度使用各自的 D_d
        D = D_cells{i};
        D = (D + D') / 2;
        D(1:n_obs+1:end) = 0;
        mask = ~eye(size(D));
        D(mask & D <= 0) = 1e-6;   % 保证 1/D 权重有限
        fprintf('Min D: %.6e, Max D: %.6f\n', min(D(mask)), max(D(:)));

        % --- Classic ---
        fprintf('\n>>> Classic Sammon MDS\n');
        [X, s] = classic_sammon_mds(D, n, max_iter, tol);
        classic_coords{i} = X; classic_stress(i) = s;
        fprintf('    Stress=%.6f\n', s);

        % --- Huber ---
        fprintf('\n>>> Huber Sammon MDS (c=%.3f)\n', c_huber);
        [X, s] = huber_sammon_mds(D, n, c_huber, max_iter, tol);
        huber_coords{i} = X; huber_stress(i) = s;
        fprintf('    Stress=%.6f\n', s);

        % --- IRLS ---
        fprintf('\n>>> IRLS Sammon MDS (c=%.1f)\n', c_irls);
        [X, s, keep_mask] = irls_sammon_mds(D, n, c_irls, 50, tol, inner_max_iter);
        irls_coords{i} = X; irls_stress(i) = s; irls_masks{i} = keep_mask;
        fprintf('    Stress=%.6f | kept=%.2f%%\n', s, 100*sum(keep_mask(mask))/sum(mask(:)));
    end

    fprintf('\n========================================\n');
    fprintf('Saving results to: %s\n', output_path);
    save(output_path, 'dims', ...
         'classic_coords', 'classic_stress', ...
         'huber_coords', 'huber_stress', ...
         'irls_coords', 'irls_stress', 'irls_masks');
    % 若文件超过 2GB，改用 '-v7.3'：
    % save(output_path, 'dims', ...
    %      'classic_coords', 'classic_stress', ...
    %      'huber_coords', 'huber_stress', ...
    %      'irls_coords', 'irls_stress', 'irls_masks', '-v7.3');
    fprintf('Done.\n');
end

%% ========================================================================
%% 内嵌 cmdscale（零依赖）
%% ========================================================================
function [Y, eigvals] = cmdscale(D, p)
    N = size(D, 1);
    D2 = D.^2;
    J = eye(N) - ones(N, N) / N;
    B = -0.5 * J * D2 * J;
    [V, L] = eig((B + B') / 2);
    [eigvals, idx] = sort(diag(L), 'descend');
    V = V(:, idx);
    pos_idx = eigvals > 0;
    eigvals = eigvals(pos_idx);
    V = V(:, pos_idx);
    n_eig = min(p, length(eigvals));
    Y = V(:, 1:n_eig) * diag(sqrt(eigvals(1:n_eig)));
    eigvals = eigvals(1:n_eig);
end

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
        warning(ME.identifier, '%s', sprintf('CMDSCALE failed (%s), using random init', ME.message));
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
        warning(ME.message);
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
        warning(ME.message);
        rng(42);
        X = randn(N, n) * 0.01;
    end

    mask = ~eye(N);
    W_base = zeros(N);
    W_base(mask) = 1 ./ D(mask);
    W_base(isinf(W_base) | isnan(W_base)) = 0;

    fprintf('Outer\tInner\tRelChange\tThreshold\tTrunc%%\n');

    W_irls = ones(N);   % 若首轮即收敛也能返回合法掩码
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
