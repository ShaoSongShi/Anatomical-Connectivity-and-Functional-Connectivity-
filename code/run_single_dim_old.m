function run_single_dim(input_path, output_path, dim_val)
    m = matfile(input_path, 'Writable', false);
    dims = m.dims(1, :);
    idx = find(dims == dim_val, 1);
    if isempty(idx)
        error('维度 %d 不在 dims 列表中', dim_val);
    end

    D_cell = m.D_cells(idx, 1);
    D = D_cell{1};
    D = (D + D') / 2;
    n_obs = size(D,1);
    D(1:n_obs+1:end) = 0;
    mask = ~eye(n_obs);
    D(mask & D <= 0) = 1e-6;

    fprintf('正在处理维度 %d (%d/%d)\n', dim_val, idx, length(dims));

    c_huber = 0.05;
    c_irls = 4.0;
    max_iter = 600;
    tol = 1e-6;
    inner_max_iter = 20;

    [X_classic, s_classic] = classic_sammon_mds(D, dim_val, max_iter, tol);
    [X_huber, s_huber] = huber_sammon_mds(D, dim_val, c_huber, max_iter, tol);
    [X_irls, s_irls, keep_mask] = irls_sammon_mds(D, dim_val, c_irls, 50, tol, inner_max_iter);

    fprintf('维度 %d 完成：Classic stress=%.6f, Huber stress=%.6f, IRLS stress=%.6f\n', ...
            dim_val, s_classic, s_huber, s_irls);

    % ---- 增量保存 ----
    if exist(output_path, 'file') == 2
        saved = load(output_path);
    else
        saved = struct();
    end

    n_dims = length(dims);
    if ~isfield(saved, 'dims')
        saved.dims = dims;
    end

    % 数值字段
    num_fields = {'classic_stress', 'huber_stress', 'irls_stress'};
    for f = num_fields
        if ~isfield(saved, f{1}) || isempty(saved.(f{1}))
            saved.(f{1}) = nan(n_dims, 1);
        end
    end

    % Cell 字段
    cell_fields = {'classic_coords', 'huber_coords', 'irls_coords', 'irls_masks'};
    for f = cell_fields
        if ~isfield(saved, f{1}) || isempty(saved.(f{1}))
            saved.(f{1}) = cell(n_dims, 1);
        end
    end

    saved.classic_stress(idx) = s_classic;
    saved.huber_stress(idx)   = s_huber;
    saved.irls_stress(idx)    = s_irls;
    saved.classic_coords{idx} = X_classic;
    saved.huber_coords{idx}   = X_huber;
    saved.irls_coords{idx}    = X_irls;
    saved.irls_masks{idx}     = keep_mask;

    save(output_path, '-struct', 'saved', '-v7.3');
    fprintf('结果已更新到 %s\n', output_path);
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
