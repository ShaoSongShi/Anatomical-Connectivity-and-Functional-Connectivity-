using Random, StatsBase, Printf, Distributed, LinearAlgebra, Dates
using Plots, Distributions
using Hungarian
using ToeplitzMatrices
using Distances
using MultivariateStats, Statistics, Clustering, StatsPlots, Bootstrap
using LaTeXStrings
using MAT, Distributions
using LowRankApprox, MATLAB
using JSON, SHA, BSON
using InvertedIndices
using SpecialFunctions
using Roots, ForwardDiff
using QuadGK, SpecialFunctions
using MDBM
import Base.*
using CurveFit
using NearestNeighbors
using HypergeometricFunctions
using KernelDensity, Interpolations, DataInterpolations, KernelFunctions
using Flux
using Flux: @epochs
using CUDA, CUDAKernels, KernelAbstractions
using Tullio
using GaussianMixtures
using Parameters


fit = Distributions.fit


@with_kw mutable struct ERMParameter
    N::Int      #number of neurons
    L::Float64  #region size 2*r
    ρ::Float64  #density
    n::Int      # dimensionality
    ϵ::Float64  # parameter in distance function f
    μ::Float64  # parameter in distance function f
    ξ::Float64  # parameter in distance function f
    β::Float64  # parameter in distance function f β = 1/ξ
    σ̄²::Float64 # mean variance of neural activity
    σ̄⁴::Float64  # mean 4th moment of neural activity
end

# Convenience constructor for β = 1/ξ
ERMParameter(;N, L, ρ, n, ϵ, μ, ξ, σ̄², σ̄⁴) = ERMParameter(N, L, ρ, n, ϵ, μ, ξ, 1/ξ, σ̄², σ̄⁴)
#=
@with_kw mutable struct ERMParameter
    N::Int      #number of neurons
    L::Float32  #region size 2*r
    ρ::Float32  #density
    n::Int      # dimensionality
    ϵ::Float32  # parameter in distance function f
    μ::Float32  # parameter in distance function f
    ξ::Float32  # parameter in distance function f 
    β::Float32 = 1/ξ # parameter in distance function f β = 1/ξ
    σ̄²::Float32 # mean variance of neural activity
    σ̄⁴::Float32  # mean 4th moment of neural activity
end
=#

function collapse_index_rank(C; p0 = 0.01, ntrial =2000, sample = "random")
    
    s = 0.5 # subsampling fraction
    #p0 = 0.01 # lambda_max cutoff

    N,_ = size(C)
    Ns = round(Int, N*s)
    lb_N = -sort(-eigvals(Hermitian(C)))
    #ntrial = 20
    lb_Ns = zeros(Ns,ntrial)
    Threads.@threads for i in 1:ntrial
        if sample == "random"
            i_s = randperm(N)[1:Ns]
            Cs = copy(C[:,i_s])
            Cs = Cs[i_s,:]
            lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs)))
        elseif sample == "RG"
            i_s_1 = RG_clusters(C)[end-1][1]
            i_s_2 = RG_clusters(C)[end-1][2]
            Cs1 = copy(C[:,i_s_1])
            Cs1 = Cs1[i_s_1,:]
            lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs1)))
            Cs2 = copy(C[:,i_s_2])
            Cs2 = Cs2[i_s_2,:]
            lb_Ns[:,i] += -sort(-eigvals(Hermitian(Cs2)))
            lb_Ns[:,i] = lb_Ns[:,i]/2
        end
    end
    lb_Ns = vec(mean(lb_Ns,dims=2))

    r0 = round(Int, Ns*p0-0.5) # lambda_max cutoff
    r1 = maximum(findall(lb_Ns .> 1)) # lambda_min cutoff
    f_Ns = log.(lb_Ns[r0+1:r1])
    f_N = zeros(length(f_Ns))
    for k in 1:length(f_N)
        i = r0 + k
        j = (i)/Ns*N
        if isinteger(j)
            j = convert(Int,j)
            f_N[k] = log.(lb_N[j])
        else
            # interpolation in loglog scale
            j0 = convert(Int,floor(j))
            # f_N[k] = (j0+1-j)*lb_N[j0] + (j-j0)*lb_N[j0+1]
            #=
            x = np.log(j/N)
            x0 = np.log(j0/N)
            x1 = np.log((j0+1)/N)
            y0 = np.log(lb_N[j0])
            y1 = np.log(lb_N[j0+1])
            =#
            x = log(j/N)
            x0 = log(j0/N)
            x1 = log((j0+1)/N)
            y0 = log(lb_N[j0])
            y1 = log(lb_N[j0+1])
            f_N[k] = (y0*(x1-x) + y1*(x-x0)) / (x1-x0)
        end
    end
    S = 0
    for k in 1:length(f_N)-1
        i = r0 + k
        a = (i)/Ns
        b = (i+1)/Ns
        y1 = abs(f_N[k] - f_Ns[k])
        y2 = abs(f_N[k+1] - f_Ns[k+1])
        S += log(b/a)*(y1+y2)/2
    end
    CI = S/log(N/Ns)/log((r1)/(r0+1))
    return CI, (r0+1)/Ns, r1/Ns
end

function robust_matrix_square_root(C; rank=nothing, condition_number_threshold=1e10)
    """
    更稳健的矩阵平方根计算，处理低秩和病态矩阵
    
    参数:
    C: 对称半正定矩阵
    rank: 指定秩，如果为nothing则自动确定
    condition_number_threshold: 条件数阈值，用于自动确定秩
    
    返回:
    W: 矩阵平方根，满足 W * W' ≈ C
    """
    # 确保对称性
    C_sym = (C + C') / 2
    
    # 特征分解
    eigen_decomp = eigen(Symmetric(C_sym))
    eigenvalues = eigen_decomp.values
    eigenvectors = eigen_decomp.vectors
    
    # 按特征值大小排序（降序）
    idx = sortperm(eigenvalues, rev=true)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 处理数值误差：移除负的特征值
    eigenvalues = max.(eigenvalues, 0.0)
    
    # 确定有效秩
    if rank === nothing
        # 基于条件数自动确定秩
        max_eig = eigenvalues[1]
        if max_eig > 0
            valid_mask = eigenvalues .> max_eig / condition_number_threshold
            rank = sum(valid_mask)
        else
            rank = 0
        end
    else
        rank = min(rank, length(eigenvalues))
    end
    
    println("计算矩阵平方根:")
    println("  原始矩阵维度: $(size(C))")
    println("  最大特征值: $(eigenvalues[1]):.6f")
    println("  最小特征值: $(eigenvalues[end]):.6f")
    println("  使用秩: $rank")
    
    if rank == 0
        return zeros(size(C))
    end
    
    # 截断到有效秩
    eigenvalues_trunc = eigenvalues[1:rank]
    eigenvectors_trunc = eigenvectors[:, 1:rank]
    
    # 计算平方根
    sqrt_eigenvalues = sqrt.(eigenvalues_trunc)
    
    # 重构矩阵
    W = eigenvectors_trunc * Diagonal(sqrt_eigenvalues) * eigenvectors_trunc'
    
    return W
end

function verify_decomposition(C, W; tolerance=1e-10)
    """验证分解的正确性"""
    reconstruction = W * W'
    error = maximum(abs.(C - reconstruction))
    is_symmetric = all(isapprox.(C, C', atol=tolerance))
    
    println("原始矩阵尺寸: $(size(C))")
    println("重构误差: $error")
    println("原始矩阵是否对称: $is_symmetric")
    println("重构是否成功: $(error < tolerance)")
    
    return error < tolerance
end

function iterative_projection_WWT(A::Matrix{Float64}, i::Int, j::Int, kind::Int,
                                max_iter=1000, tol=1e-12)
    n = size(A, 1)

    if any(isnan, A) || any(isinf, A)
        @warn "矩阵包含NaN或Inf值，进行清理"
        A = replace(A, NaN => 0.0, Inf => 1e10, -Inf => -1e10)
    end
    
    # 初始解和约束值
    # 使用SVD而不是Cholesky，更稳健
    F = svd(A)
    
    # 处理负奇异值（如果有）
    S_positive = max.(F.S, 0.0)
    
    # 构造初始W
    W = F.U * Diagonal(sqrt.(S_positive))

    if kind == 1
        target_value = sqrt(W[i, j])
    elseif kind == 2
        target_value = 0
    end
    
    @info "初始约束值" original_W_ij=W[i,j] target_value=target_value
    
    for iter in 1:max_iter
        W_prev = copy(W)
        
        # 步骤1: 投影到约束集（固定特定元素）
        W[i, j] = target_value
        W[j, i] = target_value
        
        # 步骤2: 投影到WWᵀ = A的流形
        # 使用极分解找到最近的满足WWᵀ = A的矩阵
        U, S, Vt = svd(W)
        W = U * Vt'
        
        # 重新缩放以满足WWᵀ = A
        M = W * W'
        F = eigen(Symmetric(M))
        Λ_inv_sqrt = Diagonal(1.0 ./ sqrt.(F.values))
        scaling = F.vectors * Λ_inv_sqrt * F.vectors'
        W = scaling * W
        
        # 检查收敛
        if norm(W - W_prev) < tol
            @info "收敛于迭代" iteration=iter
            break
        end
        
        if iter == max_iter
            @warn "达到最大迭代次数未收敛"
        end

    end

    
    # 最终验证
    final_error = norm(W * W' - A)
    constraint_error = abs(W[i,j] - target_value) + abs(W[j,i] - target_value)
    
    @info "最终结果" WWT_error=final_error constraint_error=constraint_error
    @info "约束检查" W_ij=W[i,j] W_ji=W[j,i] target=target_value
    
    return W
end

function add_p_coupling_alignment_W(W_coupling_alignment; p_uni, p_bi)
    println("=== add_p_coupling_alignment_W 开始执行 ===")
    println("输入矩阵尺寸: ", size(W_coupling_alignment))
    println("非零元素数量: ", count(!iszero, W_coupling_alignment))
    
    # 为每对神经元(i,j)分配连接类型
    N = size(W_coupling_alignment,1)
    println("神经元数量 N: ", N)
    
    changed_pairs = 0
    
    for i in 1:N
        for j in 1:i  # 只遍历上三角
            if i != j 
                if W_coupling_alignment[i,j] != 0
                    #println("找到非零连接: ($i, $j) = ", W_coupling_alignment[i,j])
                    
                    # 根据概率随机选择连接类型
                    r = rand()
                    p_total = p_uni*2 + p_bi
                    #println("随机数 r = $r, 阈值 = $(p_uni/p_total), $(2*p_uni/p_total)")
                    
                    if r < p_uni/p_total
                        # 不变
                        #println("($i, $j): 不改变连接方向")
                        changed_pairs += 1
                        
                    elseif r < 2*p_uni/p_total
                        # 情况2: 单向连接 j→i  
                        #println("($i, $j): 改变连接方向 i→j → j→i")
                        W_coupling_alignment[j, i] = W_coupling_alignment[i, j]  # j 连接到 i
                        W_coupling_alignment[i, j] = 0.0  # i 不连接到 j
                        changed_pairs += 1
                        println("change11")
                    else
                        # 情况3: 双向连接
                        #println("($i, $j): 改为双向连接")
                        # 临时简化处理 - 先确保双向都有值
                        W_coupling_alignment = iterative_projection_WWT(W_coupling_alignment, i, j, kind=1,
                                max_iter=1000, tol=1e-12)
                        changed_pairs += 1
                        println("change2")
                    end
                end
            end
        end
    end
    
    println("=== add_p_coupling_alignment_W 执行结束 ===")
    println("总共处理了 $changed_pairs 对连接")
    return W_coupling_alignment
end

function real_weigh_matrix(file_path::String)
    # 读取文件所有行
    lines = readlines(file_path)
    
    # 获取矩阵大小（跳过第一行和第一列）
    n = length(lines) - 1
    W = zeros(Float64, n, n)
    
    for (i, line) in enumerate(lines)
        if i == 1  # 跳过第一行（Julia索引从1开始）
            continue
        end
        
        parts = split(strip(line), ',')
        for (j, part) in enumerate(parts)
            if j == 1  # 跳过第一列
                continue
            end
            W[i-1, j-1] = parse(Float64, part)
        end
    end
    
    return W
end

function compute_covariance_matrix(W; method=:cholesky)
    N = size(W, 1)
    I_matrix = Matrix{Float64}(I, N, N)
    M = I_matrix - W
    M = (M + M') / 2  # 确保对称性
    @debug "compute_covariance_matrix 开始" method N
    
    if method == :cholesky
        try
            @debug "尝试Cholesky分解"
            F = cholesky(M)
            return F \ I_matrix
        catch e
            println("Cholesky失败，尝试LU分解: ", e)
            method = :lu
        end
    end
    
    if method == :lu
        F = lu(M)
        return F \ I_matrix
    end
    
    # 方法3：直接求逆（最后的选择）
    return inv(I_matrix - W) * I_matrix * inv(I_matrix - W)'
end

# 如果需要使用连接矩阵 A
function generate_covariance_from_connectivity(A, p::ERMParameter, Δ)
    N = size(A,1)
    
    # 方法2.1: 线性动力学模型
    # C = (I - A)^{-1} * (I - A^T)^{-1}
    J = copy(A)  # 连接矩阵作为权重
    C = compute_covariance_matrix(J, method=:cholesky)
    
    # 方法2.2: 非线性情况的近似
    # C ≈ (I - αA)^{-1} * (I - αA^T)^{-1}，其中α是缩放参数
    
    # 加入神经元特异性方差
    Ĉ = Δ * C * Δ
    Ĉ = (Ĉ + Ĉ')/2  # 确保对称性
    
    return Ĉ
end

function generate_connectivity_matrix(N; p_uni, p_bi, spatial_points=nothing, distance_dependence=false)
    """
    生成具有精确单向和双向连接概率的连接矩阵。

    参数:
    - N: 神经元数量
    - p_uni: 单向连接的概率 (对于每个有序对 i≠j)
    - p_bi: 双向连接的概率 (对于每个无序对 i≠j)
    - spatial_points: 神经元空间位置 (N×d矩阵)，如果为nothing则忽略空间依赖
    - distance_dependence: 是否让概率随距离变化
    """
    @unpack μ, ϵ, β = p
    c = (1/ϵ^μ)^(-2/μ)
    
    # 计算无连接的概率以确保概率归一化
    p_none = 1.0 - 2*p_uni - p_bi
    
    # 验证概率有效性
    if p_none < 0 || p_none > 1
        error("无效的概率参数: p_uni=$p_uni, p_bi=$p_bi (必须满足 2*p_uni + p_bi ≤ 1)")
    end
    
    # 初始化连接矩阵
    A = zeros(N, N)
    
    # 为每对神经元(i,j)分配连接类型
    for i in 1:N
        for j in 1:(i-1)  # 只遍历下三角，避免重复
            if i != j
                # 如果有空间依赖，调整概率
                if distance_dependence && spatial_points !== nothing
                    dist = norm(spatial_points[i, :] - spatial_points[j, :])
                    # 基于距离调整概率（示例：高斯衰减）
                    spatial_factor =  ϵ^μ*(c + dist^2)^(-μ/2)
                    current_p_bi = p_bi * spatial_factor
                    current_p_uni = p_uni * spatial_factor
                    current_p_none = 1.0 - 2*current_p_uni - current_p_bi
                    
                    # 重新归一化
                    total = current_p_none + 2*current_p_uni + current_p_bi
                    current_p_none /= total
                    current_p_uni /= total
                    current_p_bi /= total
                else
                    current_p_none = p_none
                    current_p_uni = p_uni  
                    current_p_bi = p_bi
                end
                
                # 根据概率随机选择连接类型
                r = rand()
                
                if r < current_p_none
                    # 情况1: 无连接 - 什么都不做
                    A[i, j] = 0.0
                    A[j, i] = 0.0
                elseif r < current_p_none + current_p_uni
                    # 情况2: 单向连接 i→j
                    A[i, j] = 1.0  # i 连接到 j
                    A[j, i] = 0.0  # j 不连接到 i
                elseif r < current_p_none + 2*current_p_uni
                    # 情况3: 单向连接 j→i  
                    A[i, j] = 0.0  # i 不连接到 j
                    A[j, i] = 1.0  # j 连接到 i
                else
                    # 情况4: 双向连接
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                end
            end
        end
    end
    
    return A
end

"""
生成神经元活动时间序列数据
"""
function generate_neural_activity(C::Matrix{Float64}, A::Matrix{Float64}, 
                                T::Int=1000, input_strength::Float64=1.0,
                                noise_level::Float64=0.1, dt::Float64=0.01)
    """
    参数:
    C: 协方差矩阵 (N x N)
    A: 连接矩阵 (N x N)
    T: 时间步数
    input_strength: 输入强度
    noise_level: 噪声水平
    dt: 时间步长
    """
    N = size(C, 1)
    
    # 1. 生成初始状态（基于协方差矩阵）
    μ = zeros(N)
    initial_activity = rand(MvNormal(μ, C), 1)[:, 1]
    
    # 2. 生成输入信号（可以是随机的或特定的模式）
    input_signal = input_strength * randn(N, T)
    
    # 3. 模拟动力学
    activity = zeros(N, T)
    activity[:, 1] = initial_activity
    
    for t in 2:T
        # 当前状态
        x_prev = activity[:, t-1]
        
        # 连接输入
        recurrent_input = A * tanh.(x_prev)
        
        # 噪声
        noise = noise_level * randn(N)
        
        # 更新方程 (简单的连续时间RNN)
        dx = -x_prev + recurrent_input + input_signal[:, t] + noise
        activity[:, t] = x_prev + dt * dx
    end
    
    return activity
end

"""
计算总相关性 (Total Correlation)
"""
function total_correlation(data::Matrix{Float64})
    """
    总相关性: TC = Σ H(x_i) - H(X)
    其中 H 是熵
    """
    N, T = size(data)
    
    # 计算个体熵
    individual_entropies = 0.0
    for i in 1:N
        marginal_entropy = 0.5 * log(2π * e * var(data[i, :]))
        individual_entropies += marginal_entropy
    end
    
    # 计算联合熵
    cov_matrix = cov(data, dims=2)
    joint_entropy = 0.5 * log((2π * e)^N * det(cov_matrix))
    
    # 总相关性
    TC = individual_entropies - joint_entropy
    
    return TC
end

"""
计算O-信息 (Organizational Information)
"""
function o_information(data::Matrix{Float64})
    """
    O-信息 = (n-2) * 多信息 + Σ ΔTC
    提供了系统组织复杂度的度量
    """
    N, T = size(data)
    
    # 计算多信息 (Multi-information)
    multi_info = total_correlation(data)
    
    # 计算ΔTC (每个神经元移除后的TC变化)
    delta_TC_sum = 0.0
    TC_full = total_correlation(data)
    
    for i in 1:N
        # 移除第i个神经元
        reduced_data = data[setdiff(1:N, i), :]
        TC_reduced = total_correlation(reduced_data)
        delta_TC = TC_full - TC_reduced
        delta_TC_sum += delta_TC
    end
    
    # O-信息
    O_info = (N - 2) * multi_info + delta_TC_sum
    
    return O_info, multi_info, delta_TC_sum / N
end

"""
计算各种信息理论量
"""
function compute_information_measures(activity::Matrix{Float64})
    N, T = size(activity)
    
    # 1. 总相关性
    TC = total_correlation(activity)
    
    # 2. O-信息
    O_info, multi_info, avg_delta_TC = o_information(activity)
    
    # 3. 互信息矩阵
    mutual_info_matrix = zeros(N, N)
    for i in 1:N
        for j in i+1:N
            # 简化版的互信息估计 (基于高斯假设)
            cov_ij = cov(activity[[i,j], :]')
            mi = 0.5 * log(det(cov_ij) / (var(activity[i, :]) * var(activity[j, :])))
            mutual_info_matrix[i, j] = max(0, mi)
            mutual_info_matrix[j, i] = mutual_info_matrix[i, j]
        end
    end
    
    # 4. 集成复杂度 (Integration)
    integration = TC
    
    return (TC = TC, 
            O_info = O_info, 
            multi_info = multi_info,
            avg_delta_TC = avg_delta_TC,
            mutual_info_matrix = mutual_info_matrix,
            integration = integration)
end

"""
主函数：完整的分析流程
"""
function analyze_network_dynamics(C::Matrix{Float64}, A::Matrix{Float64};
                                T::Int=1000, 
                                input_strengths::Vector{Float64}=[0.1, 0.5, 1.0, 2.0],
                                n_trials::Int=5)
    
    results = Dict()
    
    for input_strength in input_strengths
        trial_results = []
        
        for trial in 1:n_trials
            println("输入强度: $input_strength, 试验: $trial")
            
            # 生成活动数据
            activity = generate_neural_activity(C, A, T, input_strength)
            
            # 计算信息度量
            info_measures = compute_information_measures(activity)
            
            push!(trial_results, (activity = activity, info = info_measures))
        end
        
        results[input_strength] = trial_results
    end
    
    return results
end

"""
可视化结果
"""
function plot_results(results::Dict)
    input_strengths = sort(collect(keys(results)))
    
    # 准备数据
    TC_means = []
    O_info_means = []
    TC_stds = []
    O_info_stds = []
    
    for strength in input_strengths
        trials = results[strength]
        TCs = [trial.info.TC for trial in trials]
        O_infos = [trial.info.O_info for trial in trials]
        
        push!(TC_means, mean(TCs))
        push!(O_info_means, mean(O_infos))
        push!(TC_stds, std(TCs))
        push!(O_info_stds, std(O_infos))
    end
    
    # 绘制总相关性随输入强度的变化
    p1 = plot(input_strengths, TC_means, ribbon=TC_stds, 
             label="Total Correlation", xlabel="Input Strength", 
             ylabel="TC", title="总相关性 vs 输入强度",
             linewidth=2, marker=:circle)
    
    # 绘制O-信息随输入强度的变化
    p2 = plot(input_strengths, O_info_means, ribbon=O_info_stds,
             label="O-Information", xlabel="Input Strength", 
             ylabel="O-info", title="O-信息 vs 输入强度",
             linewidth=2, marker=:square, color=:red)
    
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

function degree_distribution(W::Array; ϵ = 1f-4)

    N = size(W,1)
    A = copy(W)
    #adjacency matrix
    A[abs.(W) .< ϵ] .= 0.0
    A[abs.(W) .> ϵ] .= 1.0
    A = A .- diagm(ones(N))
    k_in = vec(sum(A, dims=2))
    k_out = vec(sum(A, dims=1))

    xrange_in = range(minimum(k_in),stop=maximum(k_in), length=50)
    h_in = fit(Histogram, k_in, xrange_in)
    h_in = normalize(h_in, mode=:pdf)
    x_in, p_in = dropzero(h_in) #delete bins with weights = 0

    xrange_out = range(minimum(k_out), stop=maximum(k_out), length=50)
    h_out = fit(Histogram, k_out, xrange_out)
    h_out = normalize(h_out, mode=:pdf)
    x_out, p_out = dropzero(h_out)

    plot(x_in, p_in, markershape=:circle, label = "in degree" )
    plot!(x_out, p_out, markershape=:cross, label = "out degree" )
    plot!(xlabel = "k", ylabel = "pdf", yaxis = :log)
    png("figures/degree_distribution.png")

    return k_in, k_out

end

function weight_distribution(W::Array, filename; xlimit = (1f-3, 1), xlabels = "weights", space = "linspace")

    weights = vec(W)
    weights = weights[weights.>0]
    if space == "linspace"
        w_range = range(minimum(weights), stop=maximum(weights), length=100)
    elseif space == "logspace"
        e1 = log10(minimum(weights))
        e2 = log10(maximum(weights))
        w_range = 10 .^ range(e1,e2,length = 50)
    end

    h_w = fit(Histogram, weights, w_range)
    h_w = normalize(h_w, mode=:pdf)

    x, p = dropzero(h_w)
    plot(x, p, markershape=:auto)
    plot!(xlabel = xlabels, ylabel = "pdf", xlims=xlimit, xaxis = :log, yaxis = :log)
    png("figures/"*"$filename")

    return nothing

end

function dropzero(h::Histogram)
    @unpack edges, weights = h
    binsize = edges[1][2] - edges[1][1]
    x = collect(edges[1][2:end].-binsize/2)
    #center of a bin
    idx = findall(iszero,weights)
    deleteat!(x,idx)
    deleteat!(weights,idx)
    return x, weights
end

function unpack_histogram(h::Histogram)
    @unpack edges, weights = h
    binsize = edges[1][2] - edges[1][1]
    x = collect(edges[1][2:end].-binsize/2)
    #center of a bin
    return x, weights
end

function dropzeros_and_less(y::Vector)
    x = collect(1:length(y))
    real_y = real.(y)
    idx = findall(real_y.<=0)
    deleteat!(x,idx) 
    deleteat!(real_y,idx)
    return x, real_y
end


function find_hubs(y::Vector, ip; nums = 10)
    p = sortperm(y,rev=true)
    idx = p[1:nums]
    return idx, ip[idx]
end

function find_hubs(y::Vector; nums = 10)
    p = sortperm(y,rev=true)
    return p[1:nums]
end

function sample_from_powerlaw(;xmin=0.1, xmax=10, α = -2.9, N = 1000, trials = 21)
    #draw random samples from a powerlaw pdf p(x) ~ xᵅ where x ∈ [xmin, xmax]    

    y = rand(Uniform(0, 1), N, trials)
    return ((xmax^(α+1) - xmin^(α+1)).*y .+ xmin^(α+1)).^(1/(α+1))

end

compute_P0!(x) = length(findall(iszero, x))/length(x)
#compute the probability that a neuron is not active

function set_initialcondition!(x; N = 100, trials = 21)

    P0 = compute_P0!(x)
    rnd = rand(Bernoulli(1-P0), N, trials)
    h0 = rnd.*sample_from_powerlaw(N = N, trials = trials)

    return h0

end

function normalization(x; ϵ = 1.0e-6)
    a = copy(x)
    a[a.<ϵ] .= 0
    for n in 1:size(a,1)
        x̄ = mean(a[n,a[n,:].>0])
        if (x̄ > 0)
            a[n,:] = a[n,:]./x̄
        end
    end
    return a
end

function construct_Toeplitz(Cᵣ::AbstractMatrix)

    #c = mean(sort(Cᵣ, dims = 2, rev = true), dims = 1)

    iterations = 100
    N = size(Cᵣ,1)
    sums = zeros(Float32,N)
    c = copy(sums)
    for n in 1:iterations
        sample!(Cᵣ, c)
        sort!(c, rev = true)
        sums = sums + c
    end
    c = sums/iterations

    return convert(Matrix{Float32}, Toeplitz(c,c))

end

function construct_Toeplitz(N::Int; a = 1, b = -0.4, γ = 2)
    x = collect(1:N)
    c = a*x.^b.*exp.((x/N).^γ)
    return Toeplitz(c,c)
end

function generate_heatmap(Cᵣ,filename; climit = (0, 1), cbar = :thermal, xlabel = "neuron #", ylabel = "neuron #")

    gr(format=:png) #faster plotting heatmap
    theme(:dark)
    N = size(Cᵣ,1)
    M = size(Cᵣ,2)

    display(heatmap(Cᵣ, color=cgrad(cbar),clim=climit, 
            aspect_ratio=1, xlims=(1,M), ylims=(1, N),
            xlabel = xlabel,ylabel= ylabel, title = filename))
    png("figures/"*"$filename")

end

function visualize_fr(firingrate,filename; climit = (0, 1), cbar = :ice)

    gr()
    theme(:dark)
    N = size(firingrate,1)
    M = size(firingrate,2)
    firingrate = firingrate .- mean(firingrate,dims=2)

    heatmap(firingrate, color=cgrad(cbar),clim=climit, 
            xlims=(1, M), ylims=(1, N),
            xlabel="frame #",ylabel="neuron #")
    png("$filename")

end

function diff_cost(U,H,ip,a,b,p)

    k1=ip[a]
    k2=ip[b]
    ip_new = ip[p]
    r1, r2, a1, a2 = 0, 0, 0, 0
    @inbounds @simd for i=1:length(ip)
        r1 += (H[i,a] - U[ip[i],k1])^2
        r2 += (H[i,b] - U[ip[i],k2])^2
        a1 += (H[i,a] - U[ip_new[i],k2])^2 
        a2 += (H[i,b] - U[ip_new[i],k1])^2
    end

    reduced= 2*(r1+r2) - (H[a,a] - U[k1,k1])^2 - (H[b,b]-U[k2,k2])^2
    increased= 2*(a1+a2) - (H[a,a] - U[k2,k2])^2 - (H[b,b] - U[k1,k1])^2
    dE = increased - reduced
    return dE

end


function min_fnorm(A, B, ip; MAX_STEPS=3000, t = 20.0)

    #find a permutation matrix that would minimize ||P*A*P'-B|| via simulated annealing
    #initial temperature
    Tfactor=0.999
    nover=10000
    #temperature will change to t*Tfactor after nover times trials of random swapping
    nlimit=1000
    #In one loop with nover trials, if nlimit swaps whose cost change
    #satisfies the transition probability criterion, I allow the temperature to change
    #immediately, rather than going over all the nover trials.

    n = size(A,1)
    P = I(n)[ip,:]

    Cost = (norm(P*A*P'-B))^2
    
    
    s = @sprintf "initial Frobenius distance is %.1f" Cost
    println(s)

    E = Float64[];
    T = Float64[];
    push!(E,Cost)
    push!(T,t)

    @inbounds for j in 2:MAX_STEPS
        #The maximum number of annealing steps is MAX_STEPS. After each step, the
        #temprature changes by t*Tfactor.
        nsucc, n_up, n_down = 0, 0, 0
        
        @inbounds for k in 1:nover
            (a, b) = samplepair(n)
            #randomly swap two nodes whose indexes are given by a and b in the
            #look up table iorder.
            p = collect(1:n)
            p[a] = b
            p[b] = a  
           
            dE = diff_cost(A,B,ip,a,b,p)
           
            #the mismatch cost difference  before swapping and after swapping is calculated in function edit_distance_diff.  
            #The calculation is first done for chemical synapses.
                
            answer=(dE<0)||(first(rand(1)) < exp(-dE/t))
            #if dD_E<0, or if dD_E>0 and rand(1)<exp(-dD_E/t), the swaps are allowed. 
            if answer
                nsucc=nsucc+1
                if dE < 0
                    n_down = n_down + 1
                elseif dE > 0
                    n_up = n_up + 1
                end
                Cost = Cost + dE
                ip = ip[p] 
                #The new look up table is updated.
            end
        
            if (nsucc>=nlimit)
                break
            end
            #if there are nlimit successful swaps, break the loop and change
            #the temperature.
        
        end
    
        push!(E,Cost)
        push!(T,t)
        s = @sprintf "Energy cost is E=%.1f, T=%.1f" Cost t
        println(s)
        println("Successful Moves: $nsucc")
        println("Moving up: $n_up, Moving down: $n_down")
        t=t*Tfactor;
        if (nsucc==0)
            return ip, E, T
        end
        #If  there are no successful swaps, we find the minimum.
    end

    return ip, E, T

end


function Umeyama(A,B)

    N = size(A,1)
    Uₐ = eigvecs(A)
    Uᵦ = eigvecs(B)
    Ūₐ = abs.(reverse(Uₐ,dims=2))
    Ūᵦ = abs.(reverse(Uᵦ,dims=2))
    weight = -Ūₐ*transpose(Ūᵦ)
    matching = Hungarian.munkres(weight)
    ip = [findfirst(matching[i,:].==Hungarian.STAR) for i = 1:N]

    return ip

end

function distance_measure(Coordinates, P)

    Coord = P*Coordinates
    R = pairwise(Euclidean(),Coord,dims=1)  

    N = size(P,1)
    r = zeros(N)
    #mean pairwise distance
    #check wether r = f(|i-j|)

    @inbounds @simd for i in 1:N
        @inbounds @simd for j in 1:N
                            idx = abs(j-i)+1
                            r[idx] += R[j,i]
        end
    end 
    
    for i in 1:N-1
        r[i+1] = r[i+1]/(2*(N-i))
    end
    r[1] = r[1]/N

    plot(collect(1:N-1), r[2:end], markershape=:auto)
    plot!(xlabel = "rank", ylabel = "distance", xaxis = :log, yaxis = :log)
    png("figures/rank-distance.png")

    R = R./mean(R[:])
    theme(:dark)
    heatmap(R, color=cgrad(:thermal),clim=(0, 2), aspect_ratio=1, xlims=(1,N), ylims=(1, N),xlabel="neuron #",ylabel="neuron #")
    png("figures/pairwise_distance.png")
    return R
end


function C_projection(r; dims=1)

    if isempty(dims) 
        return r
    else
        #### PCA #######
        M = fit(PCA, r)
        Pr = projection(M)
        #principle components
        Pₛ = Pr[:,Not(dims)]
        #Pₛ = Pr[:,1:dims] 
        #exclude the first dims component
        Yₛ = transform(M,A)[Not(dims),:]
        #Yₛ = transform(M,A)[1:dims,:]
        #scores

        Aₛ = Pₛ*Yₛ
        #Cᵣ = Aₛ*Aₛ'/T
        return Aₛ # return the reconstructed activity in the subspace
    end

end

function expand(indices_bound, N)

    # (a, b, c, d) --> [a:b] \union [c:d]  

    K = length(indices_bound)
    if isodd(K)
        indices_bound = tuple(indices_bound...,N...)
        K=K+1
    end

    indices = collect(indices_bound[1]:indices_bound[2])
    
    for j=3:2:K-1
        append!(indices,collect(indices_bound[j]:indices_bound[j+1]))
    end
    
    return indices

end

function mds(Covariance_matrix::AbstractMatrix, ip::Vector, indices_bound...)
    
    N = size(Covariance_matrix,1)
    P = I(N)[ip,:]
    G = *(P,Covariance_matrix,P')
    D = gram2dmat(G)
    M = fit(MDS,D,distances=true)
    embedding = transform(M)
    indices = expand(indices_bound,N)
    return embedding[:,indices], indices

end

function mds(Coordinates::AbstractMatrix, r::AbstractMatrix, ip::Vector, indices_bound...; λ=1, dims=2)

    (N,T) = size(r)
    P = I(N)[ip,:]
    #r = normalization(P*r,ϵ = 1.0e-4)
    #r = r .- mean(r,dims=2)

    indices = expand(indices_bound,size(r,1))
    
    
    #D2 = pairwise(CorrDist(),r[indices,:], dims=1)


    # d²[i,j] = σ²[i] + σ²[j] - 2*C[i,j]
        
    G = r[indices,:]*r[indices,:]'/T
    D2=gram2dmat(G)
    
    #

    Coord = P*Coordinates
    D1 = pairwise(Euclidean(),Coord[indices,:],dims=1)  
    D1 = D1./mean(D1[:])
    D = D2 + λ*D1
    embedding=classical_mds(D, dims)

    return embedding, D, indices

end



function plot_coordinates(Coordinates,indices,filename;λ = 0)

    gr()
    theme(:dark)

    dims = size(Coordinates,1)

    if dims>3
        throw("cannot plot figures that are more than three dimensions!")
    end

    Coordinates = Coordinates .- mean(Coordinates,dims=2)

    if dims==3

        x = Coordinates[1,:]
        y = Coordinates[2,:]
        z = Coordinates[3,:]
        l = @layout [a b;c d]

        p1 = scatter(x,y,zcolor=indices, color=cgrad(:rainbow), aspect_ratio = 1, 
                title="x-y", markersize = 2, markerstrokewidth = 0, cbar = false, legend = false)
        p2 = scatter(z,y,zcolor=indices, color=cgrad(:rainbow), aspect_ratio = 1, 
                title="z-y", markersize = 2, markerstrokewidth = 0, cbar = false, legend = false)
        p3 = scatter(x,z,zcolor=indices, color=cgrad(:rainbow), aspect_ratio = 1, 
                title="x-z", markersize = 2, markerstrokewidth = 0, cbar = false, legend = false)
        p4 = scatter(x,y,z, zcolor=indices, color=cgrad(:rainbow), markersize = 2, 
                markerstrokewidth = 0, cbar = true, w = 0.5, legend = false, 
                title=L"\lambda = %$λ, d = %$dims")
        plot(p1,p2,p3,p4,layout=l)

        savefig("figures/"*"$filename"*".pdf")

    elseif dims==2

        scatter(Coordinates[1,:],Coordinates[2,:],zcolor=indices, 
                color=cgrad(:rainbow), aspect_ratio = 1, markerstrokewidth = 0, legend = false, cbar = true, 
                title=L"\lambda = %$λ, d = %$dims")
        png("figures/"*"$filename"*".png")
        
    elseif dims==1

        scatter(Coordinates[1,:],zcolor=indices, 
                color=cgrad(:rainbow), aspect_ratio = 1, markerstrokewidth = 0, legend = false, cbar = true, 
                title=L"\lambda = %$λ, d = %$dims")
        png("figures/"*"$filename"*".png")
    end

end

function *(A::T,B::T,C::T) where T<:Array{Real}

    (N,M) = size(A)
    if N!=M
        throw("should be a square matrix!")
    end

    (N,M) = size(B)
    if N!=M
        throw("should be a square matrix!")
    end

    (N,M) = size(C)
    if N!=M
        throw("should be a square matrix!")
    end

    temp = zeros(T,(N,N))
    D = zeros(T,(N,N))
    mul!(temp,A,B)
    mul!(D,temp,C)
    return D
end
    


function finite_sampling(r,ip)
# create a synthetic covariance matrix based on the principle of finite_sampling
    (N,T) = size(r)
    P = I(N)[ip,:]
    r = normalization(P*r,ϵ = 1.0e-4)
    r = r .- mean(r,dims=2)
    C = zeros(Float32,N,N)
    mul!(C,r,r') #experimental matrix
    C = C/T
    Cₛ = construct_Toeplitz(C)
    
    G = randn(Float32,(N,N))/sqrt(N)
    Cᵣ = *(G,Cₛ,G') #random matrix based on finite sampling
    
    α = tr(C)/tr(Cᵣ) #scaling factor
    Cᵣ = α*Cᵣ

    U = eigvecs(Cᵣ)
    V = eigvecs(C)

    U = convert(Array{Float32},U)
    V = convert(Array{Float32},V)

    Pr = zeros(Float32,N,N)
    mul!(Pr,V,U')
    return *(Pr,Cᵣ,Pr')

end

function block_sampling(C::AbstractMatrix,K::Int; p = 0.1)
    N = size(C,1)
    S = randsubseq(1:N-K-1,p)
    n = length(S)
    Λ = zeros(K,n)
    for i = 1:length(S)
        C_s = C[S[i]:S[i]+K-1,S[i]:S[i]+K-1]
        λ = eigvals(C_s)
	    sort!(λ, rev=true)
        Λ[:,i] = λ
    end
    return mean(Λ,dims=2)
end


function data_xy(datafile::String; seqlen = 100, xsize = 1, nullinputsize = 1, trials = 20)
    @unpack pre_converging_frames, onset_frames, FiringRate, predict_eye_conv_ROI_id, neural_neural_consistency_metric, ROI = matread(datafile)

    N = size(FiringRate,1)
    inputsize = length(predict_eye_conv_ROI_id)
    
    if trials > length(pre_converging_frames)
        trials = length(pre_converging_frames)
        println("trial number is bounded by the data, and is now set to $trials")
    end
    
    input_idx = Int.(predict_eye_conv_ROI_id)
    nullinput_idx = setdiff(shuffle(1:N), input_idx)
    neuron_idx = cat(input_idx, nullinput_idx, dims = 1)[1:inputsize+nullinputsize]

    ysize = inputsize+nullinputsize

    y = zeros(ysize, seqlen, trials)

    if xsize < ysize 
        xsize = ysize
    end

    x = zeros(xsize, seqlen, trials)
    onset_frames = Int.(onset_frames)
    pre_converging_frames = Int.(pre_converging_frames)
    w = 1
    
    for n = 1:trials
        start_frame = pre_converging_frames[n]
        end_frame = start_frame + seqlen - 1
        frames = start_frame:end_frame
        y[:,:,n] = FiringRate[neuron_idx,frames]
        for k = 1:length(onset_frames) #it is possible that in each sequence, there are a few events
            if onset_frames[k] in frames
                stimulated_frames = onset_frames[k] - start_frame - 5 : onset_frames[k] - start_frame - 3
                x[1:inputsize,stimulated_frames,n] .= w*neural_neural_consistency_metric[:,k] #present external inputs
            end
        end
    end
    
    raw_neuron_idx = ROI[neuron_idx]

    return x, y, raw_neuron_idx
end

function data_xy(firingrate::T; seqlen = 1000, ysize = 1) where T<:Array{Float32,2}

    N = size(firingrate,1)
    if ysize < N
        neuron_idx = shuffle(1:N)[1:ysize]
        r_train = firingrate[neuron_idx,1:seqlen]
    elseif ysize == N
        r_train = firingrate[:,1:seqlen]
        neuron_idx = collect(1:N)
    else
        error("ysize is larger than the number of neurons in the dataset!")
        return nothing
    end

    return r_train, neuron_idx
    
end

function data_xy(firingrate::T, sequence::UnitRange{Int64}; ysize = 1) where T<:Array{Float32,2} 

    N = size(firingrate,1)
    if ysize<N
        neuron_idx = shuffle(1:N)[1:ysize] 
        r_train = firingrate[neuron_idx,sequence]
    elseif ysize == N
        r_train = firingrate[:,sequence]
        neuron_idx = collect(1:N)
    else
        error("ysize is larger than the number of neurons in the dataset!")
        return nothing
    end

    return r_train, neuron_idx
    
end

function predict_CovarianceMatrix(J,Cᵣ; stepsize = 0.1)

    a = collect(0.1:stepsize:1)
    n = length(a)
    cost = zeros(n)

    for i in 1:n
        C = predict_CovarianceMatrix(J,Cᵣ,a[i])
        cost[i] = norm(C-Cᵣ)
    end

    (value,index) = findmin(cost)
    return value, a[index]
end

function predict_CovarianceMatrix(J,Cᵣ,a)
    N = size(Cᵣ,1)
    K = inv(I(N)-a*J)
    C = K*K'
    α = tr(Cᵣ)/tr(C) #scaling factor
    C = α*C
    return C
end

function lowrank_approx(J; rank = 200)
    mat"""
        [$U,$S,$V] = svdsketch($J, 4e-4, 'MaxSubspaceDimension',$rank);
    """
    return U*S*V'
end

function RG_analysis(spike,root)
    mat"""
        matlabroot = getenv('MATLAB_PROJECT');
        addpath([matlabroot, '/code/RG/functions']);
        N = size($spike,1);
        iterations = floor(log2(N));
        h=coarse_grain($spike,iterations,"dims",1,"do_toeplitz_construction",1,'NORM',0);
        saveas(h,[$root,'/eigen.png']);
        close all;
    """
end


function gendir(param,root)
    dir = mkpath(joinpath(root,bytes2hex(sha1(json(param))))) 
    open(joinpath(dir, "param.json"), "w") do io 
    JSON.print(io, param, 4) 
    end 
    return dir 
end 

function binary_connectivity(J;threshold = 0.1)
    N = size(J,1)
    J_t = zeros(eltype(J),N,N)
    J_t .= J
    J_t[abs.(J).<threshold].= 0.0
    J_t[J.>threshold] .= 1.0
    J_t[J.<-threshold] .= -1.0
    return J_t
end

function randomclusters(N)
    mat"""
        matlabroot = getenv('MATLAB_PROJECT');
        addpath([matlabroot, '/code/RG/functions']);
        $neural_set = random_clusters($N);
    """
    return neural_set
end

function similarity_matching(a;trials=10)
    gr(format=:png)
    (N,T) = size(a)
    k_max = convert(Int,floor(log2(N)))
    k_min = 6
    p = plot()
    for n in k_min:k_max
        K = 2^n
        G = zeros(T,T)
        for j in 1:trials 
            id = shuffle(Vector(1:N))[1:K]
            r = a[id,:]
            G = G .+ r'*r/K
        end
        G = G/trials
        λ = eigvals(G)
        sort!(λ, rev=true)
        rank = collect(1:K)
        plot!(p, rank, λ[1:K],markershape=:auto, markersize=1,xaxis = :log, yaxis = :log)
        generate_heatmap(G, "gram_map"*"K=$K", climit = (0,0.01), xlabel = "time", ylabel = "time")
    end
    display(p)
end

function remove_outliers(v)
    mat"""
        $u = rmoutliers($v);
    """
    return u
end

function subsampling(C,root)
    mat"""
        matlabroot = getenv('MATLAB_PROJECT');
        addpath([matlabroot, '/code/RG/functions']);
        [h1,h2] = subsampling($C);
        saveas(h1,[$root,'/eigen.png']);
        saveas(h2,[$root,'/density.png']);
        close all;
    """
end
#add theoretical calculation to the subsampling as well
function subsampling(C::Matrix, λ::Vector, P::Vector, p::ERMParameter, root)
    @unpack L, ρ, μ, ϵ, n, ξ, σ̄⁴ = p
    titles = "L = $(round(L,digits=1)), \\mu = $μ, \\epsilon = $(round(ϵ,digits=3)), \\rho^{-1/d} = $(round(ρ^(-1/n),digits=3)),
                d =  $n, \\xi = $(round(ξ,digits=1)), \\beta = 1/\\xi, E(\\sigma^4) = $(round(σ̄⁴,digits=2))"
    mat"""
        matlabroot = getenv('MATLAB_PROJECT');
        addpath([matlabroot, '/code/RG/functions']);
        [h1,h2] = subsampling($C);
        ax2 = h2.CurrentAxes;
        ax1 = h1.CurrentAxes;
        lgd = ax2.Legend.String;
        lgd{end+1}='theory';
        plot(ax2,$λ,$P,'color',[0.5,0.5,0.5],'linewidth',2);
        legend(ax2,lgd);
        title(ax2,$titles);
        title(ax1,$titles);
        saveas(h1,[$root,'/eigenERM.png']);
        saveas(h2,[$root,'/densityERM.png']);
        close all;
    """
end

function subsampling_matrix_ensemble(λ::Vector, P::Vector, p::ERMParameter; root = joinpath(ENV["JULIA_PROJECT"],"figures"), N = 10)
    
    @unpack L, ρ, μ, ϵ, n, β, σ̄⁴ = p
    titles = "L = $(round(L,digits=1)), \\mu = $μ, \\epsilon = $(round(ϵ,digits=3)), \\rho^{-1/d} = $(round(ρ^(-1/n),digits=3)),
                d =  $n, \\beta = $β, E(\\sigma^4) = $(round(σ̄⁴,digits=2))"
    C_ensemble = generate_ERM_ensemble(p, ensemble_size = N)
    
    mat"""
        matlabroot = getenv('MATLAB_PROJECT');
        addpath([matlabroot, '/code/RG/functions']);
        [h1,h2] = subsampling_matrix_ensemble($C_ensemble);
        ax2 = h2.CurrentAxes;
        ax1 = h1.CurrentAxes;
        lgd = ax2.Legend.String;
        lgd{end+1}='theory';
        plot(ax2,$λ,$P,'color',[0.5,0.5,0.5],'linewidth',2);
        legend(ax2,lgd);
        title(ax2,$titles);
        title(ax1,$titles);
        saveas(h1,[$root,'/eigenERM.png']);
        saveas(h2,[$root,'/densityERM.png']);
        close all;
    """

end

function generate_ERM_ensemble(p::ERMParameter; ensemble_size = 100, different_neural_variability = true)
    
    @unpack N, L, n = p
    C_ensemble = zeros(N,N,ensemble_size)
    
    if different_neural_variability
        σ² = rand(LogNormal(0,0.5),N,1)
    else
        σ² = ones(N,1)
    end
    
    σ = vec(broadcast(√,σ²))
    Δ = diagm(σ)

    points = rand(Uniform(-L/2,L/2),N,n) #points are uniformly distributed in a region 
    for i in 1:ensemble_size
        C = reconstruct_covariance(points, p, subsample = false)
        C_ensemble[:,:,i] = Δ*C*Δ
    end
    
    return C_ensemble

end


ranking(v) = collect(1:length(v))/length(v)

function compare_v(v1::Vector,v2::Vector)
    V = hcat(v2, -v2, reverse(v2), -reverse(v2))
    D = sum(abs2, V.-v1, dims = 1)
    (d,index) = findmin(vec(D))
    return V[:,index]
end

function compare_v(V1::AbstractMatrix, V2::AbstractMatrix)
    (rows,cols) = size(V1)
    U = copy(V2)
    for i in 1:cols
        U[:,i] = compare_v(vec(V1[:,i]), vec(V2[:,i]))
    end
    return U
end

function find_eigenvectors(C)
    N = size(C,1)
    F = eigen(C)
    λ, U = F
    U = U[:,sortperm(λ,rev=true)]
    U = U*√N
    sort!(λ,rev=true)
    return λ, U
end

# single thread
function random_sampling_covariance(Covariance::Matrix, U::Matrix; K = 1024, dims = 4)

    N = size(Covariance,1)
    samples = Int(round(N/K))
    W = zeros(K,dims)
    V_template = copy(W)

    for trial in 1:samples
        id = shuffle(1:N)[1:K]
        C = Covariance[id,id]
        Cₜ = construct_Toeplitz(C)
        ip_K, E, t = min_fnorm(C, Cₜ, collect(1:K), t=5.0, MAX_STEPS=10000)
        Pm = I(K)[ip_K,:]
        C = Pm*C*Pm'
        V = find_eigenvectors(C)
        if trial == 1
            V_template .= U[sort!(id),:]
        else
            V_template .= W/trial
        end
        W = W .+ compare_v(V_template,V[:,1:dims])
    end
    return W/samples
end

#multi-thread
function random_sampling_covariance(Covariance::Matrix, U::Matrix; K = 1024, dims = 4, samples = 10)

    N = size(Covariance,1)
    W = zeros(K,dims,samples)
    V_template = zeros(K,dims)

    Threads.@threads for trial in 1:samples
        id = shuffle(1:N)[1:K]
        C = Covariance[id,id]
        Cₜ = construct_Toeplitz(C)
        ip_K, E, t = min_fnorm(C, Cₜ, collect(1:K), t=5.0, MAX_STEPS=10000)
        Pm = I(K)[ip_K,:]
        C = Pm*C*Pm'
        V = find_eigenvectors(C)
        if trial == 1
            V_template .= U[sort!(id),:]
            W[:,:,1] .= compare_v(V_template,V[:,1:dims])
        else
            W[:,:,trial] .= V[:,1:dims]
        end
    end

    for trial in 2:samples
        V_template = dropdims(sum(W[:,:,1:trial-1],dims=3)/(trial-1), dims=3)
        W[:,:,trial] = compare_v(V_template,W[:,:,trial])
    end
    
    return dropdims(sum(W,dims=3)/samples, dims=3)

end


function random_sampling_similarity(a::Matrix, template::Matrix; K = 1024,  dims = 4, samples = 1)
    #a activity matrix
    #template: dims × T orthogonal matrix
    
    (N, T) = size(a)
    W = zeros(T,dims,samples)

    Threads.@threads for trial in 1:samples
        id = shuffle(1:N)[1:K]
        a_sampled = a[id,:]
        F = svd(a_sampled)
        Vt = F.Vt
        V = Vt'
        W[:,:,trial] .= compare_v(template,V[:,1:dims])
    end
    
    return dropdims(sum(W,dims=3)/samples, dims=3)
end


function changing_basis(C::Matrix; basis="Cauchy")
    N = size(C,1)
    if basis == "Cauchy"
        H = rand(Cauchy(0,1), N, N)
    elseif basis == "Gaussian"
        H = rand(Gaussian(0,1), N, N)
    end
    F = qr(H)
    Q, R = F

    G = eigen(C)
    λ, U = G
    return Q*diagm(0=>vec(λ))*Q'
end

#uppertriangular matrix excluding the diagonal
function uppertriangular(C::Matrix)
    N = size(C,1)
    m = triu(trues(N,N),1)
    return C[m]
end
#construct uppertriangular matrix from a vector
function construct_uppertriangular(C::Matrix, v::Vector)
    n = size(C,1)
    k=0
    return diagm(diag(C)) .+ [ j<i ? (k+=1; v[k]) : 0 for i=1:n, j=1:n ]'
end

function ball_point_picking(N; R = 1, dims = 3)
    points = zeros(N,dims)
    for j in 1:N
        while true
            test_point = rand(Uniform(-R,R),1,dims) 
            if norm(test_point) <= R
                points[j,:] = test_point
                break
            end
        end
    end
    return points
end

function sphere_point_picking(N; R = 1, dims = 3)
    points = zeros(N, dims)
    for j in 1:N 
        while true
            x₁ = rand(Uniform(-R,R))
            x₂ = rand(Uniform(-R,R))
            if √(x₁^2 + x₂^2) <= R
                x = 2*x₁*√(1-x₁^2 - x₂^2)
                y = 2*x₂*√(1-x₁^2 - x₂^2)
                z = 1 - 2*(x₁^2 + x₂^2)
                points[j,:] = [x y z]
                break
            end
        end
    end
    return points
end

#points picking with short-range repulsion 
function point_picking(N, d::Distribution; R = 1, ϵ = 0.01, dims = 2)
    points = zeros(N,dims)
    points[1,:] = rand(d,1,dims)
    for j in 2:N
        while true
            test_point = rand(d,1,dims)
            pairwise_distance = map(norm, eachrow(points[1:j-1,:] .- test_point))
            if isempty(findall(d->d<ϵ,pairwise_distance))
                points[j,:] = test_point
                break
            end
        end
    end
    return points
end




function reconstruct_covariance(points, p::ERMParameter; subsample = false)
    
    @unpack μ, ϵ, β, L = p 

    N = size(points, 1)
    D = pairwise(Euclidean(), points, dims=1)
    
    C = copy(D)
    #C[D .<= ϵ] .= 1
    #C[D .> ϵ] .= ϵ^μ*(D[D .> ϵ]).^(-μ).*exp.(-(D[D .> ϵ] .- ϵ)*β)
    f(d) = ϵ^μ*(ϵ^2 .+ d^2).^(-μ/2)
    C .= ϵ^μ*(ϵ^2 .+ D.^2).^(-μ/2)
    #C[D .< L] .= f.(D[D .< L])
    #C[D .> L] .= f.(L * exp.(D[D .> L]./L .- 1))
    #D = D .+ α*ones(N,N)
    #D = D .+ diagm(ones(N))
    #C = D.^β
    #C = D.^β.*exp.(-D/ξ)
    #C = exp.(-D.^2/ξ)
    #γ = N/tr(C)
    #C = C*γ
    if subsample
        subsampling(C, "./figures")
    end

    return C
end

function reconstruct_covariance(points, p::ERMParameter; subsample = false)
    
    @unpack μ, ϵ, β, L = p 

    N = size(points, 1)
    D = pairwise(Euclidean(), points, dims=1)
    
    C = copy(D)
    #C[D .<= ϵ] .= 1
    #C[D .> ϵ] .= ϵ^μ*(D[D .> ϵ]).^(-μ).*exp.(-(D[D .> ϵ] .- ϵ)*β)
    f(d) = ϵ^μ*(ϵ^2 .+ d^2).^(-μ/2)
    C .= ϵ^μ*(ϵ^2 .+ D.^2).^(-μ/2)
    #C[D .< L] .= f.(D[D .< L])
    #C[D .> L] .= f.(L * exp.(D[D .> L]./L .- 1))
    #D = D .+ α*ones(N,N)
    #D = D .+ diagm(ones(N))
    #C = D.^β
    #C = D.^β.*exp.(-D/ξ)
    #C = exp.(-D.^2/ξ)
    #γ = N/tr(C)
    #C = C*γ
    if subsample
        subsampling(C, "./figures")
    end

    return C
end

function get_rho(points, p::ERMParameter)
    
    @unpack μ, ϵ, β = p 

    N = size(points, 1)
    D = pairwise(Euclidean(), points, dims=1)

    rho = ϵ^μ*(ϵ^2 .+ D.^2).^(-μ/2)

    return rho 
end

function ERM_eig_trial(p::ERMParameter;ntrial=3, points0 = nothing)
    @unpack L,N,n = p
    d = n
    eig_trial = zeros((N,ntrial))
    Threads.@threads for t = 1:ntrial
        if points0 == nothing
            points = rand(Uniform(-L/2,L/2),N,d)
        else
            points = points0
        end
        C = reconstruct_covariance(points, p)
        λ = sort(eigvals(Hermitian(C)), rev=true)
        eig_trial[:,t] = λ
    end
    return eig_trial
end


function eigendensity(C::Matrix, p::ERMParameter; length = 50, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10^10)

    λ = eigvals(C)
    id, λ = dropzeros_and_less(λ)
    N = size(C,1)

    λᵣ = range(minimum(λ),maximum(λ),length=length)

    h_λ = fit(Histogram, λ, LogRange(minimum(λ),maximum(λ),length))
    h_λ = normalize(h_λ, mode=:pdf)

    λᵣ, p_sim = dropzero(h_λ)
    p_sim = p_sim.*size(id,1)/N #correction for deleting negative eigenvalues 

    @unpack σ̄² = p

    if use_variation_model
        σ² = diag(C)
        λ_theory, p_theory = eigendensity_variation_model(λᵣ,p_sim,σ²,p,λ_min = λ_min,λ_max = λ_max)
        id, p_theory = dropzeros_and_less(p_theory.-10^-6)
        λ_theory = λ_theory[id]
        p_theory = p_theory .+ 10^-6
    else
        if !correction
            λ_theory, p_theory = eigendensity(λᵣ/σ̄²,p, λ_min = λ_min)
            p_theory = p_theory/σ̄²
        else
            λ_theory, p_theory = eigendensity_with_correction(λᵣ,p, λ_min = λ_min)
        end
    end

    return λᵣ, p_sim, λ_theory, p_theory

end

function eigendensity(C::Matrix; length = 50, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10^10)

    λ = eigvals(C)
    id, λ = dropzeros_and_less(λ)
    N = size(C,1)

    #λᵣ = range(minimum(λ),maximum(λ),length=length)

    h_λ = fit(Histogram, λ, LogRange(minimum(λ),maximum(λ),length))
    h_λ = normalize(h_λ, mode=:pdf)

    λᵣ, p_sim = dropzero(h_λ)
    p_sim = p_sim.*size(id,1)/N #correction for deleting negative eigenvalues 

    return λᵣ, p_sim

end

function eigendensity_range(C::Matrix; length = 10, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10)

    λ = eigvals(C)
    id, λ = dropzeros_and_less(λ)
    N = size(C,1)

    #λᵣ = range(minimum(λ),maximum(λ),length=length)
    N2 = size(findall(λ_min.< λ .<λ_max))[1]
    h_λ = fit(Histogram, λ, LogRange(λ_min, λ_max,length))
    h_λ = normalize(h_λ, mode=:pdf)

    λᵣ, p_sim = h_λ.edges , h_λ.weights
    p_sim = p_sim.*N2/N #correction for deleting negative eigenvalues 

    binsize = λᵣ[1][2] - λᵣ[1][1]
    λᵣ = collect(λᵣ[1][2:end].-binsize/2)
    return λᵣ, p_sim

end

function eigendensity(p::ERMParameter; length = 100, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10^10, m = 100, different_neural_variability = false, theory = true, σ² = Nothing)

    @unpack N, L, n = p
    λ_all = zeros(m,p.N)
    for i = 1:m
        points = rand(Uniform(-L/2,L/2),N,n)
        C = reconstruct_covariance(points, p, subsample = false)
        if different_neural_variability
            if σ² == Nothing
                σ² = vec(rand(LogNormal(0,0.5),N,1))
            end
        else
            σ² = ones(N,1)
        end
        σ = vec(broadcast(√,σ²))
        Δ = diagm(σ)
        Ĉ = Δ*C*Δ
        λ = eigvals(Ĉ)
        λ_all[i,:] = λ
    end

    λ = reshape(λ_all,p.N*m)
    λ = sort(λ, rev=true)
    id, λ = dropzeros_and_less(λ)

    #λᵣ = range(minimum(λ),maximum(λ),length=length)

    h_λ = fit(Histogram, λ, LogRange(minimum(λ),maximum(λ),length))
    h_λ = normalize(h_λ, mode=:pdf)

    λᵣ, p_sim = dropzero(h_λ)
    p_sim = p_sim.*size(id,1)/m/N #correction for deleting negative eigenvalues 

    @unpack σ̄² = p

    if theory
        if use_variation_model
            σ² = diag(C)
            λ_theory, p_theory = eigendensity_variation_model(λᵣ,p_sim,σ²,p,λ_min = λ_min,λ_max = λ_max)
            id, p_theory = dropzeros_and_less(p_theory.-10^-6)
            λ_theory = λ_theory[id]
            p_theory = p_theory .+ 10^-6
        else
            if !correction
                λ_theory, p_theory = eigendensity(λᵣ/σ̄²,p, λ_min = λ_min)
                p_theory = p_theory/σ̄²
            else
                λ_theory, p_theory = eigendensity_with_correction(λᵣ,p, λ_min = λ_min)
            end
        end

        return λᵣ, p_sim, λ_theory, p_theory, [λ[Int(N*m/10)], λ[Int(N*m/100)]]
    end

    if !theory
        return λᵣ, p_sim, [λ[Int(N*m/10)], λ[Int(N*m/100)]]
    end

end

#########  compute cov pdf by corr pdf ####################

function eigendensity_convolution(p::ERMParameter; length = 100, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10^10, m = 100, different_neural_variability = false, theory = true, σ² = Nothing)

    @unpack N, L, n = p
    λ_all = zeros(m,p.N)
    λ_all2 = zeros(m,p.N)
    for i = 1:m
        points = rand(Uniform(-L/2,L/2),N,n)
        C = reconstruct_covariance(points, p, subsample = false)
        if different_neural_variability
            if σ² == Nothing
                σ² = vec(rand(LogNormal(0,0.5),N,1))
            end
        else
            σ² = ones(N,1)
        end
        σ = vec(broadcast(√,σ²))
        Δ = diagm(σ)
        Ĉ = Δ*C*Δ
        λ = eigvals(Ĉ)
        λ2 = eigvals(C) + vec(rand(LogNormal(0,0.5),N,1).^2) .-1
        λ_all[i,:] = λ
        λ_all2[i,:] = λ2
    end

    λ = reshape(λ_all,p.N*m)
    λ = sort(λ, rev=true)
    id, λ = dropzeros_and_less(λ)

    #λᵣ = range(minimum(λ),maximum(λ),length=length)

    h_λ = fit(Histogram, λ, LogRange(minimum(λ),maximum(λ),length))
    h_λ = normalize(h_λ, mode=:pdf)

    λᵣ, p_sim = dropzero(h_λ)
    p_sim = p_sim.*size(id,1)/m/N #correction for deleting negative eigenvalues 
    #############################################################
    λ2 = reshape(λ_all2,p.N*m)
    λ2 = sort(λ2, rev=true)
    id, λ2 = dropzeros_and_less(λ2)

    #λᵣ = range(minimum(λ),maximum(λ),length=length)

    h_λ2 = fit(Histogram, λ2, LogRange(minimum(λ),maximum(λ),length))
    h_λ2 = normalize(h_λ2, mode=:pdf)

    λᵣ2, p_sim2 = dropzero(h_λ2)
    p_sim2 = p_sim2.*size(id,1)/m/N #correction for deleting negative eigenvalues 
    
    return λᵣ, p_sim, λᵣ2, p_sim2

end


#########  high density theory  ####################

function eigendensity(Λ::Vector, p::ERMParameter; λ_min = 1)

    @unpack ρ, μ, ϵ, n, β = p

    S = 2π^(n/2)/gamma(n/2)
    large_eigs = findall(x->x>=λ_min, Λ)
    𝜦 = Λ[large_eigs]
    k0 = guess_zeros(𝜦/ρ,p)
    Prob = similar(𝜦)

    for j in 1:length(𝜦)

        λ = 𝜦[j]
        k = f̂_inv(λ/ρ,p,k0[j])
        Prob[j] = S/((2π)^n) * k^(n-1)/(ρ^2*abs(∂f̂(k,p)))

    end

    return 𝜦, Prob

end

####### high density theory with second order correction #############

function eigendensity_with_correction(Λ::Vector, p::ERMParameter; λ_min = 1)


    @unpack ρ, μ, ϵ, n, β, L, σ̄², σ̄⁴ = p

    S = 2π^(n/2)/gamma(n/2)

    #determine upper and lower cutoff wave vector
    #upper cutoff wave vector is proportional to ρ^(1/n): 
    #where a is the lattice size ρ^(-1/n)
    #lower cutoff is given by the box diameter L 
    #kx ∈ [-π/a, -π/L]&&[π/L, π/a], ky ∈ [-π/a, -π/L]&&[π/L, π/a]
    # |k| = √(kx² + ky²)
    
    #k₂ = π*ρ^(1/n)
    k₂ = π/ϵ
    k₁ = π/L

    𝝆 = ρ*σ̄² #simplify the notations below 

    
    large_eigs = findall(x -> x>=λ_min, Λ)
    𝜦 = Λ[large_eigs]
    k0 = guess_zeros(𝜦/𝝆,p)
    Prob = similar(𝜦)

    for j in 1:length(𝜦)

        λ = 𝜦[j]
        k = f̂_inv(λ/𝝆,p,k0[j])
        P₀ = S/((2π)^n) * k^(n-1)/(ρ * 𝝆 * abs(∂f̂(k,p)))
        P₁ = 1/2 * (S/((2π)^n))^2 * k^(n-1)/(𝝆^2 * abs(∂f̂(k,p))) * σ̄⁴/𝝆 * 
        (-g(k₂,p)*λ/(λ - 𝝆*f̂(k₂,p))+
        g(k₁,p)*λ/(λ - 𝝆*f̂(k₁,p))+
        quadgk_cauchy(x->Q1(x,p,λ),f̂(k₂,p), λ/𝝆, f̂(k₁,p)))
        P₂ = 1/2 * (S/((2π)^n))^2 * σ̄⁴/(𝝆^2*abs(∂f̂(k,p)))*
                ∂g(k,p) * quadgk_cauchy(x->Q2(x,p),f̂(k₂,p), λ/𝝆, f̂(k₁,p))
        Prob[j] = P₀ - P₁ - P₂
    end

    return 𝜦, Prob 

end

function f_r_t_family(r, p::ERMParameter)
    @unpack μ, ϵ, β = p
    c = (1/ϵ^μ)^(-2/μ)
    y = ϵ^μ.*(c .+ r.^2).^(-μ/2)
    return y
end

f_r = f_r_t_family

#######  Fourier transform of distance function and its various corrections ##############
include("t_pdf_FT.jl")

#f̂(x, p) = f̂₀(x, p) + f̂_correction(x,p)
#∂f̂₀(x,p) = ForwardDiff.derivative(x -> f̂₀(x,p),x)
#∂f̂(x,p) = ∂f̂₀(x,p) + ∂f̂_correction(x,p)

#f̂(x, p) = f̂₀_num(x,p)

f̂(x, p) = f̂₀_th(x, p)
∂f̂(x,p) = ForwardDiff.derivative(x -> f̂(x,p),x)

#∂²f̂(x,p) = ForwardDiff.derivative(x -> ∂f̂(x,p),x)
#f̂(x, p) = min(f̂₀(x, p) + f̂_correction(x,p), f̂_numerical(10^-8, p))
#f̂(x, p) = f̂_numerical(x, p)


#∂²f̂(x,p) = ∂²f̂₀(x,p) + ∂²f̂_correction(x,p)

#∂²f̂₀(x,p) = ForwardDiff.derivative(x -> ∂f̂₀(x,p),x)

function f̂₀(x, p)

    @unpack μ, β, ϵ, n = p

    if (β!=0) && (μ!=0)
        α₁ = μ*(1 - μ*log(2))/(1 - μ)
        α₂ = (1-2*μ)*(1+μ*log(2))*β/(1-μ)
        k = β/x * (β^2/x^2 + 1)^(-1/2)
        #return exp(β*ϵ)*2π*ϵ^μ/x^(2-μ)*(α₂/x + α₁)*gamma(2 - μ)
        return exp(β*ϵ)*2π*ϵ^μ/x^(2-μ)*gamma(2 - μ)*pFq([-(1-μ),(1-μ)+1], [1],(1-k)/2)
        #special case only for n=2, othercase has not been computed 
    elseif β==0
        c = π^(n/2)*2^μ*gamma(μ/2)/gamma((n-μ)/2)
        return exp(β*ϵ)*(2π)^n*ϵ^μ/c/x^(n-μ)
        #https://en.wikipedia.org/wiki/Fourier_transform, power law function
    elseif μ==0
        c = gamma((n+1)/2)/π^((n+1)/2)
        return exp(β*ϵ)*c*β*exp(β*ϵ)*(2π)^n/((β^2 + x^2)^((n+1)/2))
        #https://en.wikipedia.org/wiki/Fourier_transform, exponential function
    end
end

function f̂_numerical(k, p)
    @unpack μ, β, ϵ, n, L = p
    return 𝔅(k,p) + quadgk(x -> 𝔉(x,(k,p)), k*ϵ, k*L)[1]
end

function f̂_correction(k,p)
    @unpack μ, ϵ, β = p
    #return -μ*π*ϵ^2/(2-μ)
     if μ!=0
        return 𝔅(k,p) - quadgk(x -> 𝔉(x,(k,p)), 0, k*ϵ)[1]
        #return 0
     else
        return 0
     end
end

function 𝔅(k,p)
    @unpack μ, ϵ, n = p
    return (2π*ϵ/k)^(n/2)*besselj(n/2,k*ϵ)
end

function 𝔉(x,param)
    k, p = param
    @unpack β, μ, ϵ, n = p
    #return (2π)^(n/2)*ϵ^μ/k^(n-μ)*exp(-β*x/k)*x^(n/2-μ)*besselj(n/2-1,x) #ϵ ≪ 1/β
    return exp(β*ϵ)*(2π)^(n/2)*ϵ^μ/k^(n-μ)*x^(n/2-μ)*besselj(n/2-1,x)
end

function 𝔣(k,p) 
    # 𝔣(k) = 𝔉(kϵ)
    @unpack μ, ϵ, n, β = p
    return exp(β*ϵ)*(2π)^(n/2)*ϵ^μ/k^(n/2)*ϵ^(n/2-μ)*besselj(n/2-1,k*ϵ)
end

function ∂f̂_correction(k,p)
    @unpack μ, ϵ, n = p
    if (f̂_correction(k,p) == -μ*π*ϵ^2/(2-μ)) || μ == 0 
        return 0
    elseif μ!=0
        ∂𝔅∂k = ForwardDiff.derivative(k -> 𝔅(k,p),k)
        #∂𝔅∂k = (-n/2)*(2π*ϵ/k)^(n/2)/k*besselj(n/2,k*ϵ) + 
        #    (2π*ϵ/k)^(n/2)*ϵ/2*(besselj(n/2-1,k*ϵ) - besselj(n/2+1,k*ϵ))
        ∂∫𝔉∂k = 𝔣(k,p)*ϵ - (n-μ)/k * quadgk(x -> 𝔉(x,(k,p)), 0, k*ϵ)[1]
        return  ∂𝔅∂k - ∂∫𝔉∂k
    end
end


function ∂²f̂_correction(k,p)
    @unpack μ, ϵ, n = p
    if ∂f̂_correction(k,p) == 0 || μ==0
        return 0
    elseif μ!=0
        ∂²𝔅∂k² = ForwardDiff.derivative(k->ForwardDiff.derivative(k ->𝔅(k,p),k),k)
        ∂∫𝔉∂k = 𝔣(k,p)*ϵ - (n-μ)/k * quadgk(x -> 𝔉(x,(k,p)), 0, k*ϵ)[1] 
        ∂²∫𝔉∂k² = ForwardDiff.derivative(k->𝔣(k,p),k)*ϵ +
                    (n-μ)/k^2 * quadgk(x -> 𝔉(x,(k,p)), 0, k*ϵ)[1] -
                    (n-μ)/k * ∂∫𝔉∂k
        return ∂²𝔅∂k² - ∂²∫𝔉∂k²
    end
end


function fzero(x, param)
    p, y = param
    return f̂(x, p) - y
end

function guess_zeros(y, p)

    @unpack μ, β, ϵ, n, ρ = p

    if β!=0
        a = 2μ*π*ϵ^μ*(1 - μ*log(2))/(1 - μ)*gamma(2 - μ)
    else
        c = π^(n/2)*2^μ*gamma(μ/2)/gamma((n-μ)/2)
        a = (2π)^n*ϵ^μ/c
    end

    b = -μ*π*ϵ^2/(2-μ)

    #guess the zeros
    if μ!=0
        x0 = ((y .- b)/a).^(-(1/(n - μ)))
    else
        x0 = y.^(-1/(n+1))
    end
    #=
    S = 2π^(n/2)/gamma(n/2)
    df = -(2 - μ)*a*k0.^(-(3 - μ))
    return S/((2π)^n) * k0.^(n-1)./(ρ^2*abs.(df))
    =#
    return x0

end

function f̂_inv(y,p,x0)
    #x0 is the guessed value
    # p is the parameters in the function
    # compute the inverse function of f̂ via root finding
    fx = ZeroProblem(fzero,x0)
    return solve(fx,(p,y))
end

function g(x,p)
    @unpack n = p
    return x^(n-1)*f̂(x,p)/∂f̂(x,p)
end

#∂g(x,p) = ForwardDiff.derivative(x -> g(x,p),x)
function ∂g(x,p)
    @unpack n = p
    return (n-1)*x^(n-2)*f̂(x,p)/∂f̂(x,p) + x^(n-1) - x^(n-1)*f̂(x,p)*∂²f̂(x,p)/(∂f̂(x,p))^2
end


function Q1(x,p,λ)
    #reformulate the integral by letting f̂(k, p) -> x, and express k as f̂⁻¹(k, p) or x
    #thus the principal_value problem can be solved using quadgk_cauchy,
    #see the function quadgk_cauchy
    @unpack ρ, σ̄² = p
    k0 = guess_zeros(x,p)
    k = f̂_inv(x,p,k0)
    return λ/(ρ*σ̄²)*∂g(k,p)/∂f̂(k,p)
end


function Q2(x,p)
    #reformulate the integral by letting f̂(k, p) -> x, and express k as f̂⁻¹(k, p) or x
    #thus the principal_value problem can be solved using quadgk_cauchy,
    #see the function quadgk_cauchy
    @unpack n, ρ, σ̄² = p
    k0 = guess_zeros(x,p)
    k = f̂_inv(x,p,k0)
    return k^(n-1)*x/∂f̂(k,p)/(ρ*σ̄²)
end

function quadgk_cauchy(fun, a, c, b)
    #using the transformation discussed at
    #https://github.com/JuliaMath/QuadGK.jl/pull/44#issuecomment-590606772
    fc = fun(c)
    g(x) = (fun(x) - fc) / (x - c)
    return quadgk(g, a, c, b, rtol=1e-9)[1] + fc * log(abs((b - c)/(a - c)))
end

function test_ρ_dependence(λ,p::ERMParameter; ρ_min = 10, ρ_max = 1000, N = 10)

    @unpack n, L, σ̄², σ̄⁴ = p
    density = LogRange(ρ_min,ρ_max,N)
    P0 = similar(density)
    P1 = similar(density)
    P2 = similar(density)
    S = 2π^(n/2)/gamma(n/2)
    k₁ = π/L
    k₂ = π/ϵ #a choice I have not completely understood

    for j in 1:length(density)
        ρ = density[j]
        p.ρ = ρ
        #k₂ = π*ρ^(1/n)
        𝝆 = ρ*σ̄²
        k0 = guess_zeros(λ/𝝆,p)
        k = f̂_inv(λ/𝝆,p,k0)
        #=
        s = @sprintf "f̂(k₂,p) is %.10f"  f̂(k₂,p)
        println(s)

        s = @sprintf "∂f̂(k₂,p) is %.10f"  ∂f̂(k₂,p) 
        println(s)
        =#
        P₀ = S/((2π)^n) * k^(n-1)/(ρ * 𝝆 * abs(∂f̂(k,p)))
        P0[j] = P₀
        P1[j] = 1/2 * (S/((2π)^n))^2 * k^(n-1)/(𝝆^2 * abs(∂f̂(k,p))) * σ̄⁴/𝝆 *
        (-g(k₂,p)*λ/(λ - 𝝆*f̂(k₂,p))+
        g(k₁,p)*λ/(λ - 𝝆*f̂(k₁,p))+
        quadgk_cauchy(x->Q1(x,p,λ),f̂(k₂,p), λ/𝝆, f̂(k₁,p)))
        P2[j] = 1/2 * (S/((2π)^n))^2 * σ̄⁴/(𝝆^2*abs(∂f̂(k,p)))*
                ∂g(k,p) * quadgk_cauchy(x->Q2(x,p),f̂(k₂,p), λ/𝝆, f̂(k₁,p))
    end

    plot(density,P0, label="P0")
    plot!(density, -P1, label="-P1")
    plot!(density, -P2, label="-P2")
    plot!(density, P0-P1-P2, label="P0-P1-P2")
    plot!(xlabel = L"\rho", ylabel = L"\textrm{Pdf }(\lambda = %$λ)", legend=:bottomright)
    savefig("figures/rho_dependence.png")
end

###############  Gaussian Variational Theory ######################### 

function eigendensity_variation_model(Λ::Vector, P_sim::Vector, ς²::Vector, p::ERMParameter; λ_min = 1, λ_max = 10^10)

    @unpack ρ = p

    large_eigs = findall(x->x>=λ_min, Λ)
    𝜦 = Λ[large_eigs]
    Prob_sim = P_sim[large_eigs]
    small_eigs = findall(x->x<=λ_max, 𝜦)
    𝜦 = 𝜦[small_eigs]
    Prob_sim = Prob_sim[small_eigs] #simulated eigendensity, used to guess initial values
    Prob = similar(𝜦) #predicted eigendensity from the gaussian_variation_model
    C0 = 2 #guessed value for C 

    for j in 1:length(𝜦)
        λ = 𝜦[j]
        C = find_zero(C->𝔊(C,ς²,λ,p),C0)[1]
        if !isempty(C)
            a0 = C
            b0 = Prob_sim[j]*ρ*π
            Prob[j] = gaussian_variation_model_fast(λ,ς²,p,a0,b0,plotroot=true)
        else
            Prob[j] = NaN
        end
        s = @sprintf "λ = %.2f; probability density is %.5f" 𝜦[j] Prob[j]
        println(s)
    end

    return 𝜦, Prob

end

function test_ρ_dependence_variation_model(λ,P_sim,p::ERMParameter; ρ_min = 10, ρ_max = 1000, μ = 0, σ = 1, N = 10)
    # λ eigenvalue, whose density will be evaluated
    # P_sim, simulated density

    density = LogRange(ρ_min,ρ_max,N)
    ς² = vec(rand(LogNormal(μ,σ),1000,1))
    P = similar(density)
    C0 = 2 #guessed value for C

    for j in 1:length(density)
        p.ρ = density[j]
        C = find_zero(C->𝔊(C,ς²,λ,p),C0)

        if C > 0
            a0 = C
            b0 = P_sim*density[j]*π
            P[j] = gaussian_variation_model_fast(λ,ς²,p,a0,b0,plotroot=true)
        else
            P[j] = NaN
        end 

        s = @sprintf "ρ = %.3f; probability density is %.5f" density[j] P[j]
        println(s)
        
        #=
        C = find_zeros(C->𝔊(C,ς²,λ,p),a,b)[1]
        G_inv = G(C,p)
        s = @sprintf "G_inv is %.10f"  G_inv
        println(s)
        =#
    end
    plot(density,P)
    plot!(xlabel = L"\rho", ylabel = L"\textrm{Pdf }(\lambda = %$λ)", legend=:bottomright)
    savefig("figures/rho_dependence_GV.png")

    return density, P

end

function gaussian_variation_model(λ, ς²::Vector, p::ERMParameter, a0, b0; plotroot=false)
    #input eigenvalue λ
    #output probability density at λ
    #ς² is the variance distribution of neural activity
    #a0 is the guessed real part of C, and b0 is the guessed imaginary part of C
    
    @unpack ρ = p
    
    a_min = a0/10
    a_max = a0*10
    ax1 = Axis(range(a_min, a_max, length = 10),"a")
        
    b_max = b0*10
    b_min = b0/10
    ax2 = Axis(LogRange(b_min,b_max,10),"b")
    #a and b are real and imaginary part of C when taking the resolvant to the complex plane
    #identify the range of a and b for grid search

    mdbm1 = MDBM_Problem((a,b)->fₐ(a,b,λ,ς²,p),[ax1,ax2])
    mdbm2 = MDBM_Problem((a,b)->fᵦ(a,b,λ,ς²,p),[ax1,ax2])
    MDBM.solve!(mdbm1,5)
    MDBM.solve!(mdbm2,5)
    a1, b1 = getinterpolatedsolution(mdbm1)
    a2, b2 = getinterpolatedsolution(mdbm2)
    if plotroot
        scatter(a1,b1)
        scatter!(a2,b2)
        savefig("figures/test_root_finding")
    end
    #find solutions: fₐ(a,b,λ,ς²,p) = 0
    #find solutions: fᵦ(a,b,λ,ς²,p) = 0
    if isempty(a1) || isempty(a2)
        return NaN #did not find solutions in the specified range
    else
        intersection_pts = find_intersection_between_curves((a1,b1),(a2,b2))

        if !isempty(intersection_pts)
            a_sol, b_sol = intersection_pts[1]
            return find_density(a_sol,b_sol,λ,ς²,p)
        else
            return NaN #did not find a solution
        end
    end

end

function find_intersection_between_curves(curve1, curve2)
    
    (X1,Y1)=curve1
    (X2,Y2)=curve2

    N1 = length(X1)
    N2 = length(X2)

    D² = (repeat(X1,1,N2)' .- X2).^2 + (repeat(Y1, 1, N2)' .- Y2).^2
    #compute pairwise distance between all points from the two curves
    
    indices = findall(d² -> d² < 1e-1, D²)
    #find all potential intersection regions

    intersections = []

    for n in 1:length(indices)
        i,j = Tuple(indices[n])
        if i < N2 && j < N1
            line1 = ((X1[j],Y1[j]),(X1[j+1],Y1[j+1])) #(start point, end point)
            line2 = ((X2[i],Y2[i]),(X2[i+1],Y2[i+1])) #(start point, end point)
            x,y = find_intersection_between_lines(line1,line2)
            if !isempty(x)
                if (X1[j] <= x <= X1[j+1]) || (X1[j] >= x >= X1[j+1])
                    push!(intersections,(x,y))
                end
            end
        end
    end
    
    return intersections

end

function find_intersection_between_lines(line1,line2)
    pt_start1, pt_end1 = line1
    pt_start2, pt_end2 = line2
    k1 = (pt_end1[2] - pt_start1[2])/(pt_end1[1] - pt_start1[1])
    k2 = (pt_end2[2] - pt_start2[2])/(pt_end2[1] - pt_start2[1])
    if k1==k2
        return []
    else
        x_sol = (k1*pt_start1[1] - k2*pt_start2[1] + pt_start2[2] - pt_start1[2])/(k1 - k2)
        y_sol = k1*(x_sol - pt_start1[1]) + pt_start1[2]
        return x_sol, y_sol
    end
end

function gᵣ(a,b,p::ERMParameter)
    @unpack n, L, ρ, ϵ = p
    S = 2π^(n/2)/gamma(n/2)
    #k₂ = π*ρ^(1/n)
    k₂ = π/ϵ #*100
    k₁ = π/L
    #k₁ = 10^-12
    return S/(2π)^n*quadgk(k->k^(n-1)*f̂(k,p)*(1-a*f̂(k,p))/((1-a*f̂(k,p))^2 + b^2*f̂(k,p)^2),k₁,k₂)[1]
end

function gᵢ(a,b,p::ERMParameter)
    @unpack n, L, ρ, ϵ = p
    S = 2π^(n/2)/gamma(n/2)
    #k₂ = π*ρ^(1/n)
    k₂ = π/ϵ #*100
    k₁ = π/L
    #k₁ = 10^-12
    return S/(2π)^n*quadgk(k->k^(n-1)*b*f̂(k,p)^2/((1-a*f̂(k,p))^2 + b^2*f̂(k,p)^2),k₁,k₂)[1]
end

function fₐ(a,b,λ,ς²::Vector,p::ERMParameter)
    @unpack ρ = p
    #ReC = ρ*ς² .* (λ .- ς²*gᵣ(a,b,p)) ./ ((λ .- ς²*gᵣ(a,b,p)).^2 .+ gᵢ(a,b,p)^2)
    ReC = ρ*ς² .* (λ .- ς²*gᵣ(a,b,p)) ./ ((λ .- ς²*gᵣ(a,b,p)).^2 .+ (ς²*gᵢ(a,b,p)).^2)
    return mean(ReC) - a 
end

function fᵦ(a,b,λ,ς²::Vector,p::ERMParameter)
    @unpack ρ = p
    #ImC = ρ*ς²*gᵢ(a,b,p) ./ ((λ .- ς²*gᵣ(a,b,p)).^2 .+ gᵢ(a,b,p)^2)
    ImC = ρ*ς².*ς²*gᵢ(a,b,p) ./ ((λ .- ς²*gᵣ(a,b,p)).^2 .+ (ς² *gᵢ(a,b,p)).^2)
    return mean(ImC) - b
end

function find_density(a,b,λ,ς²::Vector,p::ERMParameter)
    #return mean(gᵢ(a,b,p) ./ ((λ .- ς²*gᵣ(a,b,p)).^2 .+ gᵢ(a,b,p)^2))/π
    return mean(ς²*gᵢ(a,b,p) ./ ((λ .- ς²*gᵣ(a,b,p)).^2 .+ (ς²*gᵢ(a,b,p)).^2))/π
end

function gaussian_variation_model_fast(λ, ς²::Vector, p::ERMParameter, a0, b0; plotroot=false)
    #input eigenvalue λ
    #output probability density at λ
    #ς² is the variance distribution of neural activity
    #a0 is the guessed real part of C, and b0 is the guessed imaginary part of C
    
    @unpack ρ = p

    b = 1
    a = 1
    for i = 1:10
        ffa(a0) = fₐ(a0,b,λ,ς²,p)
        a = find_zero(ffa,a)
        ffb(b0) = fᵦ(a,b0,λ,ς²,p)
        b = find_zero(ffb,b,Order1())
    end

    return find_density(a,b,λ,ς²,p)
    
end

function ĝ(x,C,p)
    @unpack n = p
    k0 = guess_zeros(x,p)
    k = f̂_inv(x,p,k0)
    S = 2π^(n/2)/gamma(n/2)
    return S/(2π)^n*k^(n-1)/∂f̂(k,p)*x/C
end

function Ĝ(k,C,p) 
    @unpack n = p
    S = 2π^(n/2)/gamma(n/2)
    return S/(2π)^n*k^(n-1)*f̂(k,p)/(1-C*f̂(k,p))
end


function G(C,p)
    @unpack n, L, ρ, ϵ = p
    #k₂ = π*ρ^(1/n)
    k₂ = π/ϵ
    k₁ = π/L
    if f̂(k₂,p) < 1/C <  f̂(k₁,p)
        G_inv = quadgk_cauchy(x->ĝ(x,C,p),f̂(k₂,p), 1/C, f̂(k₁,p))
        #G_inv = ∫dxĝ(x)/(x-1/C)
    else
        G_inv = quadgk(k->Ĝ(k,C,p),k₁,k₂)[1]
        #G_inv = ∫dkf̂(k)/(1-Cf̂(k))
    end
    return G_inv
end

function 𝔊(C,ς²::Vector,λ,p)
    #sampling method
    @unpack ρ = p
    return mean(ρ*ς² ./ (λ .- ς²*G(C,p))) - C
end

#=
function 𝔊(C,μ,σ,λ,p)
    #direct integral
    @unpack ρ = p
    ς²_min = 0
    ς²_max = 30
    return quadgk(ς²->lognormal(ς²,μ = μ, σ = σ)*ρ*ς²/(λ - ς²*G(C,p)),ς²_min,ς²_max)[1] - C
end
=#


lognormal(x; μ = 0, σ = 1) = 1/(x*√(2π*σ^2))*exp(-(log(x)-μ)^2/(2σ^2))

#=
function principal_value(fun, a, b, pole; ϵ = 1e-6)
    if a<pole<b
        integral1, err = quadgk(fun, a, pole-ϵ, rtol=1e-8)
        integral2, err = quadgk(fun, pole+ϵ, b, rtol=1e-8)
        return integral1+integral2
    else
        integral, err = quadgk(fun, a, b, rtol=1e-8)
        return integral
    end
end
=#


LogRange(x1, x2, n) = 10 .^ range(log10(x1), log10(x2), length=n)


function construct_grid_points(N; dims = 3)
    points = zeros(N^dims, dims)
    xs = LinRange(-N/2,N/2,N)
    ys = LinRange(-N/2,N/2,N)
    zs = LinRange(-N/2,N/2,N)
    p = 0
    for j in xs
        for i in ys
            for k in zs
                p+=1
                points[p,:]=[i j k]
            end
        end
    end
    return points
end

function model(dim, d, α; f = "normal")
    global p = 0
    plot(p)
    for i = 6:10
    n = 2^i
    if f == "normal"
        x = randn(n,dim)
    elseif f == "uniform"
        x = rand(n,dim)
    end
    l = zeros(n,n)
    for i = 1:n
        for j = i:n
            l[i,j] = l[j,i] = sqrt(sum(abs2, x[j,:] - x[i,:]))
        end
    end
    c = (l.+d).^(-α)

    eig = sort(eigvals(c), rev=true)
    plot!((1:n)/n, eig, xaxis=:log, yaxis=:log)
    global p =plot!((1:n)/n, eig, xaxis=:log, yaxis=:log) 
    end

    plot(p)
    png("figures/"*"power"*"_normal"*"_$d"*"_$α")
end

## convert Covariance matrix to Distance matrix with customized function

function cov2dmat(G::AbstractMatrix{T}; ϵ = 0.1f0, μ = 0.3f0) where {T<:Real}
    # argument checking
    m = size(G, 1)
    n = size(G, 2)
    m == n || error("D should be a square matrix.")
    
    D = similar(G,T)

    # implementation
    for j = 1:n
        for i = 1:j-1
            @inbounds D[i,j] = D[j,i]
        end
        D[j,j] = zero(T)
        for i = j+1:n
            η = abs((G[i,i] + G[j,j])/(2*G[i,j]))
            @inbounds D[i,j] = ϵ*(η^(1/μ)-1)
        end
    end
    return D
end

function replace_matrix_diag(C)
    N = size(C,1)
    Δ = diag(C)
    δ = shuffle(Δ)
    return C - diagm(Δ) + diagm(δ)
end

function find_nearest_neighbor(D)
    n = size(D,1) 
    d = zeros(n-1)
    for j = 1:n-1
        d[j]=minimum(D[j,j+1:n])
    end
    return d
end

function hirarchical_clustering(C;k=2)
    D = gram2dmat(C)
    result = hclust(D, linkage=:single)
    z = cutree(result,k=k) 
    return result, z 
end

function activity_distribution(C; n = 20)
    var = diag(C)
    xrange = LogRange(minimum(var), maximum(var), n)
    h = fit(Histogram,var,xrange)
    h = normalize(h, mode=:pdf)
    x, p = dropzero(h)
    plot(x,p, markershape=:circle, xaxis = :log)
    png("figures/variance_distribution.png")
end


function gaussian_variation_model_fast_ab(λ, ς²::Vector, p::ERMParameter, a0, b0; plotroot=false)
    #input eigenvalue λ
    #output probability density at λ
    #ς² is the variance distribution of neural activity
    #a0 is the guessed real part of C, and b0 is the guessed imaginary part of C
    
    @unpack ρ = p

    b = 1
    a = 1
    for i = 1:10
        ffa(a0) = fₐ(a0,b,λ,ς²,p)
        a = find_zero(ffa,a)
        ffb(b0) = fᵦ(a,b0,λ,ς²,p)
        b = find_zero(ffb,b,Order1())
    end

    return find_density(a,b,λ,ς²,p), a, b
    
end

function mdscale(D; n=2, criterion = "sammon", weight = nothing, Start = "cmdscale")
    if weight == nothing
        mat"""
        [$X, $stress]= mdscale([$D],[$n], 'criterion',[$criterion],'Start',[$Start]); 
        """
    else
        mat"""
        [$X, $stress]= mdscale([$D],[$n], 'criterion',[$criterion],'weight',[$weight],'Start',[$Start]); 
        """
    end
    return X', stress
end

function umeyama(X,Y)
    N = size(X,2)
    d = size(X,1)
    if d == 2
        X2 = vcat(X,zeros(1,N))
        Y2 = vcat(Y,zeros(1,N))
    end
    mat"""
    [$R, $t]= umeyama(X2, Y2); 
    """
    if d == 2
        R = R[1:2,1:2]
        t = t[1:2]
    end
    return R, t
end

function Parameter_estimation(C; ϵ = 0.03125, bin = 0:0.01:1, bin_fit = 0.1:0.01:0.5, n = 2)
    cd1 = max.(0,reshape(C,:)) 
    S = 2π^(n/2)/gamma(n/2)
    h = StatsBase.fit(Histogram, cd1, bin)
    h = normalize(h, mode = :pdf)
    cor, p_corr = dropzero(h)
    a, b = log_fit(cor[10:end-10], log.(p_corr[10:end-10]))
    μ = -n/(1+b)
    N = size(C)[1]
    ρ = N*μ*exp(a)/(S*ϵ^n)
    return ρ, μ
end

function power_exp(r; ϵ = 0.01, ξ = 1, μ = 0.3)
    c = (r/ϵ)^(-μ) * exp(-(r-ϵ)/ξ)
    return c
end

function inv_f(c, ϵ, ξ, μ)
    p_e(r) = power_exp(r; ϵ = ϵ, ξ = ξ, μ = μ)-c
    r = find_zero(p_e,0.0001, Order1())
    return r
end

function Parameter_estimation2(C; ϵ = 0.03125, μ = nothing, bin = 0:0.01:1, st = 10, distribution = "uniform")
    cd1 = max.(0,reshape(C,:)) 
    #cd1 = abs.(reshape(C,:))
    h = StatsBase.fit(Histogram, cd1, bin)
    h = normalize(h, mode = :pdf)
    function loss(x, a0; μ = μ, n = 3086, ϵ = 0.01)
        p, c = x
        if μ == nothing
            ρ, ξ, μ = a0
        else
            ρ, ξ = a0
        end
        rr = inv_f(c, ϵ, ξ, μ)
        if distribution == "uniform"
            l1= -log(p)
            l2 = - log(c/2/pi)
            l3 = + 2*log(rr)
            l4 = - log((μ +rr/ξ))
            ρ = max(ρ, 10)
            l5 = - log(n/ρ)
            l = l1+l2+l3+l4+l5
            #l = -log(p) - log(c/2/pi) + 2*log(rr) - log((μ +rr/ξ)*n/ρ)
        elseif distribution == "normal"
            l = -log(p) - log(c*2) + 2*log(rr) - log((μ +rr/ξ)*n/ρ) - rr^2/(4*n/ρ)
        end
        return l
    end
    x = hcat(h.weights[st:end], bin[st:end-1])
    x = hcat(h.weights[st:50], bin[st:50])
    if μ == nothing
        a0 = [1000, 0.1, 0.3]
        coefs, converged, iter = nonlinear_fit(x, loss, a0, 1e-3, 200)
        ρ, ξ, μ = coefs
    else
        a0 = [1000, 0.1]
        coefs, converged, iter = nonlinear_fit(x, loss, a0, 1e-3, 200)
        ρ, ξ = coefs
    end
    return ρ, ξ, μ
end


function diff_gaussian_variation_model(λ; p = nothing, ς² = nothing, dρ = 10^-6, a0 = 1, b0 = 1)
    if p == nothing
        N = 1024; L = 10; ρ = 5.12; n = 2; ϵ = 0.03125; μ = 0.5; ξ = 10
    else
        @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    end
    if ς² == nothing
        ς² = vec(ones(N,1))
    end
    p0 = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = ϵ, μ = μ, ξ = ξ, β =β, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    P_λ = gaussian_variation_model_fast(λ,ς²,p0,a0,b0,plotroot=true)
    p1 = ERMParameter(;N = N, L = L, ρ = ρ+dρ, n = n, ϵ = ϵ, μ = μ, ξ = ξ, β =β, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    P_λ1 = gaussian_variation_model_fast(λ,ς²,p1,a0,b0,plotroot=true)
    dP_λdρ = (P_λ1 - P_λ)/dρ
    return dP_λdρ/P_λ*ρ  # dlnp/dlnρ
    #return dP_λ 
end

function diff_gaussian_variation_theory(λ;N = 1024, L = 10, ρ = 5.12, n = 2, ϵ = 0.03125, μ = 0.5, 
    ξ = 10, ς² = nothing, dρ = 10^-6, a0 = 1, b0 = 1)
    if ς² == nothing
        ς² = vec(ones(N,1))
    end
    p0 = ERMParameter(;N = N, L = L, ρ = ρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    P_λ, a, b = gaussian_variation_model_fast_ab(λ,ς²,p0,a0,b0,plotroot=true)
    p1 = ERMParameter(;N = N, L = L, ρ = ρ+dρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    P_λ1, a1, b1 = gaussian_variation_model_fast_ab(λ,ς²,p1,a0,b0,plotroot=true)
    dgᵢdρ = (gᵢ(a1,b1,p1) - gᵢ(a,b,p0))/dρ
    dgᵣdρ = (gᵣ(a1,b1,p1) - gᵣ(a,b,p0))/dρ
    dpdρ = dgᵢdρ .* ς² ./ (λ .- ς².* gᵣ(a,b,p0)).^2 .+ dgᵣdρ .* gᵢ(a,b,p0) .* ς².^2 ./ (λ .- ς².* gᵣ(a,b,p0)).^3
    return  mean(dpdρ/P_λ)/π*ρ
end

function diff_high_density_model(λ;N = 1024, L = 10, ρ = 5.12, n = 2, ϵ = 0.03125, μ = 0.5, 
    ξ = 10, ς² = nothing, dρ = 10^-6, a0 = 1, b0 = 1)
    if ς² == nothing
        ς² = vec(ones(N,1))
    end
    σ̄² = mean(ς²)
    p0 = ERMParameter(;N = N, L = L, ρ = ρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    _, P_λ = eigendensity(λ/σ̄²,p0, λ_min = 0)
    P_λ = P_λ/σ̄²
    p1 = ERMParameter(;N = N, L = L, ρ = ρ+dρ, n = d, ϵ = ϵ, μ = μ, ξ = ξ, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    _, P_λ1 = eigendensity(λ/σ̄²,p1, λ_min = 0)
    P_λ1 = P_λ1/σ̄²
    dP_λ = (P_λ1 - P_λ)./dρ
    return dP_λ./P_λ*ρ
    #return dP_λ 
end

function eigendensity_kernel(C::Matrix; length = 100, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10^10,
    m = 1, different_neural_variability = false, theory = true, σ² = nothing, bandwidth = nothing, N_sample = nothing)
    
    N_all = size(C)[1]
    if N_sample ==nothing
        N_sample = N_all
    end

    λ_all = zeros(m,N_sample)
    for i = 1:m
        id = sample(1:N_all,N_sample, replace = false)
        λ = eigvals(C[id,id])
        λ = max.(λ,10^-36)
        λ_all[i,:] = λ
    end

    λ = reshape(λ_all,N_sample*m)

    if bandwidth == nothing
        y = kde(λ)
    else
        y = kde(λ, bandwidth = bandwidth)
    end

    return y, λ
end

function eigendensity_kernel_log(C::Matrix; length = 100, correction = false, use_variation_model = false, λ_min = 1, λ_max = 10^10,
    m = 1, different_neural_variability = false, theory = true, σ² = nothing, bandwidth = nothing, N_sample = nothing)

    N_all = size(C)[1]
    if N_sample ==nothing
        N_sample = N_all
    end

    λ_all = zeros(m,N_sample)
    for i = 1:m
        id = sample(1:N_all,N_sample, replace = false)
        λ = eigvals(C[id,id])
        λ = max.(λ,10^-36)
        λ_all[i,:] = λ
    end

    λ = reshape(λ_all,N_sample*m)

    if bandwidth == nothing
        y = kde(log.(λ))
    else
        y = kde(log.(λ), bandwidth = bandwidth)
    end

    return y, λ
end


function pdf_log_kernal(y,x)
    return pdf(y,log.(x)) ./ x
end


function dlnpdlnρ_fit(C, λ; fit_length = 10, N_sample = [1000,500], bandwidth = nothing, m = 1)
    fit_length = fit_length
    N_sample_fit = LogRange(N_sample[1],N_sample[2],fit_length)
    N_sample_fit = Int.(round.(N_sample_fit))
    p_all = zeros(fit_length,length(λ))
    for i = 1:fit_length
        y, _ = eigendensity_kernel_log(C, m = m, N_sample = N_sample_fit[i], bandwidth = bandwidth)
        p_all[i,:] .= pdf_log_kernal(y,λ)
    end
    
    dlnpdlnρ = zeros(size(λ))
    for i = 1:length(λ)
        ft = curve_fit(LinearFit, log.(N_sample_fit), log.(p_all[:,i]))
        dlnpdlnρ[i] = ft(1) - ft(0)
    end
    return dlnpdlnρ
end

function dlnpdlnρ_fit(p::ERMParameter, λ; σ = nothing, fit_length = 10, N_sample = [1000,500], bandwidth = nothing, m = 1)
    fit_length = fit_length
    N_sample_fit = LogRange(N_sample[1],N_sample[2],fit_length)
    N_sample_fit = Int.(round.(N_sample_fit))
    p_all = zeros(fit_length,length(λ))
    for i = 1:fit_length
        points_i = rand(Uniform(-L/2,L/2),N_sample[1],d)
        C_i = reconstruct_covariance(points_i, p, subsample = false)
        if σ != nothing
            #σ_i = sample(σ,N_sample_fit[i], replace = false)
            Δ_i = diagm(σ)
            C_i = Δ_i * C_i * Δ_i
        end
        C_i = (C_i+C_i')/2
        y, _ = eigendensity_kernel_log(C_i, m = m, N_sample = N_sample_fit[i], bandwidth = bandwidth)
        p_all[i,:] .= pdf_log_kernal(y,λ)
    end
    
    dlnpdlnρ = zeros(size(λ))
    for i = 1:length(λ)
        ft = curve_fit(LinearFit, log.(N_sample_fit), log.(p_all[:,i]))
        dlnpdlnρ[i] = ft(1) - ft(0)
    end
    return dlnpdlnρ
end

#=
function collapse_index_num(C; λ_max = 10, λ_min = 1, dλ = 10^-3, m = 32, bandwidth = nothing, N_sample = nothing, use_log = true, fit_length = 2)
    #λ_corr, p_corr = eigendensity_range(Corr, λ_min = 0.1, λ_max = 100, length = 20)
    #y, λ = eigendensity_kernel(Corr)
    #N_c = size(C)[1]
    #N_c2 = Int(round(N_c/2))
    if N_sample == nothing
        N_sample1 = size(C)[1]
        N_sample2 = Int(round(size(C)[1] /2))
        N_sample = [N_sample1, N_sample2]
    end
    N = size(C)[1]
    if N < N_sample[1]
        N_sample[1] = N
    end
    λ = λ_min:dλ:λ_max
    if use_log 
        #y1, _ = eigendensity_kernel_log(C, m = m, N_sample = N_sample[1], bandwidth = bandwidth)
        #y2, _ = eigendensity_kernel_log(C, m = m, N_sample = N_sample[2], bandwidth = bandwidth)
        #p1 = pdf_log_kernal(y1,λ)
        #p2 = pdf_log_kernal(y2,λ)
        dlnpdlnρ = dlnpdlnρ_fit(C, λ; fit_length = fit_length, N_sample = N_sample, bandwidth = bandwidth, m = m)
    elseif !use_log
        points_i = rand(Uniform(-L/2,L/2),N_sample_fit[i],d)
        C_i = reconstruct_covariance(points1, p, subsample = false)
        y1, _ = eigendensity_kernel(C_i, m = m, N_sample = N_sample[1], bandwidth = bandwidth)
        y2, _ = eigendensity_kernel(C_i, m = m, N_sample = N_sample[2], bandwidth = bandwidth)
        p1 = max.(pdf(y1,λ),0)
        p2 = max.(pdf(y2,λ),0)
        dlnpdlnρ = (log.(p1)-log.(p2) )/ log(N_sample[1]/N_sample[2])
    end
    index = mean(abs.(dlnpdlnρ)) *(λ_max-λ_min)
    return index, dlnpdlnρ, λ 
end
=#

function collapse_index_num(C; λ_max = 10, λ_min = 1, dλ = 10^-3, m = 32, bandwidth = nothing, N_sample = nothing, use_log = true, fit_length = 2)
    #λ_corr, p_corr = eigendensity_range(Corr, λ_min = 0.1, λ_max = 100, length = 20)
    #y, λ = eigendensity_kernel(Corr)
    #N_c = size(C)[1]
    #N_c2 = Int(round(N_c/2))
    if N_sample == nothing
        N_sample1 = size(C)[1]
        N_sample2 = Int(round(size(C)[1] /2))
        N_sample = [N_sample1, N_sample2]
    end
    N = size(C)[1]
    if N < N_sample[1]
        N_sample[1] = N
    end
    λ = λ_min:dλ:λ_max
    if use_log 
        #y1, _ = eigendensity_kernel_log(C, m = m, N_sample = N_sample[1], bandwidth = bandwidth)
        #y2, _ = eigendensity_kernel_log(C, m = m, N_sample = N_sample[2], bandwidth = bandwidth)
        #p1 = pdf_log_kernal(y1,λ)
        #p2 = pdf_log_kernal(y2,λ)
        dlnpdlnρ = dlnpdlnρ_fit(C, λ; fit_length = fit_length, N_sample = N_sample, bandwidth = bandwidth, m = m)
    elseif !use_log
        points_i = rand(Uniform(-L/2,L/2),N_sample_fit[i],d)
        C_i = reconstruct_covariance(points1, p, subsample = false)
        y1, _ = eigendensity_kernel(C_i, m = m, N_sample = N_sample[1], bandwidth = bandwidth)
        y2, _ = eigendensity_kernel(C_i, m = m, N_sample = N_sample[2], bandwidth = bandwidth)
        p1 = max.(pdf(y1,λ),0)
        p2 = max.(pdf(y2,λ),0)
        dlnpdlnρ = (log.(p1)-log.(p2) )/ log(N_sample[1]/N_sample[2])
    end
    index = mean((abs.(dlnpdlnρ))./λ) *(λ_max-λ_min)
    return index, dlnpdlnρ, λ 
end

function collapse_index_sim(p; σ = nothing, λ_max = 10, λ_min = 1, dλ = 10^-3, m = 32, bandwidth = nothing, N_sample = nothing, use_log = true, fit_length = 2)
    if N_sample == nothing
        N_sample1 = p.N
        N_sample2 = Int(round(p.N /2))
        N_sample = [N_sample1, N_sample2]
    end
    λ = λ_min:dλ:λ_max
    if use_log 
        dlnpdlnρ = dlnpdlnρ_fit(p, λ; σ = σ, fit_length = fit_length, N_sample = N_sample, bandwidth = bandwidth, m = m)
    elseif !use_log
        y1, _ = eigendensity_kernel(C1, m = m, N_sample = N_sample[1], bandwidth = bandwidth)
        y2, _ = eigendensity_kernel(C2, m = m, N_sample = N_sample[2], bandwidth = bandwidth)
        p1 = max.(pdf(y1,λ),0)
        p2 = max.(pdf(y2,λ),0)
        dlnpdlnρ = (log.(p1)-log.(p2) )/ log(N_sample[1]/N_sample[2])
    end
    index = mean(abs.(dlnpdlnρ)) *(λ_max-λ_min)
    return index, dlnpdlnρ, λ 
end

#=
function collapse_index_theory(p; λ_max = 10, λ_min = 1, dλ = 10^-3, m = 32, bandwidth = nothing, N_sample = nothing, len = 10, ς² = nothing)
    #λ_corr, p_corr = eigendensity_range(Corr, λ_min = 0.1, λ_max = 100, length = 20)
    #y, λ = eigendensity_kernel(Corr)
    #N_c = size(C)[1]
    #N_c2 = Int(round(N_c/2))
    λ_all =  λ_min:(λ_max - λ_min) / (len - 1):λ_max
    dlnP_λ_all_corr = diff_gaussian_variation_model.(λ_all, p = p, ς² = ς²)

    A = hcat(λ_all,dlnP_λ_all_corr)
    itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), λ_all, 1:2)

    tfine = λ_min:dλ:λ_max
    λ, dlnpdlnρ = [itp(t,1) for t in tfine], [itp(t,2) for t in tfine]

    index = mean(abs.(dlnpdlnρ)) *(λ_max-λ_min)
    return index, dlnpdlnρ, λ 
end
=#

function collapse_index_theory(p; λ_max = 10, λ_min = 1, dλ = 10^-3, m = 32, bandwidth = nothing, N_sample = nothing, len = 10, ς² = nothing)
    #λ_corr, p_corr = eigendensity_range(Corr, λ_min = 0.1, λ_max = 100, length = 20)
    #y, λ = eigendensity_kernel(Corr)
    #N_c = size(C)[1]
    #N_c2 = Int(round(N_c/2))
    λ_all =  λ_min:(λ_max - λ_min) / (len - 1):λ_max
    dlnP_λ_all_corr = diff_gaussian_variation_model.(λ_all, p = p, ς² = ς²)

    A = hcat(λ_all,dlnP_λ_all_corr)
    itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), λ_all, 1:2)

    tfine = λ_min:dλ:λ_max
    λ, dlnpdlnρ = [itp(t,1) for t in tfine], [itp(t,2) for t in tfine]

    index = mean((abs.(dlnpdlnρ))./λ) *(λ_max-λ_min)
    return index, dlnpdlnρ, λ 
end

function collapse_index_theory2(p; p0 = 0.01, λ_max = 10, λ_min = 1, dλ = 10^-3, m = 32, bandwidth = nothing, N_sample = nothing, len = 10, ς² = nothing)
    #λ_corr, p_corr = eigendensity_range(Corr, λ_min = 0.1, λ_max = 100, length = 20)
    #y, λ = eigendensity_kernel(Corr)
    #N_c = size(C)[1]
    #N_c2 = Int(round(N_c/2))
    @unpack L, n, N = p 
    points = rand(Uniform(-L/2,L/2),Int(N/2),d)
    C = reconstruct_covariance(points, p, subsample = false)
    lb_N = -sort(-eigvals(Hermitian(C)))
    λ_max = lb_N[Int(round(p0*N/2))]
    λ_all =  λ_min:(λ_max - λ_min) / (len - 1):λ_max
    dlnP_λ_all_corr = diff_gaussian_variation_model.(λ_all, p = p, ς² = ς²)

    A = hcat(λ_all,dlnP_λ_all_corr)
    itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), λ_all, 1:2)

    tfine = λ_min:dλ:λ_max
    λ, dlnpdlnρ = [itp(t,1) for t in tfine], [itp(t,2) for t in tfine]

    index = mean((abs.(dlnpdlnρ))./λ) *(λ_max-λ_min) / log(λ_max/λ_min)
    return index, dlnpdlnρ, λ 
end

function collapse_index_rank(C; p0 = 0.01)
    
    s = 0.5 # subsampling fraction
    #p0 = 0.01 # lambda_max cutoff

    N,_ = size(C)
    Ns = round(Int, N*s)
    lb_N = -sort(-eigvals(Hermitian(C)))
    ntrial = 2000
    lb_Ns = zeros(Ns,ntrial)
    Threads.@threads for i in 1:ntrial
        i_s = randperm(N)[1:Ns]
        Cs = copy(C[:,i_s])
        Cs = Cs[i_s,:]
        lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs)))
    end
    lb_Ns = vec(mean(lb_Ns,dims=2))

    r0 = round(Int, Ns*p0-0.5) # lambda_max cutoff
    r1 = maximum(findall(lb_Ns .> 1)) # lambda_min cutoff
    f_Ns = log.(lb_Ns[r0+1:r1])
    f_N = zeros(length(f_Ns))
    for k in 1:length(f_N)
        i = r0 + k
        j = (i)/Ns*N
        if isinteger(j)
            j = convert(Int,j)
            f_N[k] = log.(lb_N[j])
        else
            # interpolation in loglog scale
            j0 = convert(Int,floor(j))
            # f_N[k] = (j0+1-j)*lb_N[j0] + (j-j0)*lb_N[j0+1]
            #=
            x = np.log(j/N)
            x0 = np.log(j0/N)
            x1 = np.log((j0+1)/N)
            y0 = np.log(lb_N[j0])
            y1 = np.log(lb_N[j0+1])
            =#
            x = log(j/N)
            x0 = log(j0/N)
            x1 = log((j0+1)/N)
            y0 = log(lb_N[j0])
            y1 = log(lb_N[j0+1])
            f_N[k] = (y0*(x1-x) + y1*(x-x0)) / (x1-x0)
        end
    end
    S = 0
    for k in 1:length(f_N)-1
        i = r0 + k
        a = (i)/Ns
        b = (i+1)/Ns
        y1 = abs(f_N[k] - f_Ns[k])
        y2 = abs(f_N[k+1] - f_Ns[k+1])
        S += log(b/a)*(y1+y2)/2
    end
    CI = S/log(N/Ns)/log((r1)/(r0+1))
    return CI, (r0+1)/Ns, r1/Ns
end

function loss_mds(C,p)
    D2 = pairwise(Euclidean(), X', dims=1)
    C2 = corr(D2,p)  
    loss = Flux.mse(C,C2)
    return loss
end


function loss_mds(C)
    D2 = pairwise(Euclidean(), X', dims=1)
    loss = Flux.mse(C,D2)
    return loss
end


function pairwise_dot_kai(x)
    d, n = size(x)
    xixj = x' * x
    xsq = sum(x .^ 2; dims=1)
    return repeat(xsq, n, 1) + repeat(xsq', 1, n) - 2xixj
end

dis(x) = @tullio C[i,j] := x[k, i] ^ 2 - 2 * x[k, i] * x[k, j] + x[k, j] ^ 2

function loss_mds(C,p)
    D2 = sqrt.(abs.(dis(X)))
    C2 = corr(D2,p)  
    loss = Flux.mse(C,C2)
    return loss
end

function G_int(z, p; C = nothing)
    @unpack L, ϵ, ρ, n = p
    k₁ = π/L
    k₂ = π/ϵ 
    a = f̂(k₁, p)
    b = f̂(k₂, p)
    if C == nothing
        c = z/ρ
    else
        c = 1/C
    end
    #fun(x) = -2*π*f̂_inv(x,p,0.1)*x*z/ρ/∂f̂(f̂_inv(x,p,0.1),p) / (2*π)^2
    fun(x) = -2*π*f̂_inv(x,p,guess_zeros(x,p))*x*c/∂f̂(f̂_inv(x,p,guess_zeros(x,p)),p) / (2*π)^2
    if (c>a&&c<b)||(c<a&&c>b)
        λ = quadgk_cauchy(fun, a, c, b)
    else
        λ = quadgk(x ->fun(x)/(x-c), a, b)[1]
    end
    return λ
end


function ratio(z, p; C = nothing)
    return G_int(z, p, C = C)/z
end


function find_ratio(z, p; iteration = 5)
    @unpack ρ = p
    C0 = ρ/z
    #G(k, C) = f̂(k, p)/(1 - C * f̂(k, p))
    C = C0
    for i = 1:iteration
        C = C0 / (1 -ratio(z, p, C = C))
    end
    return ratio(z, p, C = C)
end

function ratio_complex(λ, p)
    @unpack N, ρ = p
    _, a, b = gaussian_variation_model_fast_ab(λ, vec(ones(N,1)), p, 1, 1)
    a, b = a/ρ, b/ρ
    ratio_re = 1 - a/λ/(a^2+b^2)
    ratio_im = b/λ/(a^2+b^2)
    return ratio_re, ratio_im
end

function ratio_theory(z, p)
    @unpack ϵ, μ, ρ, n, L = p
    G_int = (π/ϵ)^μ/μ - ρ^(μ/(n-μ))*z^(-μ/(n-μ))*π*cot(π*n/(n-μ))*1.5 - (π/L)^μ/μ
    return G_int/z
end

function generate_f̂_interpolation(p; fun = f̂_numerical, len = 10^4)
    @unpack ϵ, L = p
    k₁ = 10^-12#π/L
    k₂ = π/ϵ * 10
    k = range(k₁, k₂, length = len)
    f(x) = fun(x,p)
    f̂_k = f.(k)
    return CubicSpline(f̂_k,k)
end

function mds_fig(X, Corr, root)
    D2 = pairwise(Euclidean(), X)
    C2 = corr(D2,p) 
    λ_sim, p_sim = eigendensity(C2, correction = false, λ_min = 0.5)
    λ_id = findall(λ_sim.> 0.1 )
    plot(λ_sim[λ_id], p_sim[λ_id], label = "$ifish _refactoring")
    plot!(xlabel = L"\lambda", ylabel = "pdf", xaxis = :log, yaxis = :log)
    λ_sim, p_sim = eigendensity(Corr, correction = false, λ_min = 0.5)
    plot!(λ_sim[1:end], p_sim[1:end], label = "$ifish")
    plot!(title=L"L = %$(round(p.L,digits=3)), \mu = %$(round(p.μ ,digits=3)), \epsilon = %$(round(p.ϵ,digits=3)), n =  %$(p.n), \beta = %$(round(p.β,digits=3)), E(\sigma^4) = %$(round(p.σ̄⁴,digits=2))")
    savefig(joinpath(root,"corr/lambda_density.png"))
    
    subsampling(C2, joinpath(root,"corr"))
    subsampling(Corr, joinpath(root,"corr/data"))

    R = hclust(D,linkage = :average)
    heatmap(Corr[R.order, R.order],theme = :dark, clim = (0,1))
    savefig(joinpath(root,"corr/corr.png"))
    heatmap(C2[R.order, R.order],theme = :dark, clim = (0,1))
    savefig(joinpath(root,"corr/corr_refactoring.png"))

    scatter(X[1,R.order],X[2,R.order], marker_z = 1:size(X,2), markersize = 1,  color = :jet)
    savefig(joinpath(root,"corr/point_cloud.png"))

    marginalkde(X[1,R.order],X[2,R.order])
    savefig(joinpath(root,"corr/marginalkde.png"))
end


function m_mds(C, p, resultroot; epoch = 10000, Descent0 = 1000, dacay_time = 10)
    @unpack β, n, N = p
    D = find_D(C, p)
    D = D - Diagonal(D)
    if β!=0
        global X, stress = mdscale(D,n = n, criterion = "sammon")
    else
        global X = randn(n,N)
    end
    C = C|>gpu
    global X = X|>gpu
    ps = Flux.params(X) |>gpu
    opt = Descent(Descent0)

    for i = 1:dacay_time
        opt = Descent(Descent0*exp((-i+1)*0.5))
        @time @epochs epoch Flux.train!(loss_mds, ps, [(C,p)], opt, cb = Flux.throttle(() -> println(loss_mds(C,p)), 10))
        time_now = Dates.format(now(), "Y_mm_dd_HH_MM")
        global X_cpu = X|>cpu
        resultroot2 = mkdir(joinpath(resultroot,"$time_now"))
        mkdir(joinpath(resultroot,"$time_now/corr"))
        mkdir(joinpath(resultroot,"$time_now/corr/data"))
        f = open(joinpath(resultroot2,"loss.txt"),"w")
            write(f,"loss=$(loss_mds(C,p))")
        close(f)
        matwrite(joinpath(resultroot2, "mmds"*"_$ifish.mat"),
            Dict("X"=>X_cpu', "ifish"=>ifish, "xi"=>p.ξ, "L"=>p.L, "rho"=>p.ρ, 
            "epsilon"=>p.ϵ, "mu"=>p.μ, "beta"=>p.β, "n"=>p.n, "loss" =>loss_mds(C,p)))
        if isnan(X_cpu[1])
            break
        end
        mds_fig(X_cpu, C|>cpu, resultroot2)
    end
end

#=
function corr(D,p)  
    @unpack μ, ϵ, β = p 
    C_orr = copy(D)
    #C_orr[D .<= ϵ] .= 1
    #C_orr[D .> ϵ] .= ϵ^μ*(D[D .> ϵ]).^(-μ).*exp.(-(D[D .> ϵ] .- ϵ)*β)
    C_orr = min.(ϵ^μ*(D).^(-μ).*exp.(-(D .- ϵ)*β),1)
    return C_orr
end
=#

function corr(D,p)  
    @unpack μ, ϵ, β = p 
    C_orr = copy(D)
    #C_orr[D .<= ϵ] .= 1
    #C_orr[D .> ϵ] .= ϵ^μ*(D[D .> ϵ]).^(-μ).*exp.(-(D[D .> ϵ] .- ϵ)*β)
    C_orr = ϵ^μ*(D.^2 .+ϵ.^2).^(-μ/2)
    return C_orr
end

#=
function find_D(C,p)
    @unpack μ, ϵ, β = p 
    fc(D,C) = ϵ^μ*(D).^(-μ).*exp.(-(D.- ϵ)*β) - C
    N = size(C,1)
    fcd = ZeroProblem(fc,ϵ)
    #a = solve(fcd,c)
    #d = find_zero(fc,ϵ)
    [D[i,j] = solve(fcd,C[i,j]) for i = 1:N for j = 1:N]
    return D
end
=#

function find_D(C,p)
    @unpack μ, ϵ, β, L = p 
    L = L
    D = copy(C)
    D .= ϵ*sqrt.(abs.(C.^(-2/μ) .- 1))
    D[D.>L] .= L*log.((D[D.>L] ./L)) .+L
    return D
end

function CCA_corr_dim(Corr; n_all = 2:10, ifish = 210106, figroot = nothing)
    corrs = []
    if figroot == nothing 
        #figroot = "/home/wenlab-user/wangzezhen/FISHRNN_TEST/results/CCA/$ifish"
    end
    corr_shuffle = zeros(length(n_all), 100)
    i = 0
    for n in n_all
        i = i+1
        d = n
        ρ, μ = Parameter_estimation(Corr, n = n)
        N = size(Corr)[1]
        L = (N/ρ)^(1/n)
        p = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = 0.03125, μ = μ, ξ = 10, σ̄² = 1, σ̄⁴ = 1)
        C = Corr
        #C = max.(C,0.01)
        C = abs.(C)
        D = copy(C)
        D = find_D(C, p)
        D = D - Diagonal(D)
        D = (D+D')/2
        #C = min.(C,1); D = sqrt.(1 .-C)
        #X = classical_mds(D, n); X = Float64.(X)
        X, stress = mdscale(D, n = n, criterion = "sammon")
        ROI_MDS_CCA = fit(CCA,ROIxyz',X)
        corr_cca = correlations(ROI_MDS_CCA)
        if n == 1
            corrs = vcat(corrs, corr_cca'[1])
            corrs = vcat(corrs, 0)
        else
            corrs = vcat(corrs, corr_cca'[1:2])
        end
        for j = 1:100
            shuffle_index = randperm(size(X,2))
            ROI_MDS_CCA_shuffle = fit(CCA,ROIxyz',X[:,shuffle_index])
            corr_shuffle[i,j] = correlations(ROI_MDS_CCA_shuffle)[1]
        end
    end
    corrs = reshape(corrs,2,:)
    plot(n_all,corrs[1,:],ylims = (0,1),label = "CCA_corr1,ifish = $ifish")
    plot!(n_all,corrs[2,:],ylims = (0,1),label = "CCA_corr2,ifish = $ifish")
    plot!(xlabel = "mds_dimension", ylabel = "CCA_corr")
    if figroot != nothing
        savefig(joinpath(figroot,"CCAcorrdim.png"))
        savefig(joinpath(figroot,"CCAcorrdim.svg"))
    end
    return corrs, corr_shuffle
end


function subsample_fig6(C, fig; subfig = 1, is_pyfig = false, fontsize =5)
    if is_pyfig == false
        pyfig = fig.o
    else
        pyfig = fig
    end
    plt.rc("font",family="Arial")
    plt.rc("pdf", fonttype=42)
    ax1 = pyfig[:axes][subfig]

    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.2,marker="o",color=colorList[i,:],
        ms=0.3,label=@sprintf("N=%d",2^n))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color = colorList[i,:])

        append!(eigenspectrums,eigenvalues[4:end])
        append!(nranks, k[4:end])
    end
    ax1.set_ylim([1e-1, 1e+2])
    ax1.set_xlim([0.1^(3.2), 1])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N")
    ax1.set_ylabel("eigenvalue "* L"\lambda")
    #ax1.legend()
    ax1.legend(loc="upper right", fontsize = fontsize)
    return pyfig
end

function subsample_fig7(C, fig; subfig = 1, is_pyfig = false, fontsize =5, CI = 0.67, dataset= "original")
    if is_pyfig == false
        pyfig = fig.o
    else
        pyfig = fig
    end
    plt.rc("font",family="Arial")
    ax1 = pyfig[:axes][subfig]

    N = size(C,1)
    neural_set = random_clusters(N)
    iterations = floor(log2(N))
    cmap = plt.get_cmap("winter")
    colorList = cmap(range(0, 1, length=4))
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    leg2 = []
    eigenspectrums = []
    nranks = []
    i = 0
    for n in range(iterations, 7, step=-1)
        i+=1
        eigenvalues, errorbars, k = pca_cluster2(neural_set, C, Int(2^n))
        I = findall(eigenvalues .> 0)
        eigenvalues = eigenvalues[I]
        k = k[I]
        errorbars = errorbars[I]
        ax1.plot(k[1:end], eigenvalues[1:end], linewidth=0.2,marker="o",color=colorList[i,:],
        ms=0.3,label=@sprintf("N=%d",2^n))   
        λ_err_l = eigenvalues - errorbars./2
        λ_err_u = eigenvalues + errorbars./2
        ax1.fill_between(k[1:end],λ_err_l,λ_err_u,alpha=0.3,color = colorList[i,:])

        append!(eigenspectrums,eigenvalues[4:end])
        append!(nranks, k[4:end])
    end
    ax1.set_ylim([1e-1, 1e+2])
    ax1.set_xlim([0.1^(3.2), 1])
    c_diag = sort(diag(C), rev=true)
    ax1.scatter((1:N)/N, c_diag,marker="o",label="diagonal",color="gray",alpha=0.3,s=2)
    ax1.set_xlabel("rank/N")
    ax1.set_ylabel("eigenvalue "* L"\lambda")
    #ax1.legend()
    ax1.legend(loc="upper right", fontsize = fontsize)
    ax1.text(1e-3,0.3,s="CI = $CI", fontsize = 2*fontsize)
    ax1.text(1e-3,1,s=dataset, fontsize = 2*fontsize)
    return pyfig
end

function multi_sim_subplot(ax,p,n_sim;different_neural_variability=false)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(L"\lambda")
    ax.set_ylabel(L"p(\lambda)")
    # n_sim = 10
    neural_set = random_clusters(N)
    points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
    C = reconstruct_covariance(points, p, subsample = false)
    eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, p.N)
    eigenspectrum_all = copy(eigenspectrum)
    Threads.@threads for ii = 2:n_sim
        points = rand(Uniform(-p.L/2,p.L/2),p.N,p.n) #points are uniformly distributed in a region 
        C = reconstruct_covariance(points, p, subsample = false)
        eigenspectrum, k = pca_cluster2_multi_sim(neural_set, C, p.N)
        eigenspectrum_all = hcat(eigenspectrum_all,eigenspectrum)
    end
    eigenvalues = mean(eigenspectrum_all, dims=2)
    errorbars = std(eigenspectrum_all, dims=2) ./ sqrt(size(eigenspectrum_all,2))

    λ_min = minimum(eigenspectrum[:])
    λ_max = maximum(eigenspectrum[:])

    id, λ = dropzeros_and_less(eigenspectrum_all[:,1])
    length = 200
    h_λ = fit(Histogram, λ, LogRange(λ_min,λ_max,length))
    # edges, p_sim = normalize(h_λ, mode=:pdf)
    h_λ = normalize(h_λ, mode=:pdf)
    @unpack edges, weights = h_λ       
    binsize = edges[1][2] - edges[1][1]
    x = collect(edges[1][2:end].-binsize/2)
    p_sim_all = vec(weights.*size(id,1)/p.N)

    for ii = 2:size(eigenspectrum_all,2)
        id, λ = dropzeros_and_less(eigenspectrum_all[:,ii])
        length = 200
        h_λ = fit(Histogram, λ, LogRange((λ_min),maximum(λ_max),length))
        h_λ = normalize(h_λ, mode=:pdf)
        @unpack edges, weights = h_λ   
        binsize = edges[1][2] - edges[1][1]
        x = collect(edges[1][2:end].-binsize/2)
        p_sim = vec(weights.*size(id,1)/p.N)
        p_sim_all = hcat(p_sim_all,p_sim)
    end

    p_sim,errorbars = mean_sem_nonzero(p_sim_all, dims=2)
    # errorbars = std(p_sim_all, dims=2) ./ sqrt(size(p_sim_all,2))
    p_err_l = p_sim .- errorbars
    p_err_u = p_sim .+ errorbars

    ax.plot(x, p_sim, linewidth=1,label="ERM simulation")        
    ax.fill_between(x,p_err_l[1:end],p_err_u[1:end],alpha=0.3)


    return ax
end

function subsample_fig4(C, p, fig; subfig = 1, is_pyfig = false, fontsize =5, fontsize2=10,n_sim = 100)
    if is_pyfig == false
        pyfig = fig.o
    else
        pyfig = fig
    end
    plt.rc("font",family="Arial")
    plt.rc("pdf", fonttype=42)
    ax = pyfig[:axes][subfig]

    λ_sim, p_sim, λ_theory, p_theory = eigendensity(C, p, correction = false, λ_min = 0.8, λ_max = 50);
    _, _, λ_theory_GV, p_theory_GV = eigendensity(C, p; length = 20, use_variation_model = true, λ_min = 0.8, λ_max = 50);
    multi_sim_subplot(ax,p,n_sim)
    #plt.rc("font",family="Arial")
    ax.plot(λ_theory, p_theory, label = "high density theory",lw=1,color="gray")
    ax.plot(λ_theory_GV[1:end], p_theory_GV[1:end],lw=1, label = "variational method",color="red")
    ax.loglog()
    # ax.grid(b=true)
    ax.set_xlabel("eigenvalue "* L"\lambda",fontsize = fontsize2)
    ax.set_ylabel("probability density " *L"p(\lambda)",fontsize = fontsize2)
    # ax.legend(loc="bottom right",bbox_to_anchor=(0.5, -0.15))
    ax.legend(loc="lower left", fontsize = fontsize)
    # plt.savefig("high_vs_variation.pdf",bbox_inches="tight",dpi=300)
    ax.set_xlim((1e-2,1e3/2))
    ax.set_ylim(1e-5,1e1)
    #ax.text(-0.1, 1.05, uppercase_letters[ipanel],transform=ax.transAxes,size=10,weight="bold")

    return pyfig
end

function corr2cov(Corr, Cᵣ)
    std_c = sqrt(Diagonal(Cᵣ))
    Cov = std_c*Corr*std_c
    Cov = (Cov+Cov')/2
end


function linearize_triu(A)
    # V is the upper triangular matrix of A in the vector form, excluding the main diagonal
    N = size(A, 1)
    L = Int(N * (N - 1) / 2)
    V = zeros(L)
    IDX2SUB = zeros(L, 2)
    count = 1
    for i in 1:N
        for j in i+1:N
            V[count] = A[i, j]
            IDX2SUB[count, :] = [i, j]
            count += 1
        end
    end
    return V, IDX2SUB
end


function normalize_activity(a)
    N = size(a,1)
    epsilon = 1e-4
    a[a.<epsilon] .= 0
    for i = 1:N
        IDX = a[i,:] .> 0
        mean_activity = mean(a[i,IDX]);
        if mean_activity > 0
            a[i,:] = a[i,:]./mean_activity
        end
    end
    return a
end

function data_preprocessing(FR)

    tf = vec(sum(FR, dims=2) .> 0)
    FR = convert(Array{Float64}, FR[tf, :])
    FR = normalize_activity(FR);

    K = 1024
    # Random.seed!(1)
    i_rand = randperm(size(FR, 1))[1:K]
    FR_sub = FR[i_rand,:]
    C = cov(convert(Matrix, FR_sub'))
    s = diag(C)
    s = mean(s)
    C = C ./ s

    return FR_sub, C
end


function A2corr(a)
    #a = a .- mean(a,dims=2)
    (N,T)=size(a)
    Cᵣ = a*a'/T 
    β = N/tr(Cᵣ)
    Cᵣ = Cᵣ*β

    global id_un0 = findall(diag(Cᵣ) .!=0) 
    Cᵣ =  Cᵣ[id_un0,id_un0]

    std_c = sqrt(Diagonal(Cᵣ))
    Corr = std_c\Cᵣ/std_c
    Corr = (Corr + Corr')/2
    return Cᵣ, Corr, id_un0
end

function cov_behavior(firingrate, y_conv_range)
    frame_behavior = findall(y_conv_range[:,1] .==1)
    frame_nobehavior = findall(y_conv_range[:,1] .==0)
    frame_all_1 = sample(frame_nobehavior, length(frame_nobehavior)-length(frame_behavior), replace = false)
    frame_all_2 = vcat(frame_all_1, frame_behavior)

    
    r_behavior = firingrate[:, frame_all_2]
    r_nobehavior = firingrate[:, frame_nobehavior]

    A_nobehavior = normalization(r_nobehavior, ϵ=1e-4)
    A_nobehavior = A_nobehavior .- mean(A_nobehavior,dims=2)

    A_behavior = normalization(r_behavior, ϵ=1e-4)
    A_behavior = A_behavior .- mean(A_behavior,dims=2)

    Cov_nobehavior, Corr_nobehavior, id_un0_nobehavior = A2corr(A_nobehavior)
    Cov_behavior, Corr_behavior, id_un0_behavior = A2corr(A_behavior)
    return Cov_behavior, Cov_nobehavior
end

function corr_behavior(firingrate, y_conv_range)
    frame_behavior = findall(y_conv_range[:,1] .==1)
    frame_nobehavior = findall(y_conv_range[:,1] .==0)
    frame_all_1 = sample(frame_nobehavior, length(frame_nobehavior)-length(frame_behavior), replace = false)
    frame_all_2 = vcat(frame_all_1, frame_behavior)

    
    r_behavior = firingrate[:, frame_all_2]
    r_nobehavior = firingrate[:, frame_nobehavior]

    A_nobehavior = normalization(r_nobehavior, ϵ=1e-4)
    A_nobehavior = A_nobehavior .- mean(A_nobehavior,dims=2)

    A_behavior = normalization(r_behavior, ϵ=1e-4)
    A_behavior = A_behavior .- mean(A_behavior,dims=2)

    Cov_nobehavior, Corr_nobehavior, id_un0_nobehavior = A2corr(A_nobehavior)
    Cov_behavior, Corr_behavior, id_un0_behavior = A2corr(A_behavior)
    return Corr_behavior, Corr_nobehavior
end


function quantile_variational(λ, p; length_range = 20, ς² = nothing ,p0 = 0.01, λ_max = nothing)
    @unpack ρ, μ, ϵ, n, β, L= p
    N = ρ*L^n
    if ς² == nothing
        ς² = vec(ones(round(Int,N)))
    end
    t = 1#round(N/20)
    if λ_max == nothing #|| λ_max != nothing
        p0 = t/N
        #t = p0*N
        λ_max = ρ*f̂(2*sqrt(π)/L*sqrt(t),p)
    end
    #λ_all = range(λ, λ_max ,length = length_range)
    λ_all = LogRange(λ, λ_max, length_range)
    dλ = λ_all[2] - λ_all[1]
    dlnλ = log(λ_all[2]/λ_all[1])
    GVMF(λ_all) = gaussian_variation_model_fast(λ_all, ς², p, 1, 1)
    p_all = GVMF.(λ_all)
    #q = sum(p_all) * dλ
    q = (sum(p_all.*λ_all) - p_all[1].*λ_all[1]/2 - p_all[end].*λ_all[end]/2)*length_range/(length_range-1) * dlnλ + p0
    return q
end

function dqdρ(λ; p = nothing, dρ = 0.01)
    if p == nothing
        N = 1024; L = 10; ρ = 5.12; n = 2; ϵ = 0.03125; μ = 0.5; ξ = 10
    else
        @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    end
    p0 = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = ϵ, μ = μ, ξ = ξ, β =β, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    q_λ = quantile_variational(λ, p0)
    p1 = ERMParameter(;N = N, L = L, ρ = ρ+dρ, n = n, ϵ = ϵ, μ = μ, ξ = ξ, β =β, σ̄² = mean(ς²), σ̄⁴ = mean(ς².^2))
    q_λ1 = quantile_variational(λ, p1)
    dq_λdρ = (q_λ1 - q_λ)/dρ
    return dq_λdρ
end

function dlnqdlnρ(λ;ς² =nothing, length_range = 20, p = nothing, dρ = 0.01)
    if p == nothing
        N = 1024; L = 10; ρ = 5.12; n = 2; ϵ = 0.03125; μ = 0.5; ξ = 10
    else
        @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    end
    p0 = ERMParameter(;N = N, L = L, ρ = ρ, n = n, ϵ = ϵ, μ = μ, ξ = ξ, β =β, σ̄² = 1, σ̄⁴ = 1)
    q_λ = quantile_variational(λ, p0, length_range = length_range,ς² = ς²)
    p1 = ERMParameter(;N = N, L = L, ρ = ρ+dρ, n = n, ϵ = ϵ, μ = μ, ξ = ξ, β =β, σ̄² = 1, σ̄⁴ = 1)
    q_λ1 = quantile_variational(λ, p1, length_range = length_range,ς² = ς²)
    dlnq_λdlnρ = (q_λ1 - q_λ)/dρ /q_λ*ρ
    return dlnq_λdlnρ
end
   
function dlnqdlnρ2(λ;ς² =nothing, length_range = 20, p = nothing, p2 =nothing, dρ = 0.01,λ_max01=nothing,λ_max02=nothing,p01=nothing,p02=nothing)
    if p == nothing
        N = 1024; L = 10; ρ = 5.12; n = 2; ϵ = 0.03125; μ = 0.5; ξ = 10
    else
        @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    end
    q_λ = quantile_variational(λ, p, length_range = length_range, ς² = ς², λ_max=λ_max01, p0=p01)
    q_λ1 = quantile_variational(λ, p2, length_range = length_range, ς² = ς², λ_max=λ_max02, p0=p02)
    dlnq_λdlnρ = (log(q_λ1) -log(q_λ))/log(p2.ρ/p.ρ)
    return dlnq_λdlnρ
end

function dlnqdlnρ3(λ;ς² =nothing, length_range = 20, p = nothing, p2 =nothing, p3 =nothing, dρ = 0.01,λ_max01=nothing,λ_max02=nothing,λ_max03=nothing,p01=nothing,p02=nothing,p03=nothing)
    if p == nothing
        N = 1024; L = 10; ρ = 5.12; n = 2; ϵ = 0.03125; μ = 0.5; ξ = 10
    else
        @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    end
    q_λ = quantile_variational(λ, p, length_range = length_range, ς² = ς², λ_max=λ_max01, p0=p01)
    q_λ1 = quantile_variational(λ, p2, length_range = length_range, ς² = ς², λ_max=λ_max02, p0=p02)
    q_λ3 = quantile_variational(λ, p3, length_range = length_range, ς² = ς², λ_max=λ_max03, p0=p03)
    dlnq_λdlnρ = (q_λ1 -q_λ)/q_λ3/log(p2.ρ/p.ρ)
    return dlnq_λdlnρ
end

function findmax_λ(p::ERMParameter, ς²; p0 = 0.01, ntrial = 100)
    s = 0.5 # subsampling fraction
    @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    id_c = sample(1:N, N, replace =false)
    lb_N = zeros(N,ntrial)
    Threads.@threads for i in 1:ntrial
        points1 = rand(Uniform(-L/2,L/2),N,n)
        σ = vec(broadcast(√,ς²))[id_c]
        Δ = diagm(σ)
        Δ = Δ[id_c,id_c]
        C = reconstruct_covariance(points1, p, subsample = false)
        lb_N[:,i] = -sort(-eigvals(Hermitian(C)))
    end

    lb_N = vec(mean(lb_N,dims=2))
    r0 = round(Int, N*p0-0.5)
    dr0 = Int(1)
    dr = dr0+1 #Int.(sign(r0%2-0.5)+dr0)
    #return sqrt(lb_N[r0+dr] * lb_N[r0+dr0]), sqrt((r0+dr)/N * (r0+dr0)/N)
    return exp(mean([log(lb_N[r0+dr]), log(lb_N[r0+dr0]), log(lb_N[r0+dr0-1]) ])), exp(mean([log((r0+dr)/N), log((r0+dr0)/N), log((r0+dr0-1)/N) ]))
    #return lb_N[r0+1], (r0+1)/N
end

function findmax_λ0(p::ERMParameter, ς²; p0 = 0.01, ntrial = 100)
    s = 0.5 # subsampling fraction
    @unpack N, L, ρ, n, ϵ, μ, ξ, β = p
    id_c = sample(1:N, N, replace =false)
    lb_N = zeros(N,ntrial)
    Threads.@threads for i in 1:ntrial
        points1 = rand(Uniform(-L/2,L/2),N,n)
        σ = vec(broadcast(√,ς²))[id_c]
        Δ = diagm(σ)
        Δ = Δ[id_c,id_c]
        C = reconstruct_covariance(points1, p, subsample = false)
        lb_N[:,i] = -sort(-eigvals(Hermitian(C)))
    end
    lb_N = vec(mean(lb_N,dims=2))
    r0 = round(Int, N*p0-0.5)
    r1 = maximum(findall(lb_N .> 1))
    dr0 = Int(1)
    return lb_N[r0+1], (r0+1)/N, lb_N[r1], r1/N
end

function findmax_λ(C; p0 = 0.01)
    s = 0.5 # subsampling fraction

    N,_ = size(C)
    Ns = round(Int, N*s)
    lb_N = -sort(-eigvals(Hermitian(C)))
    ntrial = 20
    lb_Ns = zeros(Ns,ntrial)
    Threads.@threads for i in 1:ntrial
        i_s = randperm(N)[1:Ns]
        Cs = copy(C[:,i_s])
        Cs = Cs[i_s,:]
        lb_Ns[:,i] = -sort(-eigvals(Hermitian(Cs)))
    end
    lb_Ns = vec(mean(lb_Ns,dims=2))

    r0 = round(Int, Ns*p0-0.5) # lambda_max cutoff
    r1 = maximum(findall(lb_Ns .> 1)) # lambda_min cutoff
    f_Ns = log.(lb_Ns[r0+1:r1])
    f_N = zeros(length(f_Ns))
    return lb_Ns[r0]
end


function collapse_index_rank_theory(p, C; p0 = 0.01,ς² = nothing, ntrial = 1, λ_min=1, len =100, dλ =0.01, length_range = 20)
    @unpack ρ, μ, ϵ, n, β, N, ξ = p
    ς² = diag(C)

    p2 = ERMParameter(;N = N/2, L = L, ρ = ρ/2, n = d, ϵ = ϵ, μ = μ, ξ = ξ, β = 0, σ̄² = 1, σ̄⁴ = 1)
    p3 = ERMParameter(;N = round(Int,N/sqrt(2)), L = L, ρ = ρ/sqrt(2), n = d, ϵ = ϵ, μ = μ, ξ = ξ, β = 0, σ̄² = 1, σ̄⁴ = 1)

    #λ_max, p00 = findmax_λ0(p2, ς², p0 =p0)
    λ_max, p00, λ_min, p0min = findmax_λ0(p2, ς², p0 =p0)
    #λ_max, p00 = findmax_λ(C, p0 =p0)
    λ_all =  λ_min:(λ_max - λ_min) / (len - 1):λ_max

    pt = 0.01
    λ_max01, p01 = findmax_λ(p, ς², p0 =pt)
    λ_max02, p02 = findmax_λ(p2, ς², p0 =pt)
    λ_max03, p03 = findmax_λ(p3, ς², p0 =pt)

    p_L = gaussian_variation_model_fast(ρ*f̂(π/L,p) , ς², p, 1, 1)
    GVMF(λ_all) = gaussian_variation_model_fast(λ_all, ς², p, 1, 1)
    #p_all = GVMF.(λ_all)
    #dlnqdlnρ_all = dlnqdlnρ.(λ_all,p=p3,ς²=ς², length_range = length_range)
    dlnqdlnρ_all = dlnqdlnρ2.(λ_all,p=p,p2=p2,ς²=ς²,length_range=length_range,λ_max01=λ_max01,λ_max02=λ_max02,p01=p01,p02=p02)
    #dlnqdlnρ_all = dlnqdlnρ3.(λ_all,p=p,p2=p2,p3=p3,ς²=ς²,length_range=length_range,λ_max01=λ_max01,λ_max02=λ_max02,λ_max03=λ_max03,p01=p01,p02=p02,p03=p03)

    A = hcat(λ_all,dlnqdlnρ_all)
    itp = Interpolations.scale(interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), λ_all, 1:2)

    tfine = λ_min:dλ:λ_max
    λ2, dlnqdlnρ_all2 = [itp(t,1) for t in tfine], [itp(t,2) for t in tfine]

    #index = mean((abs.(dlnqdlnρ_all2))./λ2) *(λ_max-λ_min)/abs(log.(p00)-log.(quantile_variational(λ_min,p2,ς²=ς², λ_max=λ_max02, p0=p02)))
    index = mean((abs.(dlnqdlnρ_all2))./λ2) *(λ_max-λ_min)/abs(log.(quantile_variational(λ_max,p2,ς²=ς², λ_max=λ_max02, p0=p02))-log.(quantile_variational(λ_min,p2,ς²=ς², λ_max=λ_max02, p0=p02)))
    #index = mean((abs.(dlnqdlnρ_all2))./λ2) *(λ_max-λ_min)/abs(log.(p03)-log.(quantile_variational(λ_min,p3,ς²=ς², λ_max=λ_max03, p0=p03)))
    #index = mean((abs.(dlnqdlnρ_all2))./λ2) *(λ_max-λ_min)/abs(log.(p00)-log.(p0min))
    return index
end


function arrow_color(X01, k01, X_pjec; color_map = :PRGn, num = 1000)
    X_all = zeros(2,num)
    for i = 1:num
        X_all[1,i] = X01[1] + k01*X_pjec[1,1] *(i-1)/(num-1)
        X_all[2,i] = X01[2] + k01*X_pjec[2,1] *(i-1)/(num-1)
    end
    for i = 1:num-50
        plot!([X_all[1,i],X_all[1,i+1]],[X_all[2,i],X_all[2,i+1]],arrow=false,color=color_map[i],linewidth=4,label="",legend=:topleft)
    end
    plot!([X_all[1,num-1],X_all[1,num]],[X_all[2,num-1],X_all[2,num]],arrow=Plots.arrow(:closed, 1.0),color=color_map[num],linewidth=1,label="",legend=:topleft)
    #return plot!([X_all[1,num-1],X_all[1,num]],[X_all[2,num-1],X_all[2,num]],arrow=true,color_map=color_map[num],linewidth=2,label="",legend=:topleft)
    return plot!()
end

#=
function arrow_color(X01, k01; color_map = :PRGn,  num = 100)
    X_all = zeros(num-1,2)
    Y_all = zeros(num-1,2)
    for i = 1:num-1
        X_all[i,1] = X01[1] + k01*X_pjec[1,1] *(i-1)/(num-1)
        X_all[i,2] = X01[1] + k01*X_pjec[1,1] *(i)/(num-1)
        Y_all[i,1] = X01[2] + k01*X_pjec[2,1] *(i-1)/(num-1)
        Y_all[i,2] = X01[2] + k01*X_pjec[2,1] *(i)/(num-1)
    end
    plot!(X_all,Y_all,line_z=(1:num-1)/num',arrow=false,color=color_map,linewidth=2,label="",legend=:topleft)#, colorbar=false)
    #plot(X_all,Y_all,line_z=(1:num-1)/100',arrow=false,color=palette(cc),linewidth=2,label="",legend=:topleft, colorbar=false)
    return plot!(X_all[end,:],Y_all[end,:],line_z=1,arrow=true,color=color_map,linewidth=2,label="",legend=:topleft)#, colorbar=false)
end
=#


function palette_colormap(color_map; N_color =1000)
    p_c = palette(color_map)
    l = size(p_c)[1]
    c_mp = colormap("RdBu", N_color)
    for i = 1:N_color
        j0 = i/N_color*(l-1)+1
        j = floor(Int,i/N_color*(l-1)+1-1/N_color/10)
        c_mp[i] = (j+1-j0) * p_c[j] + (j0-j) * p_c[j+1]
    end
    return c_mp
end


function plot_scale_fig6(;dpi=300, fsize = 6,yscale = 300)
    plot!([340,390],[yscale,yscale],color = :black,grid = :none,label="",legend=:none, dpi = dpi)
    #plot!([340,340],[235,260],color = :black,grid = :none,label="",legend=:none, dpi = dpi)
    annotate!(365,yscale-13, Plots.text("100 μm",fsize), dpi = dpi) 
    #annotate!(315,247.5, Plots.text(L"50\mu m",10), dpi = dpi)
end