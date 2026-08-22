function f̂₀_num(i,p;flag_k=true)
    @unpack L,ϵ,μ,n = p
    c = ϵ^μ
    # c=1
    # k=i
    
    if n == 1
        if flag_k == true # flag_k = true: pdf; flag_k = false: rank plot
            k = i
        else
            k = (i)/2 * (2π/L)
        end
        f_Pth = k -> π^(μ-n/2)*gamma((n-μ)/2)/gamma(μ/2)/abs(k)^(n-μ)/(2π)^(μ-n)/2
        # k = i
        f_int = (r,k) -> cos(r.*k)*f_r(r,p)
        f_int1 = (r,k) -> cos(r.*k)*c*r^(-μ)
        # f̂_numeric = k -> 2*(quadgk(r -> f_int(r,k),0,1,rtol=1e-9)[1] + 
        #     c*sin(π/2*μ)*gamma(1-μ)./abs(k)^(1-μ) - quadgk(r -> f_int1(r,k),0,1,rtol=1e-9)[1])
        f̂_numeric = k -> 2*(quadgk(r -> f_int(r,k),0,L,rtol=1e-9)[1] + 
            c*f_Pth(k) - quadgk(r -> f_int1(r,k),0,L,rtol=1e-9)[1])
    elseif n == 2
        if flag_k == true
            k = i
        else          
            k = sqrt.((i)/π) * (2π/L)
        end
        f_Pth = k -> π^(μ-n/2)*gamma((n-μ)/2)/gamma(μ/2)/abs(k)^(n-μ)/(2π)^(μ-n)/(2π)
        f_int = (r,k) -> r*f_r(r,p)*besselj0(r.*k)
        f_int1 = (r,k) -> r*c*r^(-μ)*besselj0(r.*k)
        f̂_numeric = k -> 2π*(quadgk(r -> f_int(r,k),0,L,rtol=1e-9)[1] +
            c*f_Pth(k) - quadgk(t -> f_int1(t,k),0,L, rtol=1e-9)[1]) 
    end
        
    return f̂_numeric(k)
end

function f̂₀_th(i, p::ERMParameter; flag_k = true)
    @unpack μ,n,ϵ = p
    if n == 1
        if flag_k == true # flag_k = true: pdf; flag_k = false: rank plot
            k = i
        else
            k = (i)/2 * (2π/L)
        end
        y = sqrt(π)*2^(3/2-μ/2)*1/ϵ*ϵ^μ*(ϵ^2)^(3/4-μ/4)*abs(k)^((μ-1)/2)*besselk((μ-1)/2,k*ϵ*sign(k))/gamma(μ/2)
    elseif n == 2
        if flag_k == true # flag_k = true: pdf; flag_k = false: rank plot
            k = i
        else
            k = sqrt.((i)/π) * (2π/L)
        end
        y = 2^(1-μ/2)/gamma(μ/2)*k^(μ/2-1)*π^2*(1/ϵ^2)^(1/2-μ/4)*ϵ^μ*(ϵ^2)^(1-μ/2)*(-besseli(1-μ/2, k*ϵ)+besseli(-1+μ/2,k*ϵ))*csc(π*μ/2)
        # y = 2π*2^(-μ/2)/gamma(1+μ/2)*k^(μ/2-1)*(1/ϵ^2)^(1/2-μ/4)*ϵ^μ*(ϵ^2)^(1-μ/2)*μ*besselk((-1+μ/2),k*ϵ)
    elseif n == 3
        k = i
        y = 2^(5/2-μ/2)*π^(3/2)*ϵ^((3+μ)/2)*abs(k)^(1/2*(-3+μ))*besselk((3-μ)/2,abs(k)*ϵ) / gamma(μ/2)
    end
    return y   
end
