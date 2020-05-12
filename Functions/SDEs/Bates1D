## Author: Patrick Chang
# Function to simulate a 1 dimensional Bates model
# using the first order Euler discretization

using Distributions, Random, LinearAlgebra

#---------------------------------------------------------------------------
# ρ = the correlation between the volatility process and the price process
# λ = the rate of jumps on the volatility and price process
# vp = variance of jump size of price process
# vj = variace of jump size of volatility process

# starting price is at 100 and volatility is at 0.04

# parameters are preset using the choices from Mancino, Recchioni and Sanfelici 2017
function Bates1D(n, ρ=-0.5, λ=20/250, vp=0.2, vj=0.01, α=0.5; kwargs...)
    kwargs = Dict(kwargs)

    if haskey(kwargs, :SP)
        SP = kwargs[:SP]
    else
        SP = 100
    end

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else
        seed = 1
    end

    if haskey(kwargs, :SV)
        SV = kwargs[:SV]
    else
        SV = sqrt(0.04)
    end

    local dt::Float64
    if haskey(kwargs, :dt)
        dt = kwargs[:dt]
    else
        dt = 21600
    end

    Random.seed!(seed)
    Z = randn(2, n-1)

    Random.seed!(seed+1)
    Z2 = randn(2, n-1)

    sig = [1 ρ; ρ 1]
    A = cholesky(sig).L
    Z = A*Z

    Sigma2 = fill(0.0, (1,n))
    Sigma2[1] = log(SV)

    Random.seed!(seed)
    N = rand(Poisson(λ/dt), n-1)

    for i in 2:n
        M = vj*sqrt.(N[i-1]/dt)*Z2[2, i-1]
        d = -(α^2/2 + (exp(vj^2/2) -1)*λ)/dt
        Sigma2[i] = Sigma2[i-1] + d + α*Z[2, i-1]*sqrt(1/dt) + M
    end

    Sigma2 = exp.(Sigma2)

    P = zeros(n, 1)
    P[1] = log(SP)

    for i in 2:n
        M = vp*sqrt.(N[i-1]/dt)*Z2[1, i-1]
        d = -(Sigma2[i]^2/2 - (exp(vp^2/2) -1)*λ)/dt
        P[i] = P[i-1] + d + Sigma2[i]*Z[1, i-1]*sqrt(1/dt) + M
    end

    return exp.(P), Sigma2'.^2
end

function Bates1D(n, ρ = -0.5, M = -1.6, Σ = 0.25, λ_Y = 100, λ_X = 10, μ = -0.005, ν = 0.015, θ = 0.05; kwargs...)
    kwargs = Dict(kwargs)

    if haskey(kwargs, :SP)
        SP = kwargs[:SP]
    else
        SP = 100
    end

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else
        seed = 1
    end

    if haskey(kwargs, :SV)
        SV = kwargs[:SV]
    else
        SV = sqrt(0.04)
    end

    local dt::Float64
    if haskey(kwargs, :dt)
        dt = kwargs[:dt]
    else
        dt = 21600
    end

    X_minus = zeros(n,1)
    X_minus[1] = 0.09
    X_plus = zeros(n,1)
    X_plus[1] = 0.09
    Y = zeros(n,1)
    Y[1] = log(SP)

    α = Σ^2
    b = 3.5*α

    Random.seed!(seed)
    B = randn(2, n)
    A = [1 0; ρ sqrt(1-ρ^2)]
    Z = A*B

    Random.seed!(2*seed)
    Z_jump = randn(n)

    Random.seed!(seed)
    N_X = rand(Poisson(λ_X/dt), n)
    Random.seed!(2*seed)
    N_Y = rand(Poisson(λ_Y/dt), n)

    for i in 2:n
        X_minus[i] = X_plus[i-1] + (b + 2*M*X_plus[i-1])/dt + 2*sqrt(X_plus[i-1])*Z[1,i]*Σ/sqrt(dt)
        if N_X[i] == 0
            X_plus[i] = X_minus[i]
        else
            X_plus[i] = X_minus[i] + sum(rand(Exponential(θ), N_X[i]))
        end
    end

    for i in 2:n
        bs = -0.5*X_plus[i] - λ_Y*(exp(μ - 0.5*ν^2) - 1)
        M = μ*N_Y[i] + ν*sqrt(N_Y[i])*Z_jump[i]
        Y[i] = Y[i-1] + bs/dt + sqrt(X_minus[i])*Z[2,i]/sqrt(dt) + M
    end

    return exp.(Y), X_plus
end

 - λ_X*(exp(θ) - 1)

 # function Bates1D(n, ρ = -0.5, μ_σ = 0.15, μ_p = 4.6, κ = 1, λ_Y = 100, λ_X = 10, μ = -0.005, ν = 0.015, θ = 0.05; kwargs...)
 #     kwargs = Dict(kwargs)
 #
 #     if haskey(kwargs, :SP)
 #         SP = kwargs[:SP]
 #     else
 #         SP = 100
 #     end
 #
 #     if haskey(kwargs, :seed)
 #         seed = kwargs[:seed]
 #     else
 #         seed = 1
 #     end
 #
 #     if haskey(kwargs, :SV)
 #         SV = kwargs[:SV]
 #     else
 #         SV = 0.2
 #     end
 #
 #     local dt::Float64
 #     if haskey(kwargs, :dt)
 #         dt = kwargs[:dt]
 #     else
 #         dt = 21600
 #     end
 #
 #     V = zeros(n,1)
 #     V[1] = SV
 #     Y = zeros(n,1)
 #     Y[1] = log(SP)
 #
 #     Random.seed!(seed)
 #     B = randn(2, n)
 #     A = [1 0; ρ sqrt(1-ρ^2)]
 #     Z = A*B
 #
 #     Random.seed!(2*seed)
 #     Z_jump = randn(n)
 #
 #     Random.seed!(seed)
 #     N_X = rand(Poisson(λ_X/dt), n)
 #     Random.seed!(2*seed)
 #     N_Y = rand(Poisson(λ_Y/dt), n)
 #
 #     for i in 2:n
 #         V[i] = V[i-1] + κ*(μ_σ - V[i-1])/dt + V[i-1]*Z[1,i]/sqrt(dt)
 #         if N_X[i] == 0
 #             V[i] = V[i]
 #         else
 #             V[i] = V[i] + sqrt(sum(rand(Exponential(θ), N_X[i])))
 #         end
 #     end
 #
 #     for i in 2:n
 #         bs = -0.5*V[i-1]^2 - λ_Y*(exp(μ - 0.5*ν^2) - 1)
 #         M = μ*N_Y[i] + ν*sqrt(N_Y[i])*Z_jump[i]
 #         Y[i] = Y[i-1] + bs/dt + V[i-1]*Z[2,i]/sqrt(dt) + M
 #     end
 #
 #     return exp.(Y), V.^2
 # end
