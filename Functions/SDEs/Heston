## Author: Patrick Chang
# Function to simulate a 2 dimensional Heston model
# There are two types of Heston models:
#   1. Only volatility follows an SDE
#   2. Vol and co-vol is a matrix valued affine process

using Distributions, Random, LinearAlgebra

#---------------------------------------------------------------------------

# Heston_CT simulates a Heston model using the specification of the Bates model by
# Cuchiero and Teichmann 2015 without the jumps. Here the vol and co-vol are all SDEs

function Heston_CT(n, ρ = [-0.3; -0.5], M = [-1.6 -0.2; -0.4 -1], α = [0.0725 0.06; 0.06 0.1325]; kwargs...)
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

    local dt::Float64
    if haskey(kwargs, :dt)
        dt = kwargs[:dt]
    else
        dt = 21600
    end

    X = zeros(ComplexF64, 2, 2, n)
    X[:,:,1] = [0.09 -0.036; -0.036 0.09]
    Y = zeros(ComplexF64, n, 2)
    Y[1,:] = log.([SP SP])

    Σ = sqrt(α)
    b = 3.5*α

    Random.seed!(seed)
    W = randn(2, n)
    Random.seed!(2*seed)
    B = randn(2, 2, n)

    for i in 2:n
        X[:,:,i] = X[:,:,i-1] + (b + M*X[:,:,i-1] + X[:,:,i-1]*M')/dt + sqrt(X[:,:,i-1])*B[:,:,i]*Σ/sqrt(dt) + Σ*B[:,:,i]'*sqrt(X[:,:,i-1])/sqrt(dt)
    end

    for i in 2:n
        Z = sqrt(1-ρ'ρ)*W[:,i] + B[:,:,i]*ρ
        Y[i,:] = Y[i-1,:] - 0.5*diag(X[:,:,i-1])/dt + sqrt(X[:,:,i-1])*Z/sqrt(dt)
    end

    σ_1 = zeros(ComplexF64, n, 1)
    σ_2 = zeros(ComplexF64, n, 1)
    σ_12 = zeros(ComplexF64, n, 1)

    for i in 1:n
        σ_1[i] = X[:,:,i][1,1]
        σ_2[i] = X[:,:,i][2,2]
        σ_12[i] = X[:,:,i][1,2]
    end

    return real(exp.(Y)), real(σ_1), real(σ_2), real(σ_12)
end

# Heston simulates a Heston model using the specification by
# Kalogeropoulos et al. 2018. Here only the volatility are SDEs

function Heston(n, μ = [0.15; 0.18; 4.6; 4.6], κ = [1; 1], ρ = [1 0.1 0.2 0.3; 0.1 1 0.4 0.5; 0.2 0.4 1 0.35; 0.3 0.5 0.35 1]; kwargs...)
    kwargs = Dict(kwargs)

    if haskey(kwargs, :SP)
        SP = kwargs[:SP]
    else
        SP = 100
    end

    if haskey(kwargs, :SV)
        SV = kwargs[:SV]
    else
        SV = 0.2
    end

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else
        seed = 1
    end

    local dt::Float64
    if haskey(kwargs, :dt)
        dt = kwargs[:dt]
    else
        dt = 21600
    end

    C = cholesky(ρ).L

    X = zeros(4, n)
    X[1:2,:1] .= SV
    X[3:4,:1] .= log(SP)

    Random.seed!(seed)
    B = randn(4,n)

    for i in 2:n
        V = X[1:2, i-1]
        b = zeros(4,1)
        b[1] = κ[1]*(μ[1] - V[1])
        b[2] = κ[2]*(μ[2] - V[2])
        b[3] = μ[3] - 0.5*V[1]^2
        b[4] = μ[4] - 0.5*V[2]^2
        # F = diagm(sqrt.([V;V]))
        F = diagm(([V;V]))

        X[:,i] = X[:,i-1] + b/dt + F*C*B[:,i]/sqrt(dt)
    end

    P = exp.(X[3:4,:])'
    V = X[1:2,:]'
    σ_12 = X[1,:].*X[2,:]*ρ[3,4]

    return P, V[:,1].^2, V[:,2].^2, σ_12
end
