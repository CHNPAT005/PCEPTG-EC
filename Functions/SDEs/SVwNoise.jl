## Author: Patrick Chang
# Script file to simulate a Stochastic Volatility process with the U-shaped
# volatility with the observations also contaminated with noise.
# Based off the model from BUCCHERI ET AL. (2020).

# Note that the functions here are replicating the Matlab code
# provided by BUCCHERI ET AL. in Julia but we remove the lead-lag effects
# i.e. our F = 0_d matrix

# Parameters are all built inside the function. Using the same parameters
# as the Matlab code by BUCCHERI ET AL.

#------------------------------------------------------------------------------

using LinearAlgebra, Random

#------------------------------------------------------------------------------

function SVwNoise(T, ρ; seed = 1, d = 2)
    # Sample the normal
    Random.seed!(seed)
    W = randn(d, T); W1 = randn(2, T)
    # Create storage variables
    ν = zeros(d, T); RX = zeros(d, T); Qt = zeros(d, d, T)
    # Parameter settings for U-shape vol
    CC = 0.88929198; AA = 0.75; BB = 0.25; a = 10; b = 10
    # Parameter settings for CIR
    k = 10*ones(d,1); θ = sqrt.([0.1; 0.2]/10).^2
    s = 0.1*ones(d,1); lev = 0.01*ones(d,1)
    # Parameter for noise of observations
    H = sqrt.([0.1; 0.2]/10).^2 # 1 to 1 signal vs noise ratio
    # Grids
    deltaT = 1/T
    aux1 = collect(1:1:T)/T
    # Generate U-shape
    sigmaU = (CC .+ AA .* exp.(-a .* aux1) .+ BB .* exp.(-b .* (1 .- aux1)))
    # Initialise
    S0 = log(100)*ones(d,1)
    RX[:,1] = zeros(d, 1); ν[:,1] = θ
    aux2 = sigmaU[1] .* sqrt.(ν[:,1])
    rho = [1 ρ; ρ 1]
    # Qt[:,:,1] = diagm(aux2) * rho * diagm(aux2)
    Qt[:,:,1] = [aux2[1]^2 aux2[1]*ρ*aux2[2]; aux2[1]*ρ*aux2[2] aux2[2]^2]
    # Simulate the efficient process
    for t in 2:T
        ν[:,t] = ν[:,t-1] +  k.*(θ - ν[:,t-1]) .* deltaT + s .* sqrt.(ν[:,t-1] .* deltaT) .* W1[:,t-1]
        aux2 = sigmaU[t] .* sqrt.(ν[:,t])
        # Qt[:,:,t] = diagm(aux2) * rho * diagm(aux2)
        Qt[:,:,t] = [aux2[1]^2 aux2[1]*ρ*aux2[2]; aux2[1]*ρ*aux2[2] aux2[2]^2]
        cholQ = cholesky(Qt[:,:,t]).L
        RX[:,t] = cholQ*W[:,t-1]
    end
    state = cumsum([S0 RX[:,2:end]], dims = 2)
    chH = cholesky(diagm(H)).L
    noise = chH*randn(d,T)
    obs = state + noise
    return exp.(obs'), exp.(state')
end

# test = SVwNoise(23400, 0.35)
# P = test[1]
# t = [collect(1:1:23400.0) collect(1:1:23400.0)]
#
# testcor1 = NUFFTcorrDKFGG(P, t)[1][1,2]
# testcor2 = NUFFTcorrDKFGG(P, t)[1][1,2]
# testcor3 = NUFFTcorrDKFGG(P, t)[2]
#
# testcor4 = KEM(P, testcor3, testcor3, 300, 1e-5)
#
#
# rm1 = sample(2:23400, 9360, replace = false)
# rm2 = sample(2:23400, 9360, replace = false)
#
# P[rm1, 1] .= NaN
# t[rm1, 1] .= NaN
# P[rm2, 2] .= NaN
# t[rm2, 2] .= NaN
#
# testcor5 = KEM(P, testcor3, testcor3, 300, 1e-5)
# testcor6 = NUFFTcorrDKFGG(P, t)[1][1,2]
