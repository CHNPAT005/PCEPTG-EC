## Author: Patrick Chang
# Script file for the MLA by G. BUCCHERI, F. CORSI, AND S. PELUSO (2020)
# but with Ψ = Id which reduces to the KEM of F. CORSI, S. PELUSO AND F. AUDRINO (2015)
# Supporting Algorithms are at the start of the script

# Note that the functions here are replicating the Matlab code
# provided by BUCCHERI ET AL. in Julia

#---------------------------------------------------------------------------

using LinearAlgebra

#---------------------------------------------------------------------------
### Supporting functions
# Kalman Filter and smoother for the dynamic linear model
function KF(y, Q, R)
    # Preliminaries and variable creation
    (d, T) = size(y)
    a = .!isnan.(y)
    Id = Matrix{Float64}(I, d, d)
    x_pred = zeros(d,T); x_filt = zeros(d,T); x_smooth = zeros(d,T)
    V_filt = zeros(d,d,T); V_pred = zeros(d,d,T); V_smooth = zeros(d,d,T)
    Vt_smooth = zeros(d,d,T); J = zeros(d,d,T)
    loglik = zeros(1,T)

    y[a.==0] .= 0

    # Diffuse initialization (see Durbin & Koopman, 2001)
    x_pred[:,1] = zeros(d,1)
    V_pred[:,:,1] = 10^3 .* Id

    # First step update
    D = [Id[a[:,1] .== 1,:]; zeros(sum(a[:,1] .== 0),d)]

    auxY = D*y[:,1]
    auxC = D
    auxR = D*R*D'
    auxR[findall(!iszero, diagm(iszero.(diag(auxR))))] .= 1

    F = auxC*V_pred[:,:,1]*auxC' + auxR
    K = V_pred[:,:,1]*auxC'/F

    v = auxY - auxC*x_pred[:,1]
    x_filt[:,1] = x_pred[:,1] + K*v
    V_filt[:,:,1] = V_pred[:,:,1] - K*auxC*V_pred[:,:,1]

    loglik[1] = -0.5*(log(abs(det(F))) + v'/F*v)

    # Loop through data and update
    for t in 2:T
        D = [Id[a[:,t] .== 1,:]; zeros(sum(a[:,t] .== 0),d)]

        auxY = D*y[:,t]
        auxC = D
        auxR = D*R*D'
        auxR[findall(!iszero, diagm(iszero.(diag(auxR))))] .= 1

        x_pred[:,t] = x_filt[:,t-1]
        V_pred[:,:,t] = V_filt[:,:,t-1] + Q

        F = auxC*V_pred[:,:,t]*auxC' + auxR
        K = V_pred[:,:,t]*auxC'/F

        v = auxY - auxC*x_pred[:,t]
        x_filt[:,t] = x_pred[:,t] + K*v
        V_filt[:,:,t] = V_pred[:,:,t] - K*auxC*V_pred[:,:,t]

        loglik[t] = -0.5*(log(abs(det(F))) + v'/F*v)
    end
    # Smoothing initialization
    x_smooth[:,T] = x_filt[:,T]
    V_smooth[:,:,T] = V_filt[:,:,T]
    Vt_smooth[:,:,T] = (Id-K*auxC)*V_filt[:,:,T-1]
    J[:,:,T-1] = V_filt[:,:,T-1]/V_pred[:,:,T]
    x_smooth[:,T-1] = x_filt[:,T-1] + J[:,:,T-1]*(x_smooth[:,T] - x_filt[:,T-1])
    V_smooth[:,:,T-1] = V_filt[:,:,T-1] + J[:,:,T-1]*(V_smooth[:,:,T] - V_pred[:,:,T])*J[:,:,T-1]'

    # Rauch recursions
    for t in T-1:-1:2
        J[:,:,t-1] = V_filt[:,:,t-1]/V_pred[:,:,t]
        x_smooth[:,t-1] = x_filt[:,t-1] + J[:,:,t-1]*(x_smooth[:,t] - x_filt[:,t-1])
        V_smooth[:,:,t-1] = V_filt[:,:,t-1] + J[:,:,t-1]*(V_smooth[:,:,t]-V_pred[:,:,t])*J[:,:,t-1]'
        Vt_smooth[:,:,t] = V_filt[:,:,t]*J[:,:,t-1]' + J[:,:,t]*(Vt_smooth[:,:,t+1] - V_filt[:,:,t])*J[:,:,t-1]'
    end

    logLik = sum(loglik[d+1:T])

    return x_filt, x_smooth, V_smooth, Vt_smooth, logLik
end

# EM algorithm for estimating Q from CORSI ET AL. (2015)
function EM(y, Q_init, R_init, maxiter, eps)
    # Preliminaries
    (d, T) = size(y)
    a = .!isnan.(y)
    check_conv = 0
    Id = Matrix{Float64}(I, d, d)
    burnIn = d+10

    # Variable creation
    l = fill(NaN, maxiter); deltalog = fill(NaN, maxiter)

    # Initialization
    l_old = -10e10
    Q = Q_init; R = R_init

    # Perform the EM algorithm
    for i in 1:maxiter
        # Kalman filtering and smoothing
        y[a.==0] .= NaN
        (x_filt, x_smooth, V_smooth, Vt_smooth, logLik) = KF(y, Q, R)
        y[a.==0] .= 0

        # Compute incomplete data log-likelihood
        l[i] = logLik

        # E step
        S = zeros(d, d)
        S10 = zeros(d, d); eps_smooth = zeros(d, d)

        for t in burnIn:T
            Dobs = [Id[a[:,t] .== 1,:]; zeros(sum(a[:,t] .== 0),d)]
            Dback = [Id[a[:,t] .== 1,:]; Id[a[:,t] .== 0,:]]'
            Dmiss = [zeros(sum(a[:,t] .!= 0),d); Id[a[:,t] .== 0,:]]

            auxY = Dobs*y[:,t]
            auxC = Dobs

            S = S + x_smooth[:,t]*x_smooth[:,t]' + V_smooth[:,:,t]
            S10 = S10 + x_smooth[:,t]*x_smooth[:,t-1]' + Vt_smooth[:,:,t]

            eps_smooth = eps_smooth + Dback*((auxY-auxC*x_smooth[:,t])*(auxY-auxC*x_smooth[:,t])' + auxC*V_smooth[:,:,t]*auxC' + Dmiss*R*Dmiss' )*Dback'
        end

        S00 = S - x_smooth[:,T]*x_smooth[:,T]'-V_smooth[:,:,T] + x_smooth[:,burnIn-1]*x_smooth[:,burnIn-1]' + V_smooth[:,:,burnIn-1]
        S11 = S

        # M step
        BB = S10; AA = S00; CC = S11

        R = diagm(diag(eps_smooth/(T-burnIn+1)))
        Q = (CC-BB-BB'+AA)/(T-burnIn+1)

        # Break if converged
        deltalog[i] = abs(l[i] - l_old)/abs(l_old)
        if deltalog[i] < eps
            check_conv = 1
            break
        end
        l_old = l[i]
    end
    l = l[isnan.(l) .== 0]

    return Q, R, check_conv, l
end

# KEM estimate from CORSI ET AL. (2015)
function KEM(P, Q_init, R_init, maxiter, eps)
    y = log.(P)'; d = size(y, 1); Id = Matrix{Float64}(I, d, d)

    Q0 = Q_init
    R0 = R_init

    (auxQKEM, auxRKEM, cConv, logLPath) = EM(y, Q0, R0, maxiter, eps)
    Q = auxQKEM
    R = auxRKEM
    Σ = Q
    ρ = Σ[1,2] / (sqrt(Σ[1,1]) * sqrt(Σ[2,2]))
    return ρ
end

## Testing

# include("../../Hawkes/Hawkes.jl")
# include("../../SDEs/GBM.jl")
#
# function theoreticalCorr(α_12, α_13, β)
#     Γ_12 = α_12 / β; Γ_13 = α_13 / β
#     num = 2*Γ_13*(1+Γ_12)
#     den = 1 + Γ_13^2 + 2*Γ_12 + Γ_12^2
#
#     return num/den
# end
#
# function theoreticalEpps(τ, μ, α_12, α_13, β)
#     Γ_12 = α_12 / β; Γ_13 = α_13 / β
#     Λ = μ / (1 - Γ_12 - Γ_13)
#     Q_1 = -(μ * (Γ_12^2 + Γ_12 - Γ_13^2)) / (((Γ_12 + 1)^2 - Γ_13^2) * (1 - Γ_12 - Γ_13))
#     Q_2 = -(μ * Γ_13) / (((Γ_12 + 1)^2 - Γ_13^2) * (1 - Γ_12 - Γ_13))
#     R = (β * μ) / (Γ_12 + Γ_13 - 1)
#     G_1 = β * (1 + Γ_12 + Γ_13)
#     G_2 = β * (1 + Γ_12 - Γ_13)
#     C_1 = (2 + Γ_12 + Γ_13) * (Γ_12 + Γ_13) / (1 + Γ_12 + Γ_13)
#     C_2 = (2 + Γ_12 - Γ_13) * (Γ_12 - Γ_13) / (1 + Γ_12 - Γ_13)
#
#     C_11 = Λ + (R*C_1)/(2*G_1) + (R*C_2)/(2*G_2) + R * (C_2*G_1^2*exp(-τ*G_2) - C_1*G_2^2 + Q_1*G_2^2*exp(-τ*G_1) -C_2*G_1^2) / (2*G_2^2*G_1^2*τ)
#     C_12 = - (R*C_1)/(2*G_1) + (R*C_2)/(2*G_2) + R * (C_1*G_2^2 - C_2*G_1^2 - C_1*G_2^2*exp(-τ*G_1) + C_2*G_1^2*exp(-τ*G_2)) / (2*G_2^2*G_1^2*τ)
#
#     return C_12/C_11
# end
#
# T = Int(3600*6.5)
# ρ = 0.35
# ρ = theoreticalCorr(0.023, 0.05, 0.11)
#
# mu = [0.01/86400, 0.01/86400]
# sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
#         sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]
#
# P_GBM = GBM(T+1, mu, sigma)
#
# par_1 = BarcyParams(0.015, 0.023, 0.05, 0.11)
# lambda0_1 = par_1[1]; alpha_1 = par_1[2]; beta_1 = par_1[3]
# t1 = simulateHawkes(lambda0_1, alpha_1, beta_1, T, seed = 19549293)
# p1_1 = getuniformPrices(0, 1, T, t1[1], t1[2])
# p2_1 = getuniformPrices(0, 1, T, t1[3], t1[4])
# P_Hawkes = [p1_1 p2_1]
#
# test = KEM(P_GBM, sigma*T, sigma*T, 300, 1e-5)
# test2 = KEM(exp.(P_Hawkes), sigma*T, sigma*T, 300, 1e-5)
