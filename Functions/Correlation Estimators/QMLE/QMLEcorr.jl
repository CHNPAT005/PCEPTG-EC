## Author: Patrick Chang
# Script file for the QMLE by AÏT-SAHALIA (2010)
# Supporting Algorithms are at the start of the script

#---------------------------------------------------------------------------

using Optim, LinearAlgebra, NLSolversBase

#---------------------------------------------------------------------------
### Supporting functions

function log_likelihood(X, Δ, log_param)
    # Exponentiate to get param
    param = exp.(log_param)
    σ2 = param[1]; a2 = param[2]
    # Get log returns
    Y = diff(X, dims = 1)
    # Get number of obs
    n = length(Y)
    # Initialise Ω
    Ω = zeros(n, n)
    # Populate Ω
    for i in 1:n
        # Fill in the diagonal
        Ω[i,i] = σ2*Δ + 2*a2
    end
    for i in 2:n
        # Fill in the off diagonals
        Ω[i,i-1] = -a2
        Ω[i-1,i] = -a2
    end

    llike = -0.5*logdet(Ω) - n/2 * log(2*pi) - 0.5 * (Y'*inv(Ω)*Y)[1]
    return -llike
end

function QMLEvar(X, Δ)
    # Optimise the log-likelihood
    opt = optimize(vars -> log_likelihood(X, Δ, vars), [log(0.1); log(0.1)], NelderMead())
    # Extract parameters, dont forget to exponentiate!
    param = Optim.minimizer(opt)
    return exp.(param)[1]
end

#---------------------------------------------------------------------------

function QMLEcorr(p1, p2, Δ)
    # Get the variances
    V1 = QMLEvar(log.(p1), Δ); V2 = QMLEvar(log.(p2), Δ)
    Covar = 0.25 * (QMLEvar(log.(p1) .+ log.(p2), Δ) - QMLEvar(log.(p1) .- log.(p2), Δ))
    ρ = Covar / (sqrt(V1) * sqrt(V2))
    return ρ
end


# ## Testing
#
# T = 3600#*20
# ρ = 0.35#theoreticalCorr(0.023, 0.05, 0.11)
#
# mu = [0.01/86400, 0.01/86400]
# sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
#         sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]
#
# P_GBM = GBM(T+1, mu, sigma)
#
#
#
# @elapsed test = QMLEcorr(P_GBM[:,1], P_GBM[:,2], 1)
#
#
# p1 = P_GBM[:,1]; p2 = P_GBM[:,2];  Δ = 1
#
# X = log.(p1)
# P_GBM[:,1]
# test = diff(log.(P_GBM[:,1]), dims = 1)
# # X = GBM(501, [0.01/86400], [0.1/86400])
# # Δ = 1;
# log_param = [log(0.1); log(0.1)]
# #
# # @elapsed log_likelihood(X, Δ, log_param)
# # test'*test
#
#
# log_likelihood(X, Δ, [log(0.1); log(0.1)])
