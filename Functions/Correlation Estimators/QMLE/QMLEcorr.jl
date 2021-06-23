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
    # Create the diagonal and superdiagonal of Ω
    dv = repeat([σ2*Δ + 2*a2], n)
    ev = repeat([-a2], n-1)
    # Initialise Ω
    Ω = SymTridiagonal(dv, ev)
    # Compute log-likelihood
    llike = -0.5*logdet(Ω) - n/2 * log(2*pi) - 0.5 * (Y'*inv(Ω)*Y)[1]
    return -llike
end

function QMLEvar(X, Δ, starting)
    # Optimise the log-likelihood
    opt = optimize(vars -> log_likelihood(X, Δ, vars), starting, NelderMead())
    # Extract parameters, dont forget to exponentiate!
    param = Optim.minimizer(opt)
    return exp.(param)[1]
end

#---------------------------------------------------------------------------

function QMLEcorr(p1, p2, Δ, starting)
    # Get the variances
    V1 = QMLEvar(log.(p1), Δ, starting); V2 = QMLEvar(log.(p2), Δ, starting)
    Covar = 0.25 * (QMLEvar(log.(p1) .+ log.(p2), Δ, starting) - QMLEvar(log.(p1) .- log.(p2), Δ, starting))
    ρ = Covar / (sqrt(V1) * sqrt(V2))
    return ρ
end


## Testing

# T = Int(3600*6.5)
# ρ = 0.35
#
# mu = [0.01/86400, 0.01/86400]
# sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
#         sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]
#
# P_GBM = GBM(T+1, mu, sigma)
#
#
# p1 = P_GBM[:,1]; p2 = P_GBM[:,2];  Δ = 1
# X = log.(p1)
# log_param = [log(0.1/86400); log(0.1/86400)]
#
# @elapsed log_likelihood(X, Δ, log_param)
# @elapsed testvar = QMLEvar(X, Δ, log_param)
# @elapsed testcor = QMLEcorr(P_GBM[:,1], P_GBM[:,2], Δ, log_param)
