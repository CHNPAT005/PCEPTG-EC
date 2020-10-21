## Author: Patrick Chang
# Test Script to test the multivariate Hawkes functions

#---------------------------------------------------------------------------

using Plots, LaTeXStrings
include("../Functions/Hawkes/Hawkes.jl")

cd("/Users/patrickchang1/PCEPTG-MSC")

#---------------------------------------------------------------------------
## Testing the Functions

lambda0 = [0.015;0.015]
alpha = [0 0.023; 0.023 0]
beta = [0 0.11; 0.11 0]
T = 1000
## Simulation
# Intensity and Count process plots
tt = collect(0:1:T)
t = simulateHawkes(lambda0, alpha, beta, T)
λ1 = Intensity(1, tt, t, lambda0, alpha, beta)
λ2 = Intensity(2, tt, t, lambda0, alpha, beta)

p1 = plot(tt, λ1, xlims = (minimum([t[1]; t[2]]), maximum([t[1]; t[2]])), label = "")
xlabel!(p1, L"\textrm{Time [sec]}")
ylabel!(p1, L"\lambda_1 (t)")

p2 = plot(t[1], cumsum(repeat([1], length(t[1]))), linetype = :steppre, legend = :bottomright, xlims = (minimum([t[1]; t[2]]), maximum([t[1]; t[2]])), label = "")
xlabel!(p2, L"\textrm{Time [sec]}")
ylabel!(p2, L"N_1(t)")

p3 = plot(tt, λ2, xlims = (minimum([t[1]; t[2]]), maximum([t[1]; t[2]])), label = "")
xlabel!(p3, L"\textrm{Time [sec]}")
ylabel!(p3, L"\lambda_2 (t)")
p4 = plot(t[2], cumsum(repeat([1], length(t[2]))), linetype = :steppre, legend = :bottomright, xlims = (minimum([t[1]; t[2]]), maximum([t[1]; t[2]])), label = "")
xlabel!(p4, L"\textrm{Time [sec]}")
ylabel!(p4, L"N_2(t)")

p5 = plot(p1, p3, p2, p4, layout = (2,2))
# savefig(p5, "Plots/Hawkes/CountandIntensity.svg")

## Extracting the prices
T = 3600*42
t = simulateHawkes(lambda0, alpha, beta, T)
P = getPrices(100, t[1], t[2])

p6 = plot(P[2] .* (42/T), P[1], linetype = :steppost)
title!(p6, "Price Process")
xlabel!(p6, "Time (h)")
ylabel!(p6, L"P(t)")

ind = findall(P[2] .<= 3600*2)
p7 = plot(P[2][ind] .* (42/T), P[1][ind], linetype = :steppost, legend = :bottomright)
title!(p7, "Price Process")
xlabel!(p7, "Time (h)")
ylabel!(p7, L"P(t)")

p8 = plot(p6, p7, layout = (2,1))
# savefig(p8, "Plots/Hawkes/HawkesPrice.pdf")

## Calibration
T = 3600*18     # large enough to get good estimates, but not too long that it'll run for too long
t = simulateHawkes(lambda0, alpha, beta, T)

loglikeHawkes(t, lambda0, alpha, beta, T)

res = optimize(calibrateHawkes, [0.01, 0.015, 0.15])
par = Optim.minimizer(res)
    # par is very close to that used in simulation


## Loglikelihood surface
function loglike(param)
    lambda0 = [param[1] param[1]]
    alpha = [0 param[2]; param[2] 0]
    beta = [0 param[3]; param[3] 0]
    return loglikeHawkes(t, lambda0, alpha, beta, T)
end

alphaGrid = collect(0.022:0.0001:0.024)
betaGrid = collect(0.1:0.001:0.12)
lambdaGrid = collect(0.014:0.0001:0.016)

likeab = [loglike([0.015, a, l]) for a in alphaGrid, l in betaGrid]
likealam = [loglike([l, a, 0.11]) for a in alphaGrid, l in lambdaGrid]
likeblam = [loglike([l, 0.023, a]) for a in betaGrid, l in lambdaGrid]

my_cgrad = cgrad([:red, :yellow, :blue])

p1 = surface(betaGrid,alphaGrid,likeab, xlabel = L"\beta", ylabel = L"\alpha", zlabel = L"\ell(\theta)", fc=:heat)

# savefig(p1, "Plots/Hawkes/likeab.svg")

p2 = surface(lambdaGrid,alphaGrid,likealam, xlabel = L"\lambda", ylabel = L"\alpha", zlabel = L"\ell(\theta)", fc=:heat)

# savefig(p2, "Plots/Hawkes/likealam.svg")

p3 = surface(lambdaGrid,betaGrid,likeblam, xlabel = L"\lambda", ylabel = L"\beta", zlabel = L"\ell(\theta)", fc=:heat)

# savefig(p3, "Plots/Hawkes/likeblam.svg")
