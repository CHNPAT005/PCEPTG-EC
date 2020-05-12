## Author: Patrick Chang
# Test Script to test the univariate Hawkes functions

#---------------------------------------------------------------------------

using Plots, LaTeXStrings, Optim
include("../Functions/Hawkes/Hawkes1D")

#---------------------------------------------------------------------------
## Testing the functions

# Simulation
lambda0 = 0.2
alpha = 0.5
beta = 0.7
T = 100 #Seconds

t = simulateHawkes1D(lambda0, alpha, beta, T)
λ = Intensity1D(t, t, lambda0, alpha, beta)

p1 = plot(t, λ)
title!(p1, "Intensity")
xlabel!(p1, "Time")
ylabel!(p1, L"\lambda (t)")
p2 = plot(t, cumsum(repeat([1], length(t))), linetype = :steppre, legend = :bottomright)
title!(p2, "Count Process")
xlabel!(p2, "Time")
ylabel!(p2, L"N(t)")

p3 = plot(p1, p2, layout = (2,1))
# savefig(p3, "Plots/1DHawkes.pdf")

# Calibration
T = 3600*48     # must be large enough to ensure we have enough data for accurate estimation
t = simulateHawkes1D(lambda0, alpha, beta, T)
res = optimize(calibrateHawkes1D, [0.15, 0.3, 0.9])
par = Optim.minimizer(res)
    # par should return parameters very close to the ones used in the simulation
