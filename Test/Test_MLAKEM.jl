## Author: Patrick Chang
# Script file to compare and test the MLA and KEM on a synchronous GBM

using Plots, LaTeXStrings

cd("/Users/patrickchang1/PCEPTG-EC")

include("../Functions/SDEs/GBM.jl")
include("../Functions/Correlation Estimators/MLA/MLA.jl")
include("../Functions/Correlation Estimators/MLA/KEM.jl")

# Simulate the GBM
T = Int(3600*6.5)
ρ = 0.35

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
        sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]

P_GBM = GBM(T+1, mu, sigma)

# Compute the KEM and MLE estimates
test_KEM = KEM(P_GBM, sigma*T, sigma*T, 300, 1e-5)
test_MLA = MLA(P_GBM, sigma*T, sigma*T, 300, 1e-5)

p1 = scatter([1], [ρ], label = "True", ylims = (0, 1), legend = :bottomright, color = :black, xticks=false)
scatter!(p1, [2], [test_KEM], label = "KEM", color = :blue)
scatter!(p1, [3], [test_MLA], label = "MLA", color = :red)
hline!(p1, [ρ], label = "", color = :black)
hline!(p1, [test_KEM], label = "", color = :blue)
hline!(p1, [test_MLA], label = "", color = :red)
ylabel!(p1, L"\rho_{}^{ij}")

# savefig(p1, "Plots/Supp/TestMLAKEM.pdf")
