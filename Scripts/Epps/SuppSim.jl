## Author: Patrick Chang
# Script file to compare the QMLE, MLA and RV estimates

using LinearAlgebra, Plots, LaTeXStrings, StatsBase, Intervals, JLD, ProgressMeter, Distributions

cd("/Users/patrickchang1/PCEPTG-EC")

include("../../Functions/Hawkes/Hawkes.jl")
include("../../Functions/SDEs/GBM.jl")
include("../../Functions/SDEs/SVwNoise.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/QMLE/QMLEcorr.jl")
include("../../Functions/Correlation Estimators/MLA/KEM.jl")

#---------------------------------------------------------------------------
## Supplementary functions
# Theoretical correlations from a Hawkes price model used in Barcy et al.
function theoreticalCorr(α_12, α_13, β)
    Γ_12 = α_12 / β; Γ_13 = α_13 / β
    num = 2*Γ_13*(1+Γ_12)
    den = 1 + Γ_13^2 + 2*Γ_12 + Γ_12^2

    return num/den
end

function theoreticalEpps(τ, μ, α_12, α_13, β)
    Γ_12 = α_12 / β; Γ_13 = α_13 / β
    Λ = μ / (1 - Γ_12 - Γ_13)
    Q_1 = -(μ * (Γ_12^2 + Γ_12 - Γ_13^2)) / (((Γ_12 + 1)^2 - Γ_13^2) * (1 - Γ_12 - Γ_13))
    Q_2 = -(μ * Γ_13) / (((Γ_12 + 1)^2 - Γ_13^2) * (1 - Γ_12 - Γ_13))
    R = (β * μ) / (Γ_12 + Γ_13 - 1)
    G_1 = β * (1 + Γ_12 + Γ_13)
    G_2 = β * (1 + Γ_12 - Γ_13)
    C_1 = (2 + Γ_12 + Γ_13) * (Γ_12 + Γ_13) / (1 + Γ_12 + Γ_13)
    C_2 = (2 + Γ_12 - Γ_13) * (Γ_12 - Γ_13) / (1 + Γ_12 - Γ_13)

    C_11 = Λ + (R*C_1)/(2*G_1) + (R*C_2)/(2*G_2) + R * (C_2*G_1^2*exp(-τ*G_2) - C_1*G_2^2 + Q_1*G_2^2*exp(-τ*G_1) -C_2*G_1^2) / (2*G_2^2*G_1^2*τ)
    C_12 = - (R*C_1)/(2*G_1) + (R*C_2)/(2*G_2) + R * (C_1*G_2^2 - C_2*G_1^2 - C_1*G_2^2*exp(-τ*G_1) + C_2*G_1^2*exp(-τ*G_2)) / (2*G_2^2*G_1^2*τ)

    return C_12/C_11
end

# Simulates a random exponential sample
function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

#---------------------------------------------------------------------------
# Simulation settings
# Settings
reps = 10
T = Int(3600*6.5)
ρ = theoreticalCorr(0.023, 0.05, 0.11)
lam = 15
lam2 = 1/lam
dt = collect(1:1:100)

# GBM sim
mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
        sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]

P_GBM = GBM(T+1, mu, sigma)
t_GBM = reshape([collect(0:1:T); collect(0:1:T)], T+1, 2)

# SV with noise]
P_SV = SVwNoise(T+1, ρ)[1]
t_SV = reshape([collect(0:1:T); collect(0:1:T)], T+1, 2)

# Hawkes sim
par_1 = BarcyParams(0.015, 0.023, 0.05, 0.11)
lambda0_1 = par_1[1]; alpha_1 = par_1[2]; beta_1 = par_1[3]
t1 = simulateHawkes(lambda0_1, alpha_1, beta_1, T, seed = 19549293)

p1_1 = getuniformPrices(0, 1, T, t1[1], t1[2])
p2_1 = getuniformPrices(0, 1, T, t1[3], t1[4])
P_Hawkes = [p1_1 p2_1]

t = collect(0:1:T)

theoretical_exp = ρ .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))
theoretical = zeros(length(dt), 1)
for i in 1:length(dt)
    theoretical[i] = theoreticalEpps(dt[i], 0.015, 0.023, 0.05, 0.11)
end
#---------------------------------------------------------------------------
## Demonstration
#---------------------------------------------------------------------------
# Computing the quantities for RV and QMLE on a reduced simulation
# Storage for results
GBM_RV = zeros(length(dt), reps)
GBM_QMLE = zeros(length(dt), reps)
Hawkes_RV = zeros(length(dt), reps)
Hawkes_QMLE = zeros(length(dt), reps)

# Find starting values for QMLE
startGBM = NUFFTcorrDKFGG(P_GBM, [t t])[2][1,1]/T
startHawkes = NUFFTcorrDKFGG(exp.(P_Hawkes), [t t])[2][1,1]/T

@showprogress "Computing..." for k in 1:reps
    Random.seed!(k)
    t1 = [0; rexp(T, lam)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < T, t1)

    Random.seed!(k+reps)
    t2 = [0; rexp(T, lam)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < T, t2)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_GBM = zeros(n,1); p2_GBM = zeros(n,1)
        p1_Hawkes = zeros(n,1); p2_Hawkes = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1_GBM[j] = P_GBM[Int(floor(γ1)+1), 1]
            p2_GBM[j] = P_GBM[Int(floor(γ2)+1), 2]
            p1_Hawkes[j] = exp(P_Hawkes[Int(floor(γ1)+1), 1])
            p2_Hawkes[j] = exp(P_Hawkes[Int(floor(γ2)+1), 2])
        end
        GBM_RV[i,k] = NUFFTcorrDKFGG([p1_GBM p2_GBM], [t t])[1][1,2]
        Hawkes_RV[i,k] = NUFFTcorrDKFGG([p1_Hawkes p2_Hawkes], [t t])[1][1,2]

        GBM_QMLE[i,k] = QMLEcorr(p1_GBM, p2_GBM, dt[i], [log(startGBM); log(startGBM)])
        Hawkes_QMLE[i,k] = QMLEcorr(p1_Hawkes, p2_Hawkes, dt[i], [log(startHawkes); log(startHawkes)])
    end
end

# Computing the quantities for KEM on a reduced simulation
GBM_KEM = zeros(length(dt), reps)
Hawkes_KEM = zeros(length(dt), reps)

@showprogress "Computing..." for k in 1:reps
    Random.seed!(k)
    t1 = [0; rexp(T, lam)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < T, t1)

    Random.seed!(k+reps)
    t2 = [0; rexp(T, lam)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < T, t2)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_GBM = zeros(n,1); p2_GBM = zeros(n,1)
        p1_Hawkes = zeros(n,1); p2_Hawkes = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1_GBM[j] = P_GBM[Int(floor(γ1)+1), 1]
            p2_GBM[j] = P_GBM[Int(floor(γ2)+1), 2]
            p1_Hawkes[j] = exp(P_Hawkes[Int(floor(γ1)+1), 1])
            p2_Hawkes[j] = exp(P_Hawkes[Int(floor(γ2)+1), 2])
        end
        startGBM = NUFFTcorrDKFGG([p1_GBM p2_GBM], [t t])[2]/T
        startHawkes = NUFFTcorrDKFGG([p1_Hawkes p2_Hawkes], [t t])[2]/T

        GBM_KEM[i,k] = KEM([p1_GBM p2_GBM], startGBM, startGBM, 300, 1e-5)
        Hawkes_KEM[i,k] = KEM([p1_Hawkes p2_Hawkes], startHawkes, startHawkes, 300, 1e-5)
    end
end

# Save GBM and Hawkes demonstration
save("Computed Data/Supp/SuppDemGBMHawkes.jld", "GBM_RV", GBM_RV, "Hawkes_RV", Hawkes_RV,
"GBM_QMLE", GBM_QMLE, "Hawkes_QMLE", Hawkes_QMLE, "GBM_KEM", GBM_KEM, "Hawkes_KEM", Hawkes_KEM)

# Computing the RV, QMLE and KEM on a reduced simulation of SV with noise
SV_RV = zeros(length(dt), reps)
SV_QMLE = zeros(length(dt), reps)
SV_KEM = zeros(length(dt), reps)

@showprogress "Computing..." for k in 1:reps
    Random.seed!(k)
    t1 = [0; rexp(T, lam)]
    t1 = cumsum(t1)
    t1 = filter((x) -> x < T, t1)

    Random.seed!(k+reps)
    t2 = [0; rexp(T, lam)]
    t2 = cumsum(t2)
    t2 = filter((x) -> x < T, t2)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_SV = zeros(n,1); p2_SV = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1_SV[j] = P_SV[Int(floor(γ1)+1), 1]
            p2_SV[j] = P_SV[Int(floor(γ2)+1), 2]
        end
        RV = NUFFTcorrDKFGG([p1_SV p2_SV], [t t])
        startSV_KEM = RV[2]/T
        startSV_QMLE = startSV_KEM[1,1]

        SV_RV[i,k] = RV[1][1,2]
        SV_QMLE[i,k] = QMLEcorr(p1_SV, p2_SV, dt[i], [log(startSV_QMLE); log(startSV_QMLE)])
        SV_KEM[i,k] = KEM([p1_SV p2_SV], startSV_KEM, startSV_KEM, 300, 1e-5)
    end
end

# Save SV demonstration
save("Computed Data/Supp/SuppDemSV.jld", "SV_RV", SV_RV, "SV_QMLE", SV_QMLE, "SV_KEM", SV_KEM)

# Load
SuppDemSamples = load("Computed Data/Supp/SuppDemGBMHawkes.jld")
GBM_RV = SuppDemSamples["GBM_RV"]
Hawkes_RV = SuppDemSamples["Hawkes_RV"]
GBM_QMLE = SuppDemSamples["GBM_QMLE"]
Hawkes_QMLE = SuppDemSamples["Hawkes_QMLE"]
GBM_KEM = SuppDemSamples["GBM_KEM"]
Hawkes_KEM = SuppDemSamples["Hawkes_KEM"]

SuppDemSamplesSV = load("Computed Data/Supp/SuppDemSV.jld")
SV_RV = SuppDemSamplesSV["SV_RV"]
SV_QMLE = SuppDemSamplesSV["SV_QMLE"]
SV_KEM = SuppDemSamplesSV["SV_KEM"]

##----
q = quantile.(TDist(reps-1), [0.975])

err_GBM_RV = (q .* std(GBM_RV, dims = 2))
err_GBM_QMLE = (q .* std(GBM_QMLE, dims = 2))
err_GBM_KEM = (q .* std(GBM_KEM, dims = 2))

p1 = plot(dt, mean(GBM_RV, dims=2), ribbon=err_GBM_RV, fillalpha=.15, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{RV}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p1, dt, mean(GBM_QMLE, dims=2), ribbon=err_GBM_QMLE, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p1, dt, mean(GBM_KEM, dims=2), ribbon=err_GBM_KEM, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{KEM}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p1, dt, theoretical_exp, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Epps}")
hline!(p1, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p1, L"\Delta t\textrm{[sec]}")
ylabel!(p1, L"\rho_{\Delta t}^{ij}")

# savefig(p1, "Plots/Supp/GBMwExpSamplesNewEst.svg")

err_SV_RV = (q .* std(SV_RV, dims = 2))
err_SV_QMLE = (q .* std(SV_QMLE, dims = 2))
err_SV_KEM = (q .* std(SV_KEM, dims = 2))

pp = plot(dt, mean(SV_RV, dims=2), ribbon=err_SV_RV, fillalpha=.15, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{RV}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(pp, dt, mean(SV_QMLE, dims=2), ribbon=err_SV_QMLE, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(pp, dt, mean(SV_KEM, dims=2), ribbon=err_SV_KEM, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{KEM}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(pp, dt, theoretical_exp, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Epps}")
hline!(pp, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(pp, L"\Delta t\textrm{[sec]}")
ylabel!(pp, L"\rho_{\Delta t}^{ij}")

# savefig(pp, "Plots/Supp/SVwExpSamplesNewEst.svg")

err_Hawkes_RV = (q .* std(Hawkes_RV, dims = 2))
err_Hawkes_QMLE = (q .* std(Hawkes_QMLE, dims = 2))
err_Hawkes_KEM = (q .* std(Hawkes_KEM, dims = 2))

p2 = plot(dt, mean(Hawkes_RV, dims=2), ribbon=err_Hawkes_RV, fillalpha=.15, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{RV}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p2, dt, mean(Hawkes_QMLE, dims=2), ribbon=err_Hawkes_QMLE, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p2, dt, mean(Hawkes_KEM, dims=2), ribbon=err_Hawkes_KEM, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{KEM}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p2, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p2, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p2, L"\Delta t\textrm{[sec]}")
ylabel!(p2, L"\rho_{\Delta t}^{ij}")

# savefig(p2, "Plots/Supp/HawkeswExpSamplesNewEst.svg")

#---------------------------------------------------------------------------
## Experiment 2: Different arrivals on Hawkes price model
#---------------------------------------------------------------------------
dt = collect(1:1:100)

## Performing experiment 2 using the QMLE
Hawkes_QMLE_lam1 = zeros(length(dt), reps)
Hawkes_QMLE_lam10 = zeros(length(dt), reps)
Hawkes_QMLE_lam25 = zeros(length(dt), reps)

@showprogress "Computing..." for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    lam10 = 10
    Random.seed!(k)
    t1_lam10 = [0; rexp(T, lam10)]
    t1_lam10 = cumsum(t1_lam10)
    t1_lam10 = filter((x) -> x < T, t1_lam10)
    Random.seed!(k+reps)
    t2_lam10 = [0; rexp(T, lam10)]
    t2_lam10 = cumsum(t2_lam10)
    t2_lam10 = filter((x) -> x < T, t2_lam10)

    lam25 = 25
    Random.seed!(k)
    t1_lam25 = [0; rexp(T, lam25)]
    t1_lam25 = cumsum(t1_lam25)
    t1_lam25 = filter((x) -> x < T, t1_lam25)
    Random.seed!(k+reps)
    t2_lam25 = [0; rexp(T, lam25)]
    t2_lam25 = cumsum(t2_lam25)
    t2_lam25 = filter((x) -> x < T, t2_lam25)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_lam1 = zeros(n,1); p2_lam1 = zeros(n,1)
        p1_lam10 = zeros(n,1); p2_lam10 = zeros(n,1)
        p1_lam25 = zeros(n,1); p2_lam25 = zeros(n,1)
        for j in 1:n
            γ1_lam1 = maximum(filter(x-> x .<= t[j], t1_lam1))
            γ2_lam1 = maximum(filter(x-> x .<= t[j], t2_lam1))
            p1_lam1[j] = exp(P_Hawkes[Int(floor(γ1_lam1)+1), 1])
            p2_lam1[j] = exp(P_Hawkes[Int(floor(γ2_lam1)+1), 2])

            γ1_lam10 = maximum(filter(x-> x .<= t[j], t1_lam10))
            γ2_lam10 = maximum(filter(x-> x .<= t[j], t2_lam10))
            p1_lam10[j] = exp(P_Hawkes[Int(floor(γ1_lam10)+1), 1])
            p2_lam10[j] = exp(P_Hawkes[Int(floor(γ2_lam10)+1), 2])

            γ1_lam25 = maximum(filter(x-> x .<= t[j], t1_lam25))
            γ2_lam25 = maximum(filter(x-> x .<= t[j], t2_lam25))
            p1_lam25[j] = exp(P_Hawkes[Int(floor(γ1_lam25)+1), 1])
            p2_lam25[j] = exp(P_Hawkes[Int(floor(γ2_lam25)+1), 2])
        end

        startlam1 = NUFFTcorrDKFGG([p1_lam1 p2_lam1], [t t])[2][1,1]/T
        Hawkes_QMLE_lam1[i,k] = QMLEcorr(p1_lam1, p2_lam1, dt[i], [log(startlam1); log(startlam1)])

        startlam10 = NUFFTcorrDKFGG([p1_lam10 p2_lam10], [t t])[2][1,1]/T
        Hawkes_QMLE_lam10[i,k] = QMLEcorr(p1_lam10, p2_lam10, dt[i], [log(startlam10); log(startlam10)])

        startlam25 = NUFFTcorrDKFGG([p1_lam25 p2_lam25], [t t])[2][1,1]/T
        Hawkes_QMLE_lam25[i,k] = QMLEcorr(p1_lam25, p2_lam25, dt[i], [log(startlam25); log(startlam25)])
    end
end
# Save and Load
save("Computed Data/Supp/Hawkes_QMLE_exp2.jld", "Hawkes_QMLE_lam1", Hawkes_QMLE_lam1,
"Hawkes_QMLE_lam10", Hawkes_QMLE_lam10, "Hawkes_QMLE_lam25", Hawkes_QMLE_lam25)

QMLE_lam = load("Computed Data/Supp/Hawkes_QMLE_exp2.jld")
Hawkes_QMLE_lam1 = QMLE_lam["Hawkes_QMLE_lam1"]
Hawkes_QMLE_lam10 = QMLE_lam["Hawkes_QMLE_lam10"]
Hawkes_QMLE_lam25 = QMLE_lam["Hawkes_QMLE_lam25"]

q = quantile.(TDist(reps-1), [0.975])

err_Hawkes_QMLE_lam1 = (q .* std(Hawkes_QMLE_lam1, dims = 2))
err_Hawkes_QMLE_lam10 = (q .* std(Hawkes_QMLE_lam10, dims = 2))
err_Hawkes_QMLE_lam25 = (q .* std(Hawkes_QMLE_lam25, dims = 2))

p3 = plot(dt, mean(Hawkes_QMLE_lam1, dims=2), ribbon=err_Hawkes_QMLE_lam1, fillalpha=.3, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"1/\lambda = 1", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p3, dt, mean(Hawkes_QMLE_lam10, dims=2), ribbon=err_Hawkes_QMLE_lam10, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"1/\lambda = 10", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p3, dt, mean(Hawkes_QMLE_lam25, dims=2), ribbon=err_Hawkes_QMLE_lam25, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"1/\lambda = 25", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p3, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p3, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p3, L"\Delta t\textrm{[sec]}")
ylabel!(p3, L"\rho_{\Delta t}^{ij}")

# savefig(p3, "Plots/Supp/Hawkes_QMLE_exp2.svg")

## Performing experiment 2 using the KEM
Hawkes_KEM_lam1 = zeros(length(dt), reps)
Hawkes_KEM_lam10 = zeros(length(dt), reps)
Hawkes_KEM_lam25 = zeros(length(dt), reps)

@showprogress "Computing..." for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    lam10 = 10
    Random.seed!(k)
    t1_lam10 = [0; rexp(T, lam10)]
    t1_lam10 = cumsum(t1_lam10)
    t1_lam10 = filter((x) -> x < T, t1_lam10)
    Random.seed!(k+reps)
    t2_lam10 = [0; rexp(T, lam10)]
    t2_lam10 = cumsum(t2_lam10)
    t2_lam10 = filter((x) -> x < T, t2_lam10)

    lam25 = 25
    Random.seed!(k)
    t1_lam25 = [0; rexp(T, lam25)]
    t1_lam25 = cumsum(t1_lam25)
    t1_lam25 = filter((x) -> x < T, t1_lam25)
    Random.seed!(k+reps)
    t2_lam25 = [0; rexp(T, lam25)]
    t2_lam25 = cumsum(t2_lam25)
    t2_lam25 = filter((x) -> x < T, t2_lam25)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_lam1 = zeros(n,1); p2_lam1 = zeros(n,1)
        p1_lam10 = zeros(n,1); p2_lam10 = zeros(n,1)
        p1_lam25 = zeros(n,1); p2_lam25 = zeros(n,1)
        for j in 1:n
            γ1_lam1 = maximum(filter(x-> x .<= t[j], t1_lam1))
            γ2_lam1 = maximum(filter(x-> x .<= t[j], t2_lam1))
            p1_lam1[j] = exp(P_Hawkes[Int(floor(γ1_lam1)+1), 1])
            p2_lam1[j] = exp(P_Hawkes[Int(floor(γ2_lam1)+1), 2])

            γ1_lam10 = maximum(filter(x-> x .<= t[j], t1_lam10))
            γ2_lam10 = maximum(filter(x-> x .<= t[j], t2_lam10))
            p1_lam10[j] = exp(P_Hawkes[Int(floor(γ1_lam10)+1), 1])
            p2_lam10[j] = exp(P_Hawkes[Int(floor(γ2_lam10)+1), 2])

            γ1_lam25 = maximum(filter(x-> x .<= t[j], t1_lam25))
            γ2_lam25 = maximum(filter(x-> x .<= t[j], t2_lam25))
            p1_lam25[j] = exp(P_Hawkes[Int(floor(γ1_lam25)+1), 1])
            p2_lam25[j] = exp(P_Hawkes[Int(floor(γ2_lam25)+1), 2])
        end

        startlam1 = NUFFTcorrDKFGG([p1_lam1 p2_lam1], [t t])[2]/T
        Hawkes_KEM_lam1[i,k] = KEM([p1_lam1 p2_lam1], startlam1, startlam1, 300, 1e-5)

        startlam10 = NUFFTcorrDKFGG([p1_lam10 p2_lam10], [t t])[2]/T
        Hawkes_KEM_lam10[i,k] = KEM([p1_lam10 p2_lam10], startlam10, startlam10, 300, 1e-5)

        startlam25 = NUFFTcorrDKFGG([p1_lam25 p2_lam25], [t t])[2]/T
        Hawkes_KEM_lam25[i,k] = KEM([p1_lam25 p2_lam25], startlam25, startlam25, 300, 1e-5)
    end
end
# Save and Load
save("Computed Data/Supp/Hawkes_KEM_exp2.jld", "Hawkes_KEM_lam1", Hawkes_KEM_lam1,
"Hawkes_KEM_lam10", Hawkes_KEM_lam10, "Hawkes_KEM_lam25", Hawkes_KEM_lam25)

KEM_lam = load("Computed Data/Supp/Hawkes_KEM_exp2.jld")
Hawkes_KEM_lam1 = KEM_lam["Hawkes_KEM_lam1"]
Hawkes_KEM_lam10 = KEM_lam["Hawkes_KEM_lam10"]
Hawkes_KEM_lam25 = KEM_lam["Hawkes_KEM_lam25"]

q = quantile.(TDist(reps-1), [0.975])

err_Hawkes_KEM_lam1 = (q .* std(Hawkes_KEM_lam1, dims = 2))
err_Hawkes_KEM_lam10 = (q .* std(Hawkes_KEM_lam10, dims = 2))
err_Hawkes_KEM_lam25 = (q .* std(Hawkes_KEM_lam25, dims = 2))

p4 = plot(dt, mean(Hawkes_KEM_lam1, dims=2), ribbon=err_Hawkes_KEM_lam1, fillalpha=.3, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"1/\lambda = 1", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p4, dt, mean(Hawkes_KEM_lam10, dims=2), ribbon=err_Hawkes_KEM_lam10, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"1/\lambda = 10", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p4, dt, mean(Hawkes_KEM_lam25, dims=2), ribbon=err_Hawkes_KEM_lam25, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"1/\lambda = 25", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p4, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p4, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p4, L"\Delta t\textrm{[sec]}")
ylabel!(p4, L"\rho_{\Delta t}^{ij}")

# savefig(p4, "Plots/Supp/Hawkes_KEM_exp2.svg")

#---------------------------------------------------------------------------
## Experiment 2: Different arrivals on SV model with noise
#---------------------------------------------------------------------------
dt = collect(1:1:100)

## Performing experiment 2 using the QMLE
SV_QMLE_lam1 = zeros(length(dt), reps)
SV_QMLE_lam10 = zeros(length(dt), reps)
SV_QMLE_lam25 = zeros(length(dt), reps)

@showprogress "Computing..." for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    lam10 = 10
    Random.seed!(k)
    t1_lam10 = [0; rexp(T, lam10)]
    t1_lam10 = cumsum(t1_lam10)
    t1_lam10 = filter((x) -> x < T, t1_lam10)
    Random.seed!(k+reps)
    t2_lam10 = [0; rexp(T, lam10)]
    t2_lam10 = cumsum(t2_lam10)
    t2_lam10 = filter((x) -> x < T, t2_lam10)

    lam25 = 25
    Random.seed!(k)
    t1_lam25 = [0; rexp(T, lam25)]
    t1_lam25 = cumsum(t1_lam25)
    t1_lam25 = filter((x) -> x < T, t1_lam25)
    Random.seed!(k+reps)
    t2_lam25 = [0; rexp(T, lam25)]
    t2_lam25 = cumsum(t2_lam25)
    t2_lam25 = filter((x) -> x < T, t2_lam25)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_lam1 = zeros(n,1); p2_lam1 = zeros(n,1)
        p1_lam10 = zeros(n,1); p2_lam10 = zeros(n,1)
        p1_lam25 = zeros(n,1); p2_lam25 = zeros(n,1)
        for j in 1:n
            γ1_lam1 = maximum(filter(x-> x .<= t[j], t1_lam1))
            γ2_lam1 = maximum(filter(x-> x .<= t[j], t2_lam1))
            p1_lam1[j] = (P_SV[Int(floor(γ1_lam1)+1), 1])
            p2_lam1[j] = (P_SV[Int(floor(γ2_lam1)+1), 2])

            γ1_lam10 = maximum(filter(x-> x .<= t[j], t1_lam10))
            γ2_lam10 = maximum(filter(x-> x .<= t[j], t2_lam10))
            p1_lam10[j] = (P_SV[Int(floor(γ1_lam10)+1), 1])
            p2_lam10[j] = (P_SV[Int(floor(γ2_lam10)+1), 2])

            γ1_lam25 = maximum(filter(x-> x .<= t[j], t1_lam25))
            γ2_lam25 = maximum(filter(x-> x .<= t[j], t2_lam25))
            p1_lam25[j] = (P_SV[Int(floor(γ1_lam25)+1), 1])
            p2_lam25[j] = (P_SV[Int(floor(γ2_lam25)+1), 2])
        end

        startlam1 = NUFFTcorrDKFGG([p1_lam1 p2_lam1], [t t])[2][1,1]/T
        SV_QMLE_lam1[i,k] = QMLEcorr(p1_lam1, p2_lam1, dt[i], [log(startlam1); log(startlam1)])

        startlam10 = NUFFTcorrDKFGG([p1_lam10 p2_lam10], [t t])[2][1,1]/T
        SV_QMLE_lam10[i,k] = QMLEcorr(p1_lam10, p2_lam10, dt[i], [log(startlam10); log(startlam10)])

        startlam25 = NUFFTcorrDKFGG([p1_lam25 p2_lam25], [t t])[2][1,1]/T
        SV_QMLE_lam25[i,k] = QMLEcorr(p1_lam25, p2_lam25, dt[i], [log(startlam25); log(startlam25)])
    end
end
# Save and Load
save("Computed Data/Supp/SV_QMLE_exp2.jld", "SV_QMLE_lam1", SV_QMLE_lam1,
"SV_QMLE_lam10", SV_QMLE_lam10, "SV_QMLE_lam25", SV_QMLE_lam25)

SV_lam = load("Computed Data/Supp/SV_QMLE_exp2.jld")
SV_QMLE_lam1 = SV_lam["SV_QMLE_lam1"]
SV_QMLE_lam10 = SV_lam["SV_QMLE_lam10"]
SV_QMLE_lam25 = SV_lam["SV_QMLE_lam25"]

q = quantile.(TDist(reps-1), [0.975])

err_SV_QMLE_lam1 = (q .* std(SV_QMLE_lam1, dims = 2))
err_SV_QMLE_lam10 = (q .* std(SV_QMLE_lam10, dims = 2))
err_SV_QMLE_lam25 = (q .* std(SV_QMLE_lam25, dims = 2))

p5 = plot(dt, mean(SV_QMLE_lam1, dims=2), ribbon=err_SV_QMLE_lam1, fillalpha=.3, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"1/\lambda = 1", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p5, dt, mean(SV_QMLE_lam10, dims=2), ribbon=err_SV_QMLE_lam10, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"1/\lambda = 10", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p5, dt, mean(SV_QMLE_lam25, dims=2), ribbon=err_SV_QMLE_lam25, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"1/\lambda = 25", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p5, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p5, L"\Delta t\textrm{[sec]}")
ylabel!(p5, L"\rho_{\Delta t}^{ij}")

# savefig(p5, "Plots/Supp/SV_QMLE_exp2.svg")

## Performing experiment 2 using the KEM
SV_KEM_lam1 = zeros(length(dt), reps)
SV_KEM_lam10 = zeros(length(dt), reps)
SV_KEM_lam25 = zeros(length(dt), reps)

@showprogress "Computing..." for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    lam10 = 10
    Random.seed!(k)
    t1_lam10 = [0; rexp(T, lam10)]
    t1_lam10 = cumsum(t1_lam10)
    t1_lam10 = filter((x) -> x < T, t1_lam10)
    Random.seed!(k+reps)
    t2_lam10 = [0; rexp(T, lam10)]
    t2_lam10 = cumsum(t2_lam10)
    t2_lam10 = filter((x) -> x < T, t2_lam10)

    lam25 = 25
    Random.seed!(k)
    t1_lam25 = [0; rexp(T, lam25)]
    t1_lam25 = cumsum(t1_lam25)
    t1_lam25 = filter((x) -> x < T, t1_lam25)
    Random.seed!(k+reps)
    t2_lam25 = [0; rexp(T, lam25)]
    t2_lam25 = cumsum(t2_lam25)
    t2_lam25 = filter((x) -> x < T, t2_lam25)

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1_lam1 = zeros(n,1); p2_lam1 = zeros(n,1)
        p1_lam10 = zeros(n,1); p2_lam10 = zeros(n,1)
        p1_lam25 = zeros(n,1); p2_lam25 = zeros(n,1)
        for j in 1:n
            γ1_lam1 = maximum(filter(x-> x .<= t[j], t1_lam1))
            γ2_lam1 = maximum(filter(x-> x .<= t[j], t2_lam1))
            p1_lam1[j] = (P_SV[Int(floor(γ1_lam1)+1), 1])
            p2_lam1[j] = (P_SV[Int(floor(γ2_lam1)+1), 2])

            γ1_lam10 = maximum(filter(x-> x .<= t[j], t1_lam10))
            γ2_lam10 = maximum(filter(x-> x .<= t[j], t2_lam10))
            p1_lam10[j] = (P_SV[Int(floor(γ1_lam10)+1), 1])
            p2_lam10[j] = (P_SV[Int(floor(γ2_lam10)+1), 2])

            γ1_lam25 = maximum(filter(x-> x .<= t[j], t1_lam25))
            γ2_lam25 = maximum(filter(x-> x .<= t[j], t2_lam25))
            p1_lam25[j] = (P_SV[Int(floor(γ1_lam25)+1), 1])
            p2_lam25[j] = (P_SV[Int(floor(γ2_lam25)+1), 2])
        end

        startlam1 = NUFFTcorrDKFGG([p1_lam1 p2_lam1], [t t])[2]/T
        SV_KEM_lam1[i,k] = KEM([p1_lam1 p2_lam1], startlam1, startlam1, 300, 1e-5)

        startlam10 = NUFFTcorrDKFGG([p1_lam10 p2_lam10], [t t])[2]/T
        SV_KEM_lam10[i,k] = KEM([p1_lam10 p2_lam10], startlam10, startlam10, 300, 1e-5)

        startlam25 = NUFFTcorrDKFGG([p1_lam25 p2_lam25], [t t])[2]/T
        SV_KEM_lam25[i,k] = KEM([p1_lam25 p2_lam25], startlam25, startlam25, 300, 1e-5)
    end
end
# Save and Load
save("Computed Data/Supp/SV_KEM_exp2.jld", "SV_KEM_lam1", SV_KEM_lam1,
"SV_KEM_lam10", SV_KEM_lam10, "SV_KEM_lam25", SV_KEM_lam25)

SV_KEM_lam = load("Computed Data/Supp/SV_KEM_exp2.jld")
SV_KEM_lam1 = SV_KEM_lam["SV_KEM_lam1"]
SV_KEM_lam10 = SV_KEM_lam["SV_KEM_lam10"]
SV_KEM_lam25 = SV_KEM_lam["SV_KEM_lam25"]

q = quantile.(TDist(reps-1), [0.975])

err_SV_KEM_lam1 = (q .* std(SV_KEM_lam1, dims = 2))
err_SV_KEM_lam10 = (q .* std(SV_KEM_lam10, dims = 2))
err_SV_KEM_lam25 = (q .* std(SV_KEM_lam25, dims = 2))

p6 = plot(dt, mean(SV_KEM_lam1, dims=2), ribbon=err_SV_KEM_lam1, fillalpha=.3, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"1/\lambda = 1", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p6, dt, mean(SV_KEM_lam10, dims=2), ribbon=err_SV_KEM_lam10, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"1/\lambda = 10", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p6, dt, mean(SV_KEM_lam25, dims=2), ribbon=err_SV_KEM_lam25, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"1/\lambda = 25", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p6, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p6, L"\Delta t\textrm{[sec]}")
ylabel!(p6, L"\rho_{\Delta t}^{ij}")

# savefig(p6, "Plots/Supp/SV_KEM_exp2.svg")
