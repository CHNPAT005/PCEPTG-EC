## Author: Patrick Chang
# Script file to investigate the correction of the Epps effect arising
# from asynchronous sampling

using LinearAlgebra, Plots, LaTeXStrings, StatsBase, Intervals, JLD, ProgressMeter, Distributions

cd("/Users/patrickchang1/PCEPTG-EC")

include("../../Functions/Hawkes/Hawkes.jl")
include("../../Functions/SDEs/GBM.jl")
include("../../Functions/SDEs/Merton Model.jl")
include("../../Functions/SDEs/SVwNoise.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/HY/HYcorr.jl")

#---------------------------------------------------------------------------
## Theoretical correlations from a Hawkes price model used in Barcy et al.

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

#---------------------------------------------------------------------------
## RV correction
#---------------------------------------------------------------------------
# Computes the probability of flat trading
function zeroticks(P)
    dif = diff(P,dims=1)
    n = size(dif)[1]
    m = size(dif)[2]
    p = zeros(m, 1)

    for i in 1:m
        r = dif[:,i]
        count = 0
        for j in 1:n
            if r[j] == 0
                count += 1
            end
        end
        p[i] = count/n
    end
    return p
end
# Computes the expectation of overlapping intervals
function flattime(τ, t1, t2, T)
    t1 = [0; t1]
    t2 = [0; t2]
    syngrid = collect(0:τ:T)
    n = length(syngrid)
    γ1 = zeros(n,1)
    γ2 = zeros(n,1)
    for i in 1:n
        γ1[i] = maximum(filter(x-> x .<= syngrid[i], t1))
        γ2[i] = maximum(filter(x-> x .<= syngrid[i], t2))
    end
    ints = zeros(n-1,1)
    for i in 2:n
        a = γ1[i-1]..γ1[i]
        b = γ2[i-1]..γ2[i]
        c = intersect(a, b)
        ints[i-1] = c.last-c.first
    end
    return mean(ints) / (sqrt(mean(diff(γ1, dims = 1))*mean(diff(γ2, dims = 1))))
end
# Simulates a random exponential sample
function rexp(n, mean)
    t = -mean .* log.(rand(n))
end

#---------------------------------------------------------------------------
## Diffusive Price model
#---------------------------------------------------------------------------
## Simple asynchronous sample path

n = 500
P = GBM(n+1, [0.01/86400], [0.1/86400])
X = log.(P)
t = collect(0:1:n)

lam = 15
Random.seed!(1)
t_asyn = [0; rexp(n, lam)]
t_asyn = cumsum(t_asyn)
t_asyn = filter((x) -> x < n, t_asyn)

p_asyn = zeros(n+1,1)
for j in 1:n+1
    γ1 = maximum(filter(x-> x .<= t[j], t_asyn))
    p_asyn[j] = X[Int(floor(γ1)+1), 1]
end

p1 = plot(t, X, color = :blue, line=(1, [:dot]), legend = :topright, label = L"\textrm{Synchronous}", dpi = 300, size = (600, 500))
plot!(p1, t, p_asyn, linetype = :steppost, color = :red, line=(1, [:solid]), label = L"\textrm{Synchronised}")
xlabel!(p1, L"\textrm{time [sec]}")
ylabel!(p1, L"X_{t}")

# savefig(p1, "Plots/EppsCorrection2/GBMPricePaths.svg")

#---------------------------------------------------------------------------
# GBM price model with exponential sampling
reps = 100

T = 3600*20
ρ = theoreticalCorr(0.023, 0.05, 0.11)

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
        sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]

P_GBM = GBM(T+1, mu, sigma)
t_GBM = reshape([collect(0:1:T); collect(0:1:T)], T+1, 2)

lam = 15
lam2 = 1/lam

dt = collect(1:1:400)

measured_GBM_exp = zeros(length(dt), reps)
measured_GBM_prevtick_exp = zeros(length(dt), reps)
measured_GBM_flattime_adj_exp = zeros(length(dt), reps)
HYGBM_exp = zeros(reps, 1)

# takes roughly 2 hours to compute
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
        p1 = zeros(n,1)
        p2 = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1[j] = P_GBM[Int(floor(γ1)+1), 1]
            p2[j] = P_GBM[Int(floor(γ2)+1), 2]
        end
        p = zeroticks([p1 p2])
        adj = flattime(dt[i], t1[2:end], t2[2:end], T)

        measured_GBM_exp[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
        measured_GBM_prevtick_exp[i,k] = measured_GBM_exp[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
        measured_GBM_flattime_adj_exp[i,k] = measured_GBM_exp[i,k]/adj
    end

    P1 = P_GBM[Int.(floor.(t1).+1), 1]
    P2 = P_GBM[Int.(floor.(t2).+1), 2]
    HYGBM_exp[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

theoretical_exp = ρ .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))

# Save and Load
save("Computed Data/EppsCorrection/GBMwExpSaples2.jld", "measured_GBM_exp", measured_GBM_exp, "measured_GBM_prevtick_exp", measured_GBM_prevtick_exp,
"measured_GBM_flattime_adj_exp", measured_GBM_flattime_adj_exp, "HYGBM_exp", HYGBM_exp)

GBMwExpSaples = load("Computed Data/EppsCorrection/GBMwExpSaples2.jld")
measured_GBM_exp = GBMwExpSaples["measured_GBM_exp"]
measured_GBM_prevtick_exp = GBMwExpSaples["measured_GBM_prevtick_exp"]
measured_GBM_flattime_adj_exp = GBMwExpSaples["measured_GBM_flattime_adj_exp"]
HYGBM_exp = GBMwExpSaples["HYGBM_exp"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_GBM_exp = (q .* std(measured_GBM_exp, dims = 2))
err_measured_GBM_prevtick_exp = (q .* std(measured_GBM_prevtick_exp, dims = 2))
err_measured_GBM_flattime_adj_exp = (q .* std(measured_GBM_flattime_adj_exp, dims = 2))
err_HYGBM_exp = (q .* std(HYGBM_exp))

p2 = plot(dt, mean(measured_GBM_exp, dims=2), ribbon=err_measured_GBM_exp, fillalpha=.15, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p2, dt, mean(measured_GBM_prevtick_exp, dims=2), ribbon=err_measured_GBM_prevtick_exp, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p2, dt, mean(measured_GBM_flattime_adj_exp, dims=2), ribbon=err_measured_GBM_flattime_adj_exp, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p2, dt, theoretical_exp, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Epps}")
hline!(p2, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
hline!(p2, [mean(HYGBM_exp)], ribbon=err_HYGBM_exp, fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p2, L"\Delta t\textrm{[sec]}")
ylabel!(p2, L"\rho_{\Delta t}^{ij}")

# savefig(p2, "Plots/EppsCorrection2/GBMPriceModelwExpSamples.svg")

#---------------------------------------------------------------------------
# GBM price model with Hawkes sampling

measured_GBM_hawkes = zeros(length(dt), reps)
measured_GBM_prevtick_hawkes = zeros(length(dt), reps)
measured_GBM_flattime_adj_hawkes = zeros(length(dt), reps)
HYGBM_hawkes = zeros(reps, 1)

# Seed is set this way so the Hawkes processes look different
# this is due to how seeds are set in the Hawkes Simulation for reproducibility
Random.seed!(2020)
seeds = Int.(floor.(rand(reps) .* 1000000))

# takes roughly 12 hours to compute
@showprogress "Computing..." for k in 1:reps
    t = simulateHawkes([0.015;0.015], [0 0.023; 0.023 0], [0 0.11; 0.11 0], T, seed = seeds[k])

    t1 = [0;t[1]]
    t2 = [0;t[2]]

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1 = zeros(n,1)
        p2 = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1[j] = P_GBM[Int(floor(γ1)+1), 1]
            p2[j] = P_GBM[Int(floor(γ2)+1), 2]
        end
        p = zeroticks([p1 p2])
        adj = flattime(dt[i], t1[2:end], t2[2:end], T)

        measured_GBM_hawkes[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
        measured_GBM_prevtick_hawkes[i,k] = measured_GBM_hawkes[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
        measured_GBM_flattime_adj_hawkes[i,k] = measured_GBM_hawkes[i,k]/adj
    end
    P1 = P_GBM[Int.(floor.(t1).+1), 1]
    P2 = P_GBM[Int.(floor.(t2).+1), 2]
    HYGBM_hawkes[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

# Save and Load
save("Computed Data/EppsCorrection/GBMwHawkesSaples2.jld", "measured_GBM_hawkes", measured_GBM_hawkes, "measured_GBM_prevtick_hawkes", measured_GBM_prevtick_hawkes,
"measured_GBM_flattime_adj_hawkes", measured_GBM_flattime_adj_hawkes, "HYGBM_hawkes", HYGBM_hawkes)

GBMwHawkesSaples = load("Computed Data/EppsCorrection/GBMwHawkesSaples2.jld")
measured_GBM_hawkes = GBMwHawkesSaples["measured_GBM_hawkes"]
measured_GBM_prevtick_hawkes = GBMwHawkesSaples["measured_GBM_prevtick_hawkes"]
measured_GBM_flattime_adj_hawkes = GBMwHawkesSaples["measured_GBM_flattime_adj_hawkes"]
HYGBM_hawkes = GBMwHawkesSaples["HYGBM_hawkes"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_GBM_hawkes = (q .* std(measured_GBM_hawkes, dims = 2))
err_measured_GBM_prevtick_hawkes = (q .* std(measured_GBM_prevtick_hawkes, dims = 2))
err_measured_GBM_flattime_adj_hawkes = (q .* std(measured_GBM_flattime_adj_hawkes, dims = 2))
err_HYGBM_hawkes = (q .* std(HYGBM_hawkes))

p3 = plot(dt, mean(measured_GBM_hawkes, dims=2), ribbon=err_measured_GBM_hawkes, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (0, 2), size = (600, 500))
plot!(p3, dt, mean(measured_GBM_prevtick_hawkes, dims=2), ribbon=err_measured_GBM_prevtick_hawkes, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p3, dt, mean(measured_GBM_flattime_adj_hawkes, dims=2), ribbon=err_measured_GBM_flattime_adj_hawkes, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p3, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
hline!(p3, [mean(HYGBM_hawkes)], ribbon=err_HYGBM_hawkes, fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p3, L"\Delta t\textrm{[sec]}")
ylabel!(p3, L"\rho_{\Delta t}^{ij}")

# savefig(p3, "Plots/EppsCorrection2/GBMPriceModelwHawkesSamples.svg")

#---------------------------------------------------------------------------
## Hawkes price model - generated using bottom-up events
#---------------------------------------------------------------------------

T = 3600*20

ρ = theoreticalCorr(0.023, 0.05, 0.11)

par_1 = BarcyParams(0.015, 0.023, 0.05, 0.11)
lambda0_1 = par_1[1]; alpha_1 = par_1[2]; beta_1 = par_1[3]
t1 = simulateHawkes(lambda0_1, alpha_1, beta_1, T, seed = 19549293)

p1_1 = getuniformPrices(0, 1, T, t1[1], t1[2])
p2_1 = getuniformPrices(0, 1, T, t1[3], t1[4])
P_1 = [p1_1 p2_1]

t = collect(0:1:T)

p4 = plot(t.* (20/T), p1_1, label = "", color = :blue, linetype = :steppre, dpi = 300, size = (600, 500))
xlabel!(p4, L"\textrm{time [hour]}")
ylabel!(p4, L"X_{t}")

# savefig(p4, "Plots/EppsCorrection2/HawkesPricePathsFull.svg")

p5 = plot((t.* (20/T))[1:3600], p1_1[1:3600], label = "", color = :blue, linetype = :steppre, dpi = 300, size = (600, 500))
xlabel!(p5, L"\textrm{time [hour]}")
ylabel!(p5, L"X_{t}")

# savefig(p5, "Plots/EppsCorrection2/HawkesPricePathsTrunc.svg")

#---------------------------------------------------------------------------
## Synchronous Epps effect

dt = collect(1:1:400)

measured1_pear = zeros(length(dt), reps)
theoretical = zeros(length(dt), 1)

# Measured

Random.seed!(2020)
seeds = Int.(floor.(rand(reps) .* 1000000))

# Takes rougly 1 hour to compute
@showprogress "Computing..." for k in 1:reps
    t_hawkes = simulateHawkes(lambda0_1, alpha_1, beta_1, T, seed = seeds[k])
    for i in 1:length(dt)
        p1_2 = getuniformPrices(0, dt[i], T, t_hawkes[1], t_hawkes[2])
        p2_2 = getuniformPrices(0, dt[i], T, t_hawkes[3], t_hawkes[4])
        P_1 = exp.([p1_2 p2_2])
        t = collect(0:dt[i]:T)

        measured1_pear[i,k] = NUFFTcorrDKFGG(P_1, [t t])[1][1,2]# measuredEpps_pearson(log.(P_1))
    end
end

for i in 1:length(dt)
    theoretical[i] = theoreticalEpps(dt[i], 0.015, 0.023, 0.05, 0.11)
end

# Save and Load
save("Computed Data/EppsCorrection/SynEpps.jld", "measured1_pear", measured1_pear, "theoretical", theoretical)

SynEpps = load("Computed Data/EppsCorrection/SynEpps.jld")
measured1_pear = SynEpps["measured1_pear"]
theoretical = SynEpps["theoretical"]

q = quantile.(TDist(reps-1), [0.975])

err_measured1_pear = (q .* std(measured1_pear, dims = 2))

# Plot
p6 = plot(dt, mean(measured1_pear, dims=2), ribbon=err_measured1_pear, fillalpha=.15, legend = :bottomright, label = L"\textrm{Measured}", color = :red, marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p6, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p6, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p6, L"\Delta t\textrm{[sec]}")
ylabel!(p6, L"\rho_{\Delta t}^{ij}")

# savefig(p6, "Plots/EppsCorrection2/HawkesSynEpps.svg")


#---------------------------------------------------------------------------
## Hawkes price model with Exponential sampling

measured_hawkes_exp = zeros(length(dt), reps)
measured_hawkes_prevtick_exp = zeros(length(dt), reps)
measured_hawkes_flattime_adj_exp = zeros(length(dt), reps)
HYHawkes_exp = zeros(reps, 1)

# Takes roughly 2 hours to compute
@showprogress "Computing..." for k in 1:reps
    lam = 15
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
        p1 = zeros(n,1)
        p2 = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1[j] = exp(P_1[Int(floor(γ1)+1), 1])
            p2[j] = exp(P_1[Int(floor(γ2)+1), 2])
        end
        p = zeroticks([p1 p2])
        adj = flattime(dt[i], t1[2:end], t2[2:end], T)

        measured_hawkes_exp[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
        measured_hawkes_prevtick_exp[i,k] = measured_hawkes_exp[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
        measured_hawkes_flattime_adj_exp[i,k] = measured_hawkes_exp[i,k]/adj
    end
    P1 = exp.(P_1[Int.(floor.(t1).+1), 1])
    P2 = exp.(P_1[Int.(floor.(t2).+1), 2])
    HYHawkes_exp[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

# Save and Load
save("Computed Data/EppsCorrection/HawkeswExpSaples2.jld", "measured_hawkes_exp", measured_hawkes_exp, "measured_hawkes_prevtick_exp", measured_hawkes_prevtick_exp,
"measured_hawkes_flattime_adj_exp", measured_hawkes_flattime_adj_exp, "HYHawkes_exp", HYHawkes_exp)

HawkeswExpSaples = load("Computed Data/EppsCorrection/HawkeswExpSaples2.jld")
measured_hawkes_exp = HawkeswExpSaples["measured_hawkes_exp"]
measured_hawkes_prevtick_exp = HawkeswExpSaples["measured_hawkes_prevtick_exp"]
measured_hawkes_flattime_adj_exp = HawkeswExpSaples["measured_hawkes_flattime_adj_exp"]
HYHawkes_exp = HawkeswExpSaples["HYHawkes_exp"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_hawkes_exp = (q .* std(measured_hawkes_exp, dims = 2))
err_measured_hawkes_prevtick_exp = (q .* std(measured_hawkes_prevtick_exp, dims = 2))
err_measured_hawkes_flattime_adj_exp = (q .* std(measured_hawkes_flattime_adj_exp, dims = 2))
err_HYHawkes_exp = (q .* std(HYHawkes_exp))

p7 = plot(dt, mean(measured_hawkes_exp, dims=2), ribbon=err_measured_hawkes_exp, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p7, dt, mean(measured_hawkes_prevtick_exp, dims=2), ribbon=err_measured_hawkes_prevtick_exp, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p7, dt, mean(measured_hawkes_flattime_adj_exp, dims=2), ribbon=err_measured_hawkes_flattime_adj_exp, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p7, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p7, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
hline!(p7, [mean(HYHawkes_exp)], ribbon=err_HYHawkes_exp, fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p7, L"\Delta t\textrm{[sec]}")
ylabel!(p7, L"\rho_{\Delta t}^{ij}")

# savefig(p7, "Plots/EppsCorrection2/HawkesPriceModelwExpSamples.svg")

#---------------------------------------------------------------------------
## Hawkes price model with Hawkes sampling

measured_hawkes_hawkes = zeros(length(dt), reps)
measured_hawkes_prevtick_hawkes = zeros(length(dt), reps)
measured_hawkes_flattime_adj_hawkes = zeros(length(dt), reps)
HYHawkes_hawkes = zeros(reps, 1)

Random.seed!(2020)
seeds = Int.(floor.(rand(reps) .* 1000000))

# Takes roughly 12 hours to compute
@showprogress "Computing..." for k in 1:reps
    t = simulateHawkes([0.015;0.015], [0 0.023; 0.023 0], [0 0.11; 0.11 0], T, seed = seeds[k])

    t1 = [0;t[1]]
    t2 = [0;t[2]]

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1 = zeros(n,1)
        p2 = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1[j] = exp(P_1[Int(floor(γ1)+1), 1])
            p2[j] = exp(P_1[Int(floor(γ2)+1), 2])
        end
        p = zeroticks([p1 p2])
        adj = flattime(dt[i], t1[2:end], t2[2:end], T)

        measured_hawkes_hawkes[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
        measured_hawkes_prevtick_hawkes[i,k] = measured_hawkes_hawkes[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
        measured_hawkes_flattime_adj_hawkes[i,k] = measured_hawkes_hawkes[i,k]/adj
    end
    P1 = exp.(P_1[Int.(floor.(t1).+1), 1])
    P2 = exp.(P_1[Int.(floor.(t2).+1), 2])
    HYHawkes_hawkes[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

# Save and Load
save("Computed Data/EppsCorrection/HawkeswHawkesSaples2.jld", "measured_hawkes_hawkes", measured_hawkes_hawkes, "measured_hawkes_prevtick_hawkes", measured_hawkes_prevtick_hawkes,
"measured_hawkes_flattime_adj_hawkes", measured_hawkes_flattime_adj_hawkes, "HYHawkes_hawkes", HYHawkes_hawkes)

HawkeswHawkesSaples = load("Computed Data/EppsCorrection/HawkeswHawkesSaples2.jld")
measured_hawkes_hawkes = HawkeswHawkesSaples["measured_hawkes_hawkes"]
measured_hawkes_prevtick_hawkes = HawkeswHawkesSaples["measured_hawkes_prevtick_hawkes"]
measured_hawkes_flattime_adj_hawkes = HawkeswHawkesSaples["measured_hawkes_flattime_adj_hawkes"]
HYHawkes_hawkes = HawkeswHawkesSaples["HYHawkes_hawkes"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_hawkes_hawkes = (q .* std(measured_hawkes_hawkes, dims = 2))
err_measured_hawkes_prevtick_hawkes = (q .* std(measured_hawkes_prevtick_hawkes, dims = 2))
err_measured_hawkes_flattime_adj_hawkes = (q .* std(measured_hawkes_flattime_adj_hawkes, dims = 2))
err_HYHawkes_hawkes = (q .* std(HYHawkes_hawkes))

p8 = plot(dt, mean(measured_hawkes_hawkes, dims=2), ribbon=err_measured_hawkes_hawkes, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (0, 2), size = (600, 500))
plot!(p8, dt, mean(measured_hawkes_prevtick_hawkes, dims=2), ribbon=err_measured_hawkes_prevtick_hawkes, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p8, dt, mean(measured_hawkes_flattime_adj_hawkes, dims=2), ribbon=err_measured_hawkes_flattime_adj_hawkes, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p8, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p8, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
hline!(p8, [mean(HYHawkes_hawkes)], ribbon=err_HYHawkes_hawkes, fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p8, L"\Delta t\textrm{[sec]}")
ylabel!(p8, L"\rho_{\Delta t}^{ij}")

# savefig(p8, "Plots/EppsCorrection2/HawkesPriceModelwHawkesSamples.svg")

#---------------------------------------------------------------------------
## How the sampling frequency affects the HY: Hypothesis, botton down events
## require time to build up correlation, therefore when sampling freq is high
## HY will have lower correlation

lamrange = collect(1:45)
HYlam = zeros(length(lamrange), reps)
HYlam_GBM = zeros(length(lamrange), reps)

# Takes roughly 2.5 hours to compute
@showprogress "Computing..." for k in 1:reps
    for i in 1:length(lamrange)
        lam = lamrange[i]
        Random.seed!(i+k)
        t1 = [0; rexp(T, lam)]
        t1 = cumsum(t1)
        t1 = filter((x) -> x < T, t1)

        Random.seed!(i+k+reps)
        t2 = [0; rexp(T, lam)]
        t2 = cumsum(t2)
        t2 = filter((x) -> x < T, t2)

        P1 = exp.(P_1[Int.(floor.(t1).+1), 1])
        P2 = exp.(P_1[Int.(floor.(t2).+1), 2])
        HYlam[i,k] = HYcorr(P1,P2,t1,t2)[1][1,2]

        P1_GBM = P_GBM[Int.(floor.(t1).+1), 1]
        P2_GBM = P_GBM[Int.(floor.(t2).+1), 2]
        HYlam_GBM[i,k] = HYcorr(P1_GBM,P2_GBM,t1,t2)[1][1,2]
    end
end

# Save and Load
save("Computed Data/EppsCorrection/HYFreq2.jld", "HYlam", HYlam, "HYlam_GBM", HYlam_GBM)

HYRes = load("Computed Data/EppsCorrection/HYFreq2.jld")
HYlam = HYRes["HYlam"]
HYlam_GBM = HYRes["HYlam_GBM"]

q = quantile.(TDist(reps-1), [0.975])

err_HYlam = (q .* std(HYlam, dims = 2))

p9 = plot(lamrange, mean(HYlam, dims=2), ribbon=err_HYlam, fillalpha=.15, color = :brown, line=(1, [:dash]), legend = :bottomright, label = L"\textrm{HY}", dpi = 300, ylims = (0.1, 0.8), size = (600, 500))
hline!(p9, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p9, L"\textrm{Average inter-arrival}(1/\lambda)\textrm{[sec]}")
ylabel!(p9, L"\rho(1/\lambda)")

# savefig(p9, "Plots/EppsCorrection2/HawkesPriceModelHYSamplingFreq.svg")

err_HYlam_GBM = (q .* std(HYlam_GBM, dims = 2))

p9_2 = plot(lamrange, mean(HYlam_GBM, dims=2), ribbon=err_HYlam_GBM, fillalpha=.15, color = :brown, line=(1, [:dash]), legend = :bottomright, label = L"\textrm{HY}", dpi = 300, ylims = (0.1, 0.8), size = (600, 500))
hline!(p9_2, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p9_2, L"\textrm{Average inter-arrival}(1/\lambda)\textrm{[sec]}")
ylabel!(p9_2, L"\rho(1/\lambda)")

# savefig(p9_2, "Plots/EppsCorrection2/GBMPriceModelHYSamplingFreq.svg")

#---------------------------------------------------------------------------
## How the sampling freqency affects the overlap correction, when sampling freqency
## is high, events don't have enough time to correlate

dt = collect(1:1:100)
measured_hawkes_flattime_adj_exp_lam1 = zeros(length(dt), reps)
measured_hawkes_flattime_adj_exp_lam10 = zeros(length(dt), reps)
measured_hawkes_flattime_adj_exp_lam25 = zeros(length(dt), reps)

# Takes roughly 24 hours to compute
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
        p1_lam1 = zeros(n,1)
        p2_lam1 = zeros(n,1)
        p1_lam10 = zeros(n,1)
        p2_lam10 = zeros(n,1)
        p1_lam25 = zeros(n,1)
        p2_lam25 = zeros(n,1)
        for j in 1:n
            γ1_lam1 = maximum(filter(x-> x .<= t[j], t1_lam1))
            γ2_lam1 = maximum(filter(x-> x .<= t[j], t2_lam1))
            p1_lam1[j] = exp(P_1[Int(floor(γ1_lam1)+1), 1])
            p2_lam1[j] = exp(P_1[Int(floor(γ2_lam1)+1), 2])

            γ1_lam10 = maximum(filter(x-> x .<= t[j], t1_lam10))
            γ2_lam10 = maximum(filter(x-> x .<= t[j], t2_lam10))
            p1_lam10[j] = exp(P_1[Int(floor(γ1_lam10)+1), 1])
            p2_lam10[j] = exp(P_1[Int(floor(γ2_lam10)+1), 2])

            γ1_lam25 = maximum(filter(x-> x .<= t[j], t1_lam25))
            γ2_lam25 = maximum(filter(x-> x .<= t[j], t2_lam25))
            p1_lam25[j] = exp(P_1[Int(floor(γ1_lam25)+1), 1])
            p2_lam25[j] = exp(P_1[Int(floor(γ2_lam25)+1), 2])
        end
        adj_lam1 = flattime(dt[i], t1_lam1[2:end], t2_lam1[2:end], T)
        adj_lam10 = flattime(dt[i], t1_lam10[2:end], t2_lam10[2:end], T)
        adj_lam25 = flattime(dt[i], t1_lam25[2:end], t2_lam25[2:end], T)

        measured_lam1 = NUFFTcorrDKFGG([p1_lam1 p2_lam1], [t t])[1][1,2]
        measured_hawkes_flattime_adj_exp_lam1[i,k] = measured_lam1/adj_lam1

        measured_lam10 = NUFFTcorrDKFGG([p1_lam10 p2_lam10], [t t])[1][1,2]
        measured_hawkes_flattime_adj_exp_lam10[i,k] = measured_lam10/adj_lam10

        measured_lam25 = NUFFTcorrDKFGG([p1_lam25 p2_lam25], [t t])[1][1,2]
        measured_hawkes_flattime_adj_exp_lam25[i,k] = measured_lam25/adj_lam25
    end
end

theoretical_lam = zeros(length(dt), 1)
for i in 1:length(dt)
    theoretical_lam[i] = theoreticalEpps(dt[i], 0.015, 0.023, 0.05, 0.11)
end

# Save and Load
save("Computed Data/EppsCorrection/HawkeswDiffSamplingFreq2.jld", "measured_hawkes_flattime_adj_exp_lam1", measured_hawkes_flattime_adj_exp_lam1, "measured_hawkes_flattime_adj_exp_lam10", measured_hawkes_flattime_adj_exp_lam10,
"measured_hawkes_flattime_adj_exp_lam25", measured_hawkes_flattime_adj_exp_lam25, "theoretical_lam", theoretical_lam)

HawkeswDiffSamplingFreq = load("Computed Data/EppsCorrection/HawkeswDiffSamplingFreq2.jld")
measured_hawkes_flattime_adj_exp_lam1 = HawkeswDiffSamplingFreq["measured_hawkes_flattime_adj_exp_lam1"]
measured_hawkes_flattime_adj_exp_lam10 = HawkeswDiffSamplingFreq["measured_hawkes_flattime_adj_exp_lam10"]
measured_hawkes_flattime_adj_exp_lam25 = HawkeswDiffSamplingFreq["measured_hawkes_flattime_adj_exp_lam25"]
theoretical_lam = HawkeswDiffSamplingFreq["theoretical_lam"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_hawkes_flattime_adj_exp_lam1 = (q .* std(measured_hawkes_flattime_adj_exp_lam1, dims = 2))
err_measured_hawkes_flattime_adj_exp_lam10 = (q .* std(measured_hawkes_flattime_adj_exp_lam10, dims = 2))
err_measured_hawkes_flattime_adj_exp_lam25 = (q .* std(measured_hawkes_flattime_adj_exp_lam25, dims = 2))

p10 = plot(dt, mean(measured_hawkes_flattime_adj_exp_lam1, dims=2), ribbon=err_measured_hawkes_flattime_adj_exp_lam1, fillalpha=.3, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"1/\lambda = 1", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p10, dt, mean(measured_hawkes_flattime_adj_exp_lam10, dims=2), ribbon=err_measured_hawkes_flattime_adj_exp_lam10, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"1/\lambda = 10", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p10, dt, mean(measured_hawkes_flattime_adj_exp_lam25, dims=2), ribbon=err_measured_hawkes_flattime_adj_exp_lam25, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"1/\lambda = 25", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p10, dt, theoretical_lam, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p10, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p10, L"\Delta t\textrm{[sec]}")
ylabel!(p10, L"\rho_{\Delta t}^{ij}")

# savefig(p10, "Plots/EppsCorrection2/HawkesPriceModelwDiffSamplingFreq.svg")

#---------------------------------------------------------------------------
## K-Skip sampling
#---------------------------------------------------------------------------
# The next experiment tries to determine the underlying process using
# one set of inter-arrivals U^i and U^j by using k-skip sampling to emualate
# the sampling process with different inter-arrivals

## Hawkes price model
reps = 1    # only use 1 replication to see if it works
kskip = collect(1:1:50)
HYlam_lam1 = zeros(length(kskip), reps)
HYlam_lam1_GBM = zeros(length(kskip), reps)

# Takes roughly 2 minute to compute
for k in 1:reps
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    @showprogress "Computing..." for i in 1:length(kskip)
        t1_lam1_ind = collect(1:kskip[i]:length(t1_lam1))
        t2_lam1_ind = collect(1:kskip[i]:length(t2_lam1))

        t1_lam1_temp = t1_lam1[t1_lam1_ind]
        t2_lam1_temp = t2_lam1[t2_lam1_ind]
        # Hawkes
        P1_lam1 = exp.(P_1[Int.(floor.(t1_lam1_temp).+1), 1])
        P2_lam1 = exp.(P_1[Int.(floor.(t2_lam1_temp).+1), 2])
        # GBM
        P1_lam1_GBM = (P_GBM[Int.(floor.(t1_lam1_temp).+1), 1])
        P2_lam1_GBM = (P_GBM[Int.(floor.(t2_lam1_temp).+1), 2])


        HYlam_lam1[i,k] = HYcorr(P1_lam1,P2_lam1,t1_lam1_temp,t2_lam1_temp)[1][1,2]
        HYlam_lam1_GBM[i,k] = HYcorr(P1_lam1_GBM,P2_lam1_GBM,t1_lam1_temp,t2_lam1_temp)[1][1,2]
    end
end

# Save and Load
save("Computed Data/EppsCorrection/k_skipHY.jld", "HYlam_lam1", HYlam_lam1, "HYlam_lam1_GBM", HYlam_lam1_GBM)

k_skipHY = load("Computed Data/EppsCorrection/k_skipHY.jld")
HYlam_lam1 = k_skipHY["HYlam_lam1"]
HYlam_lam1_GBM = k_skipHY["HYlam_lam1_GBM"]


# Plot

p13 = plot(kskip, HYlam_lam1, color = :brown, line=(1, [:dash]), legend = :bottomright, label = L"\textrm{HY}", dpi = 300, ylims = (0.1, 0.8), size = (600, 500))
hline!(p13, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p13, L"\textrm{k-skip}")
ylabel!(p13, L"\rho(\textrm{k})")

# savefig(p13, "Plots/EppsCorrection2/k_skipHY_Hawkes.svg")

p14 = plot(kskip, HYlam_lam1_GBM, color = :brown, line=(1, [:dash]), legend = :bottomright, label = L"\textrm{HY}", dpi = 300, ylims = (0.1, 0.8), size = (600, 500))
hline!(p14, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p14, L"\textrm{k-skip}")
ylabel!(p14, L"\rho(\textrm{k})")

# savefig(p14, "Plots/EppsCorrection2/k_skipHY_GBM.svg")


#---------------------------------------------------------------------------
## Stochastic Volatility model with noise
#---------------------------------------------------------------------------
# Testing to see the effect of noise on the corrections
#---------------------------------------------------------------------------
# Stochastic Volatility model with exponential sampling
reps = 100

T = 3600*20
ρ = theoreticalCorr(0.023, 0.05, 0.11)

P_SV = SVwNoise(T+1, ρ)[1]
t_SV = reshape([collect(0:1:T); collect(0:1:T)], T+1, 2)

lam = 15
lam2 = 1/lam

dt = collect(1:1:400)

measured_SV_exp = zeros(length(dt), reps)
measured_SV_prevtick_exp = zeros(length(dt), reps)
measured_SV_flattime_adj_exp = zeros(length(dt), reps)
HYSV_exp = zeros(reps, 1)

# takes roughly 2 hours to compute
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
        p1 = zeros(n,1)
        p2 = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1[j] = P_SV[Int(floor(γ1)+1), 1]
            p2[j] = P_SV[Int(floor(γ2)+1), 2]
        end
        p = zeroticks([p1 p2])
        adj = flattime(dt[i], t1[2:end], t2[2:end], T)

        measured_SV_exp[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
        measured_SV_prevtick_exp[i,k] = measured_SV_exp[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
        measured_SV_flattime_adj_exp[i,k] = measured_SV_exp[i,k]/adj
    end

    P1 = P_SV[Int.(floor.(t1).+1), 1]
    P2 = P_SV[Int.(floor.(t2).+1), 2]
    HYSV_exp[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

theoretical_exp = ρ .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))

# Save and Load
save("Computed Data/EppsCorrection/SVwExpSaples2.jld", "measured_SV_exp", measured_SV_exp, "measured_SV_prevtick_exp", measured_SV_prevtick_exp,
"measured_SV_flattime_adj_exp", measured_SV_flattime_adj_exp, "HYSV_exp", HYSV_exp)

SVwExpSaples = load("Computed Data/EppsCorrection/SVwExpSaples2.jld")
measured_SV_exp = SVwExpSaples["measured_SV_exp"]
measured_SV_prevtick_exp = SVwExpSaples["measured_SV_prevtick_exp"]
measured_SV_flattime_adj_exp = SVwExpSaples["measured_SV_flattime_adj_exp"]
HYSV_exp = SVwExpSaples["HYSV_exp"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_SV_exp = (q .* std(measured_SV_exp, dims = 2))
err_measured_SV_prevtick_exp = (q .* std(measured_SV_prevtick_exp, dims = 2))
err_measured_SV_flattime_adj_exp = (q .* std(measured_SV_flattime_adj_exp, dims = 2))
err_HYSV_exp = (q .* std(HYSV_exp))

p15 = plot(dt, mean(measured_SV_exp, dims=2), ribbon=err_measured_SV_exp, fillalpha=.15, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p15, dt, mean(measured_SV_prevtick_exp, dims=2), ribbon=err_measured_SV_prevtick_exp, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p15, dt, mean(measured_SV_flattime_adj_exp, dims=2), ribbon=err_measured_SV_flattime_adj_exp, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p15, dt, theoretical_exp, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Epps}")
hline!(p15, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
hline!(p15, [mean(HYSV_exp)], ribbon=err_HYSV_exp, fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p15, L"\Delta t\textrm{[sec]}")
ylabel!(p15, L"\rho_{\Delta t}^{ij}")

# savefig(p15, "Plots/EppsCorrection2/SVPriceModelwExpSamples.svg")

#---------------------------------------------------------------------------
# Stochastic Volatility model with Hawkes sampling

measured_SV_hawkes = zeros(length(dt), reps)
measured_SV_prevtick_hawkes = zeros(length(dt), reps)
measured_SV_flattime_adj_hawkes = zeros(length(dt), reps)
HYSV_hawkes = zeros(reps, 1)

# Seed is set this way so the Hawkes processes look different
# this is due to how seeds are set in the Hawkes Simulation for reproducibility
Random.seed!(2020)
seeds = Int.(floor.(rand(reps) .* 1000000))

# takes roughly 12 hours to compute
@showprogress "Computing..." for k in 1:reps
    t = simulateHawkes([0.015;0.015], [0 0.023; 0.023 0], [0 0.11; 0.11 0], T, seed = seeds[k])

    t1 = [0;t[1]]
    t2 = [0;t[2]]

    for i in 1:length(dt)
        t = collect(0:dt[i]:T)
        n = length(t)
        p1 = zeros(n,1)
        p2 = zeros(n,1)
        for j in 1:n
            γ1 = maximum(filter(x-> x .<= t[j], t1))
            γ2 = maximum(filter(x-> x .<= t[j], t2))
            p1[j] = P_SV[Int(floor(γ1)+1), 1]
            p2[j] = P_SV[Int(floor(γ2)+1), 2]
        end
        p = zeroticks([p1 p2])
        adj = flattime(dt[i], t1[2:end], t2[2:end], T)

        measured_SV_hawkes[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
        measured_SV_prevtick_hawkes[i,k] = measured_SV_hawkes[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
        measured_SV_flattime_adj_hawkes[i,k] = measured_SV_hawkes[i,k]/adj
    end
    P1 = P_SV[Int.(floor.(t1).+1), 1]
    P2 = P_SV[Int.(floor.(t2).+1), 2]
    HYSV_hawkes[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

# Save and Load
save("Computed Data/EppsCorrection/SVwHawkesSaples2.jld", "measured_SV_hawkes", measured_SV_hawkes, "measured_SV_prevtick_hawkes", measured_SV_prevtick_hawkes,
"measured_SV_flattime_adj_hawkes", measured_SV_flattime_adj_hawkes, "HYSV_hawkes", HYSV_hawkes)

SVwHawkesSaples = load("Computed Data/EppsCorrection/SVwHawkesSaples2.jld")
measured_SV_hawkes = SVwHawkesSaples["measured_SV_hawkes"]
measured_SV_prevtick_hawkes = SVwHawkesSaples["measured_SV_prevtick_hawkes"]
measured_SV_flattime_adj_hawkes = SVwHawkesSaples["measured_SV_flattime_adj_hawkes"]
HYSV_hawkes = SVwHawkesSaples["HYSV_hawkes"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_SV_hawkes = (q .* std(measured_SV_hawkes, dims = 2))
err_measured_SV_prevtick_hawkes = (q .* std(measured_SV_prevtick_hawkes, dims = 2))
err_measured_SV_flattime_adj_hawkes = (q .* std(measured_SV_flattime_adj_hawkes, dims = 2))
err_HYSV_hawkes = (q .* std(HYSV_hawkes))

p16 = plot(dt, mean(measured_SV_hawkes, dims=2), ribbon=err_measured_SV_hawkes, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (0, 2), size = (600, 500))
plot!(p16, dt, mean(measured_SV_prevtick_hawkes, dims=2), ribbon=err_measured_SV_prevtick_hawkes, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p16, dt, mean(measured_SV_flattime_adj_hawkes, dims=2), ribbon=err_measured_SV_flattime_adj_hawkes, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p16, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
hline!(p16, [mean(HYSV_hawkes)], ribbon=err_HYSV_hawkes, fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p16, L"\Delta t\textrm{[sec]}")
ylabel!(p16, L"\rho_{\Delta t}^{ij}")

# savefig(p16, "Plots/EppsCorrection2/SVPriceModelwHawkesSamples.svg")


#---------------------------------------------------------------------------
## Experiment 1 and 3 using Stochastic Volatility model with noise
#---------------------------------------------------------------------------

## Experiment 1
lamrange = collect(1:45)
HYlam_SV = zeros(length(lamrange), reps)

# Takes roughly 10 hours to compute
@showprogress "Computing..." for k in 1:reps
    for i in 1:length(lamrange)
        lam = lamrange[i]
        Random.seed!(i+k)
        t1 = [0; rexp(T, lam)]
        t1 = cumsum(t1)
        t1 = filter((x) -> x < T, t1)

        Random.seed!(i+k+reps)
        t2 = [0; rexp(T, lam)]
        t2 = cumsum(t2)
        t2 = filter((x) -> x < T, t2)

        P1_SV = P_SV[Int.(floor.(t1).+1), 1]
        P2_SV = P_SV[Int.(floor.(t2).+1), 2]
        HYlam_SV[i,k] = HYcorr(P1_SV,P2_SV,t1,t2)[1][1,2]
    end
end

# Save and Load
save("Computed Data/EppsCorrection/HYFreq2_SV.jld", "HYlam_SV", HYlam_SV)

HYlam_SV = load("Computed Data/EppsCorrection/HYFreq2_SV.jld")["HYlam_SV"]

q = quantile.(TDist(reps-1), [0.975])

err_HYlam_SV = (q .* std(HYlam_SV, dims = 2))

p17 = plot(lamrange, mean(HYlam_SV, dims=2), ribbon=err_HYlam_SV, fillalpha=.15, color = :brown, line=(1, [:dash]), legend = :bottomright, label = L"\textrm{HY}", dpi = 300, ylims = (0.1, 0.8), size = (600, 500))
hline!(p17, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p17, L"\textrm{Average inter-arrival}(1/\lambda)\textrm{[sec]}")
ylabel!(p17, L"\rho(1/\lambda)")

# savefig(p17, "Plots/EppsCorrection2/SVPriceModelHYSamplingFreq.svg")

## Experiment 3
kskip = collect(1:1:50)
HYlam_lam1_SV = zeros(length(kskip), 1)

# Takes roughly 6 minutes to compute
for k in 1:1
    lam1 = 1
    Random.seed!(k)
    t1_lam1 = [0; rexp(T, lam1)]
    t1_lam1 = cumsum(t1_lam1)
    t1_lam1 = filter((x) -> x < T, t1_lam1)
    Random.seed!(k+reps)
    t2_lam1 = [0; rexp(T, lam1)]
    t2_lam1 = cumsum(t2_lam1)
    t2_lam1 = filter((x) -> x < T, t2_lam1)

    @showprogress "Computing..." for i in 1:length(kskip)
        t1_lam1_ind = collect(1:kskip[i]:length(t1_lam1))
        t2_lam1_ind = collect(1:kskip[i]:length(t2_lam1))

        t1_lam1_temp = t1_lam1[t1_lam1_ind]
        t2_lam1_temp = t2_lam1[t2_lam1_ind]
        # SV
        P1_lam1_SV = (P_SV[Int.(floor.(t1_lam1_temp).+1), 1])
        P2_lam1_SV = (P_SV[Int.(floor.(t2_lam1_temp).+1), 2])

        HYlam_lam1_SV[i,k] = HYcorr(P1_lam1_SV,P2_lam1_SV,t1_lam1_temp,t2_lam1_temp)[1][1,2]
    end
end

# Save and Load
save("Computed Data/EppsCorrection/k_skipHY_SV.jld", "HYlam_lam1_SV", HYlam_lam1_SV)

HYlam_lam1_SV = load("Computed Data/EppsCorrection/k_skipHY_SV.jld")["HYlam_lam1_SV"]

# Plot
p18 = plot(kskip, HYlam_lam1_SV, color = :brown, line=(1, [:dash]), legend = :bottomright, label = L"\textrm{HY}", dpi = 300, ylims = (0.1, 0.8), size = (600, 500))
hline!(p18, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p18, L"\textrm{k-skip}")
ylabel!(p18, L"\rho(\textrm{k})")

# savefig(p18, "Plots/EppsCorrection2/k_skipHY_SV.svg")

#---------------------------------------------------------------------------
## Experiment 2 using Stochastic Volatility model with noise
#---------------------------------------------------------------------------

dt = collect(1:1:100)
measured_SV_flattime_adj_exp_lam1 = zeros(length(dt), reps)
measured_SV_flattime_adj_exp_lam10 = zeros(length(dt), reps)
measured_SV_flattime_adj_exp_lam25 = zeros(length(dt), reps)

# Takes roughly 24 hours to compute
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
        p1_lam1 = zeros(n,1)
        p2_lam1 = zeros(n,1)
        p1_lam10 = zeros(n,1)
        p2_lam10 = zeros(n,1)
        p1_lam25 = zeros(n,1)
        p2_lam25 = zeros(n,1)
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
        adj_lam1 = flattime(dt[i], t1_lam1[2:end], t2_lam1[2:end], T)
        adj_lam10 = flattime(dt[i], t1_lam10[2:end], t2_lam10[2:end], T)
        adj_lam25 = flattime(dt[i], t1_lam25[2:end], t2_lam25[2:end], T)

        measured_lam1 = NUFFTcorrDKFGG([p1_lam1 p2_lam1], [t t])[1][1,2]
        measured_SV_flattime_adj_exp_lam1[i,k] = measured_lam1/adj_lam1

        measured_lam10 = NUFFTcorrDKFGG([p1_lam10 p2_lam10], [t t])[1][1,2]
        measured_SV_flattime_adj_exp_lam10[i,k] = measured_lam10/adj_lam10

        measured_lam25 = NUFFTcorrDKFGG([p1_lam25 p2_lam25], [t t])[1][1,2]
        measured_SV_flattime_adj_exp_lam25[i,k] = measured_lam25/adj_lam25
    end
end

# Save and Load
save("Computed Data/EppsCorrection/SVwDiffSamplingFreq2.jld", "measured_SV_flattime_adj_exp_lam1", measured_SV_flattime_adj_exp_lam1, "measured_SV_flattime_adj_exp_lam10", measured_SV_flattime_adj_exp_lam10,
"measured_SV_flattime_adj_exp_lam25", measured_SV_flattime_adj_exp_lam25)

SVwDiffSamplingFreq = load("Computed Data/EppsCorrection/SVwDiffSamplingFreq2.jld")
measured_SV_flattime_adj_exp_lam1 = SVwDiffSamplingFreq["measured_SV_flattime_adj_exp_lam1"]
measured_SV_flattime_adj_exp_lam10 = SVwDiffSamplingFreq["measured_SV_flattime_adj_exp_lam10"]
measured_SV_flattime_adj_exp_lam25 = SVwDiffSamplingFreq["measured_SV_flattime_adj_exp_lam25"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_SV_flattime_adj_exp_lam1 = (q .* std(measured_SV_flattime_adj_exp_lam1, dims = 2))
err_measured_SV_flattime_adj_exp_lam10 = (q .* std(measured_SV_flattime_adj_exp_lam10, dims = 2))
err_measured_SV_flattime_adj_exp_lam25 = (q .* std(measured_SV_flattime_adj_exp_lam25, dims = 2))

p20 = plot(dt, mean(measured_SV_flattime_adj_exp_lam1, dims=2), ribbon=err_measured_SV_flattime_adj_exp_lam1, fillalpha=.3, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"1/\lambda = 1", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, size = (600, 500))
plot!(p20, dt, mean(measured_SV_flattime_adj_exp_lam10, dims=2), ribbon=err_measured_SV_flattime_adj_exp_lam10, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"1/\lambda = 10", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p20, dt, mean(measured_SV_flattime_adj_exp_lam25, dims=2), ribbon=err_measured_SV_flattime_adj_exp_lam25, fillalpha=0.3, color = :green, line=(1, [:solid]), label = L"1/\lambda = 25", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p20, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p20, L"\Delta t\textrm{[sec]}")
ylabel!(p20, L"\rho_{\Delta t}^{ij}")

# savefig(p20, "Plots/EppsCorrection2/SVPriceModelwDiffSamplingFreq.svg")
