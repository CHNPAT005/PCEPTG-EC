## Author: Patrick Chang
# Script file to investigate the correction of the Epps effect arising
# from asynchronous sampling, including QMLE as a comparison

using LinearAlgebra, Plots, LaTeXStrings, StatsBase, Intervals, JLD, ProgressMeter, Distributions

cd("/Users/patrickchang1/PCEPTG-EC")

include("../../Functions/Hawkes/Hawkes.jl")
include("../../Functions/SDEs/GBM.jl")
include("../../Functions/SDEs/Merton Model.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/HY/HYcorr.jl")
include("../../Functions/Correlation Estimators/QMLE/QMLEcorr.jl")

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
# GBM price model with exponential sampling
reps = 10

T = 3600
ρ = theoreticalCorr(0.023, 0.05, 0.11)

mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*ρ*sqrt(0.2/86400);
        sqrt(0.1/86400)*ρ*sqrt(0.2/86400) 0.2/86400]

P_GBM = GBM(T+1, mu, sigma)
t_GBM = reshape([collect(0:1:T); collect(0:1:T)], T+1, 2)

lam = 15
lam2 = 1/lam

dt = collect(1:1:100)

measured_GBM_exp = zeros(length(dt), reps)
measured_GBM_prevtick_exp = zeros(length(dt), reps)
measured_GBM_flattime_adj_exp = zeros(length(dt), reps)
measured_GBM_QMLE_exp = zeros(length(dt), reps)
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
        measured_GBM_QMLE_exp[i,k] = QMLEcorr(p1, p2, dt[i])
    end

    P1 = P_GBM[Int.(floor.(t1).+1), 1]
    P2 = P_GBM[Int.(floor.(t2).+1), 2]
    HYGBM_exp[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

theoretical_exp = ρ .* (1 .+ (exp.(-lam2 .* dt) .- 1) ./ (lam2 .* dt))

# Save and Load
save("Computed Data/MiniSim/GBMwExpSaples.jld", "measured_GBM_exp", measured_GBM_exp, "measured_GBM_prevtick_exp", measured_GBM_prevtick_exp,
"measured_GBM_flattime_adj_exp", measured_GBM_flattime_adj_exp, "HYGBM_exp", HYGBM_exp, "measured_GBM_QMLE_exp", measured_GBM_QMLE_exp)

GBMwExpSaples = load("Computed Data/MiniSim/GBMwExpSaples.jld")
measured_GBM_exp = GBMwExpSaples["measured_GBM_exp"]
measured_GBM_QMLE_exp = GBMwExpSaples["measured_GBM_QMLE_exp"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_GBM_exp = (q .* std(measured_GBM_exp, dims = 2))
err_measured_GBM_QMLE_exp = (q .* std(measured_GBM_QMLE_exp, dims = 2))

p1 = plot(dt, mean(measured_GBM_exp, dims=2), ribbon=err_measured_GBM_exp, fillalpha=.15, legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300)
plot!(p1, dt, mean(measured_GBM_QMLE_exp, dims=2), ribbon=err_measured_GBM_QMLE_exp, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p1, dt, theoretical_exp, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Epps}")
hline!(p1, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p1, L"\Delta t\textrm{[sec]}")
ylabel!(p1, L"\rho_{\Delta t}^{ij}")

# savefig(p1, "Plots/MiniSim/GBMPriceModelwExpSamples.svg")

#---------------------------------------------------------------------------
# GBM price model with Hawkes sampling

measured_GBM_hawkes = zeros(length(dt), reps)
measured_GBM_prevtick_hawkes = zeros(length(dt), reps)
measured_GBM_flattime_adj_hawkes = zeros(length(dt), reps)
measured_GBM_QMLE_hawkes = zeros(length(dt), reps)
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
        measured_GBM_QMLE_hawkes[i,k] = QMLEcorr(p1, p2, dt[i])
    end
    P1 = P_GBM[Int.(floor.(t1).+1), 1]
    P2 = P_GBM[Int.(floor.(t2).+1), 2]
    HYGBM_hawkes[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

# Save and Load
save("Computed Data/MiniSim/GBMwHawkesSaples.jld", "measured_GBM_hawkes", measured_GBM_hawkes, "measured_GBM_prevtick_hawkes", measured_GBM_prevtick_hawkes,
"measured_GBM_flattime_adj_hawkes", measured_GBM_flattime_adj_hawkes, "HYGBM_hawkes", HYGBM_hawkes, "measured_GBM_QMLE_hawkes", measured_GBM_QMLE_hawkes)

GBMwHawkesSaples = load("Computed Data/MiniSim/GBMwHawkesSaples.jld")
measured_GBM_hawkes = GBMwHawkesSaples["measured_GBM_hawkes"]
measured_GBM_QMLE_hawkes = GBMwHawkesSaples["measured_GBM_QMLE_hawkes"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_GBM_hawkes = (q .* std(measured_GBM_hawkes, dims = 2))
err_measured_GBM_QMLE_hawkes = (q .* std(measured_GBM_QMLE_hawkes, dims = 2))


p2 = plot(dt, mean(measured_GBM_hawkes, dims=2), ribbon=err_measured_GBM_hawkes, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (0, 2))
plot!(p2, dt, mean(measured_GBM_QMLE_hawkes, dims=2), ribbon=err_measured_GBM_QMLE_hawkes, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
hline!(p2, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Induced } \rho")
xlabel!(p2, L"\Delta t\textrm{[sec]}")
ylabel!(p2, L"\rho_{\Delta t}^{ij}")

# savefig(p2, "Plots/MiniSim/GBMPriceModelwHawkesSamples.svg")


#---------------------------------------------------------------------------
## Hawkes price model with Exponential sampling

T = 3600

ρ = theoreticalCorr(0.023, 0.05, 0.11)

par_1 = BarcyParams(0.015, 0.023, 0.05, 0.11)
lambda0_1 = par_1[1]; alpha_1 = par_1[2]; beta_1 = par_1[3]
t1 = simulateHawkes(lambda0_1, alpha_1, beta_1, T, seed = 19549293)

p1_1 = getuniformPrices(0, 1, T, t1[1], t1[2])
p2_1 = getuniformPrices(0, 1, T, t1[3], t1[4])
P_1 = [p1_1 p2_1]

t = collect(0:1:T)

measured_hawkes_exp = zeros(length(dt), reps)
measured_hawkes_prevtick_exp = zeros(length(dt), reps)
measured_hawkes_flattime_adj_exp = zeros(length(dt), reps)
measured_hawkes_QMLE_exp = zeros(length(dt), reps)
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
        measured_hawkes_QMLE_exp[i,k] = QMLEcorr(p1, p2, dt[i])
    end
    P1 = exp.(P_1[Int.(floor.(t1).+1), 1])
    P2 = exp.(P_1[Int.(floor.(t2).+1), 2])
    HYHawkes_exp[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

theoretical = zeros(length(dt), 1)
for i in 1:length(dt)
    theoretical[i] = theoreticalEpps(dt[i], 0.015, 0.023, 0.05, 0.11)
end

# Save and Load
save("Computed Data/MiniSim/HawkeswExpSaples.jld", "measured_hawkes_exp", measured_hawkes_exp, "measured_hawkes_prevtick_exp", measured_hawkes_prevtick_exp,
"measured_hawkes_flattime_adj_exp", measured_hawkes_flattime_adj_exp, "HYHawkes_exp", HYHawkes_exp, "measured_hawkes_QMLE_exp", measured_hawkes_QMLE_exp)

HawkeswExpSaples = load("Computed Data/MiniSim/HawkeswExpSaples.jld")
measured_hawkes_exp = HawkeswExpSaples["measured_hawkes_exp"]
measured_hawkes_QMLE_exp = HawkeswExpSaples["measured_hawkes_QMLE_exp"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_hawkes_exp = (q .* std(measured_hawkes_exp, dims = 2))
err_measured_hawkes_QMLE_exp = (q .* std(measured_hawkes_QMLE_exp, dims = 2))

p3 = plot(dt, mean(measured_hawkes_exp, dims=2), ribbon=err_measured_hawkes_exp, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300)
plot!(p3, dt, mean(measured_hawkes_QMLE_exp, dims=2), ribbon=err_measured_hawkes_QMLE_exp, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p3, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p3, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p3, L"\Delta t\textrm{[sec]}")
ylabel!(p3, L"\rho_{\Delta t}^{ij}")

# savefig(p3, "Plots/MiniSim/HawkesPriceModelwExpSamples.svg")

#---------------------------------------------------------------------------
## Hawkes price model with Hawkes sampling

measured_hawkes_hawkes = zeros(length(dt), reps)
measured_hawkes_prevtick_hawkes = zeros(length(dt), reps)
measured_hawkes_flattime_adj_hawkes = zeros(length(dt), reps)
measured_hawkes_QMLE_hawkes = zeros(length(dt), reps)
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
        measured_hawkes_QMLE_hawkes[i,k] = QMLEcorr(p1, p2, dt[i])
    end
    P1 = exp.(P_1[Int.(floor.(t1).+1), 1])
    P2 = exp.(P_1[Int.(floor.(t2).+1), 2])
    HYHawkes_hawkes[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
end

# Save and Load
save("Computed Data/MiniSim/HawkeswHawkesSaples.jld", "measured_hawkes_hawkes", measured_hawkes_hawkes, "measured_hawkes_prevtick_hawkes", measured_hawkes_prevtick_hawkes,
"measured_hawkes_flattime_adj_hawkes", measured_hawkes_flattime_adj_hawkes, "HYHawkes_hawkes", HYHawkes_hawkes, "measured_hawkes_QMLE_hawkes", measured_hawkes_QMLE_hawkes)

HawkeswHawkesSaples = load("Computed Data/MiniSim/HawkeswHawkesSaples.jld")
measured_hawkes_hawkes = HawkeswHawkesSaples["measured_hawkes_hawkes"]
measured_hawkes_QMLE_hawkes = HawkeswHawkesSaples["measured_hawkes_QMLE_hawkes"]

q = quantile.(TDist(reps-1), [0.975])

err_measured_hawkes_hawkes = (q .* std(measured_hawkes_hawkes, dims = 2))
err_measured_hawkes_QMLE_hawkes = (q .* std(measured_hawkes_QMLE_hawkes, dims = 2))

p4 = plot(dt, mean(measured_hawkes_hawkes, dims=2), ribbon=err_measured_hawkes_hawkes, fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (0, 2))
plot!(p4, dt, mean(err_measured_hawkes_QMLE_hawkes, dims=2), ribbon=measured_hawkes_QMLE_hawkes, fillalpha=.3, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p4, dt, theoretical, color = :black, line=(2, [:solid]), label = L"\textrm{Theoretical Synchronous Epps}")
hline!(p4, [ρ], color = :black, line=(2, [:dot]), label = L"\textrm{Limiting } \rho")
xlabel!(p4, L"\Delta t\textrm{[sec]}")
ylabel!(p4, L"\rho_{\Delta t}^{ij}")

# savefig(p4, "Plots/MiniSim/HawkesPriceModelwHawkesSamples.svg")

#---------------------------------------------------------------------------.
# Timing the estimators

numdata = collect(100:100:3600)

timing_RV = zeros(length(numdata), 1)
timing_prevtick = zeros(length(numdata), 1)
timing_flattime = zeros(length(numdata), 1)
timing_QMLE= zeros(length(numdata), 1)

@showprogress "Computing..." for i in 1:length(numdata)
    P_GBM = GBM(numdata[i], mu, sigma)
    p1 = P_GBM[:,1]; p2 = P_GBM[:,2]
    t = collect(1:1:numdata[i])

    p = @elapsed zeroticks([p1 p2])
    adj = @elapsed flattime(1, t, t, numdata[i]-1)

    timing_RV[i] = @elapsed NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
    timing_prevtick[i] = timing_RV[i] + p
    timing_flattime[i] = timing_RV[i] + adj
    timing_QMLE[i] = @elapsed QMLEcorr(p1, p2, dt[i])
end

# Save and Load
save("Computed Data/MiniSim/TimingEst.jld", "timing_RV", timing_RV, "timing_prevtick", timing_prevtick,
"timing_flattime", timing_flattime, "timing_QMLE", timing_QMLE)

TimingData = load("Computed Data/MiniSim/TimingEst.jld")
timing_RV = TimingData["timing_RV"]
timing_prevtick = TimingData["timing_prevtick"]
timing_flattime = TimingData["timing_flattime"]
timing_QMLE = TimingData["timing_QMLE"]

p5 = plot(numdata, log.(timing_RV), legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (-10,7))
plot!(p5, numdata, log.(timing_prevtick), color = :blue, line=(1, [:dash]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p5, numdata, log.(timing_flattime), color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
plot!(p5, numdata, log.(timing_QMLE), color = :purple, line=(1, [:solid]), label = L"\textrm{QMLE}", marker=([:circle :d],1,0,stroke(2,:purple)))
xlabel!(p5, L"\textrm{Number of data points } n")
ylabel!(p5, L"\textrm{Time [log(sec)]}")

# savefig(p5, "Plots/MiniSim/TimeEst.svg")
