## Author: Patrick Chang
# Script file to investigate the correction of the Epps effect arising
# from asynchronous sampling on 40 days of JSE data

using LinearAlgebra, Plots, LaTeXStrings, StatsBase, Intervals, JLD, ProgressMeter, Distributions, CSV
#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-EC")

include("../../Functions/Hawkes/Hawkes.jl")
include("../../Functions/SDEs/GBM.jl")
include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/HY/HYcorr.jl")

#---------------------------------------------------------------------------
## RV correction
#---------------------------------------------------------------------------

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

#---------------------------------------------------------------------------
# Read in the data
days = ["2019-05-02"; "2019-05-03"; "2019-05-06"; "2019-05-07"; "2019-05-09"; "2019-05-10"; "2019-05-13"; "2019-05-14";
        "2019-05-15"; "2019-05-16"; "2019-05-17"; "2019-05-20"; "2019-05-21"; "2019-05-22"; "2019-05-23"; "2019-05-24";
        "2019-05-27"; "2019-05-28"; "2019-05-29"; "2019-05-30"; "2019-05-31"; "2019-06-03"; "2019-06-04"; "2019-06-05";
        "2019-06-06"; "2019-06-07"; "2019-06-10"; "2019-06-11"; "2019-06-12"; "2019-06-13"; "2019-06-14"; "2019-06-18";
        "2019-06-19"; "2019-06-20"; "2019-06-21"; "2019-06-24"; "2019-06-25"; "2019-06-26"; "2019-06-27"; "2019-06-28"]

JSE_Data = Matrix{Float64}[]
for i in 1:length(days)
    p = CSV.read("Real Data/JSE_"*days[i]*".csv")
    p = convert(Matrix, p[:,2:12])
    push!(JSE_Data, p)
end

# Function to clean the data such that both assets start when the slower
# asset makes its first trade. Using Previous tick interpolation to interpolate
# for the asset which traded first, if the faster asset does not trade at the
# same time as the first trade of the slower asset.
function fixdata(data)
    m = size(data)[1]
    Fixed_Data = Matrix{Float64}[]
    # Fixed_Data = Vector{Vector{Float64}}()
    for i in 1:m
        n = size(data[i])[1]
        indA1 = findall(!isnan, data[i][:,2])
        indA2 = findall(!isnan, data[i][:,3])
        indA1_min = minimum(indA1)
        indA2_min = minimum(indA2)

        if indA1_min==indA2_min
            temp_data = data[i][indA1_min:end,:]
        end

        if indA1 > indA2
            if isnan(data[i][indA1_min,3])
                data[i][indA1_min,3] = data[i][maximum(filter(x-> x < indA1_min, indA2)),3]
            end
            temp_data = data[i][indA1_min:end,:]
        else
            if isnan(data[i][indA2_min,2])
                data[i][indA2_min,2] = data[i][maximum(filter(x-> x < indA2_min, indA1)),2]
            end
            temp_data = data[i][indA2_min:end,:]
        end
        time = temp_data[:,1]
        mintime = minimum(time)
        time = time .- mintime
        temp_data[:,1] = time

        push!(Fixed_Data, temp_data)
    end
    return Fixed_Data
end

# Function to compute the correlations and corrections
function computecorrs(data, T = 28200, dt = collect(1:1:400))
    m = size(data)[1]
    measured = zeros(length(dt), m)
    prevtick = zeros(length(dt), m)
    overlap = zeros(length(dt), m)
    HY = zeros(m, 1)

    @showprogress "Computing..." for k in 1:m
        temp = data[k]
        t1 = temp[findall(!isnan, temp[:,2]),1]
        t2 = temp[findall(!isnan, temp[:,3]),1]
        for i in 1:length(dt)
            t = collect(0:dt[i]:T)
            n = length(t)
            p1 = zeros(n,1)
            p2 = zeros(n,1)
            for j in 1:n
                inds = findall(x -> x .<= t[j], temp[:,1])
                p1[j] = filter(!isnan, temp[inds,2])[end]
                p2[j] = filter(!isnan, temp[inds,3])[end]
            end
            p = zeroticks([p1 p2])
            adj = flattime(dt[i], t1[2:end], t2[2:end], T)

            measured[i,k] = NUFFTcorrDKFGG([p1 p2], [t t])[1][1,2]
            prevtick[i,k] = measured[i,k] * ((1-p[1]*p[2]) / ((1-p[1])*(1-p[2])))
            overlap[i,k] = measured[i,k]/adj
        end
        P1 = filter(!isnan, temp[:,2])
        P2 = filter(!isnan, temp[:,3])
        HY[k] = HYcorr(P1,P2,t1,t2)[1][1,2]
    end
    return measured, prevtick, overlap, HY
end

# Function to compute the HY estimates using k-skip sampling
function computecorrs_kskip(data, T = 28200, kskip = collect(1:1:50))
    m = size(data)[1]
    HY = zeros(m, length(kskip))

    @showprogress "Computing..." for k in 1:m
        temp = data[k]
        t1 = temp[findall(!isnan, temp[:,2]),1]
        t2 = temp[findall(!isnan, temp[:,3]),1]

        P1 = filter(!isnan, temp[:,2])
        P2 = filter(!isnan, temp[:,3])
        for i in 1:length(kskip)
            t1_ind = collect(1:kskip[i]:length(t1))
            t2_ind = collect(1:kskip[i]:length(t2))

            t1_temp = t1[t1_ind]
            t2_temp = t2[t2_ind]

            P1_temp = P1[t1_ind]
            P2_temp = P2[t2_ind]

            HY[k,i] = HYcorr(P1_temp,P2_temp,t1_temp,t2_temp)[1][1,2]
        end
    end
    return HY
end

# Streamline everything for computing correlations and corrections
function Empirical(A1::Int64, A2::Int64, Fulldata)
    Data = Matrix{Float64}[] #Actually FSR/SBK
    for i in 1:length(days)
        p = Fulldata[i][:,[1;A1;A2]]
        push!(Data, p)
    end
    Data_fixed = fixdata(Data)
    res = computecorrs(Data_fixed)
    return res
end

# Streamline everything for computing k-skip HY estimates
function Empirical_kskip(A1::Int64, A2::Int64, Fulldata)
    Data = Matrix{Float64}[] #Actually FSR/SBK
    for i in 1:length(days)
        p = Fulldata[i][:,[1;A1;A2]]
        push!(Data, p)
    end
    Data_fixed = fixdata(Data)
    res = computecorrs_kskip(Data_fixed)
    return res
end
#---------------------------------------------------------------------------
# Compute results for correlations and corrections

# Each take roughly 30 min
SBKFSR = Empirical(7, 11, JSE_Data)
NEDABG = Empirical(8, 9, JSE_Data)

# Save and Load
save("Computed Data/EppsCorrection/Empirical_SBKFSR.jld", "SBKFSR_RV", SBKFSR[1], "SBKFSR_FT", SBKFSR[2], "SBKFSR_OC", SBKFSR[3], "SBKFSR_HY", SBKFSR[4])
save("Computed Data/EppsCorrection/Empirical_NEDABG.jld", "NEDABG_RV", NEDABG[1], "NEDABG_FT", NEDABG[2], "NEDABG_OC", NEDABG[3], "NEDABG_HY", NEDABG[4])

ComputedResults_SBKFSR = load("Computed Data/EppsCorrection/Empirical_SBKFSR.jld")
SBKFSR_RV = ComputedResults_SBKFSR["SBKFSR_RV"]
SBKFSR_FT = ComputedResults_SBKFSR["SBKFSR_FT"]
SBKFSR_OC = ComputedResults_SBKFSR["SBKFSR_OC"]
SBKFSR_HY = ComputedResults_SBKFSR["SBKFSR_HY"]

ComputedResults_NEDABG = load("Computed Data/EppsCorrection/Empirical_NEDABG.jld")
NEDABG_RV = ComputedResults_NEDABG["NEDABG_RV"]
NEDABG_FT = ComputedResults_NEDABG["NEDABG_FT"]
NEDABG_OC = ComputedResults_NEDABG["NEDABG_OC"]
NEDABG_HY = ComputedResults_NEDABG["NEDABG_HY"]

# Plots
dt = collect(1:1:400)
m = size(JSE_Data)[1]
q = quantile.(TDist(m-1), [0.975])

p1 = plot(dt, mean(SBKFSR_RV, dims=2), ribbon=(q .* std(SBKFSR_RV, dims = 2)), fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (-0.1, 1.55), size = (600, 500))
plot!(p1, dt, mean(SBKFSR_FT, dims=2), ribbon=(q .* std(SBKFSR_FT, dims = 2)), fillalpha=.15, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p1, dt, mean(SBKFSR_OC, dims=2), ribbon=(q .* std(SBKFSR_OC, dims = 2)), fillalpha=.15, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p1, [mean(SBKFSR_HY)], ribbon=(q .* std(SBKFSR_HY, dims = 1)), fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p1, L"\Delta t\textrm{[sec]}")
ylabel!(p1, L"\rho_{\Delta t}^{ij}")

# savefig(p1, "Plots/Empirical/SBKFSRcorrection.svg")

p2 = plot(dt, mean(NEDABG_RV, dims=2), ribbon=(q .* std(NEDABG_RV, dims = 2)), fillalpha=.15, legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (-0.1, 1.55), size = (600, 500))
plot!(p2, dt, mean(NEDABG_FT, dims=2), ribbon=(q .* std(NEDABG_FT, dims = 2)), fillalpha=.15, color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p2, dt, mean(NEDABG_OC, dims=2), ribbon=(q .* std(NEDABG_OC, dims = 2)), fillalpha=.15, color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p2, [mean(NEDABG_HY)], ribbon=(q .* std(NEDABG_HY, dims = 1)), fillalpha=.15, color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p2, L"\Delta t\textrm{[sec]}")
ylabel!(p2, L"\rho_{\Delta t}^{ij}")

# savefig(p2, "Plots/Empirical/NEDABGcorrection.svg")

p3 = plot(dt, mean(NEDABG_RV, dims=2) ./ maximum(mean(NEDABG_RV, dims=2)), legend = :bottomright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured NED/ABG}", marker=([:+ :d],2,0,stroke(2,:red)), dpi = 300, ylims = (0, 1.2), size = (600, 500))
plot!(p3, dt, mean(NEDABG_OC, dims=2) ./ maximum(mean(NEDABG_RV, dims=2)), color = :red, line=(1.5, [:dot]), label = L"\textrm{Overlap correction NED/ABG}", marker=([:circle :d],2,0,stroke(2,:red)))
plot!(p3, dt, mean(SBKFSR_RV, dims=2) ./ maximum(mean(SBKFSR_RV, dims=2)), color = :blue, line=(1, [:solid]), label = L"\textrm{Measured SBK/FSR}", marker=([:+ :d],2,0,stroke(2,:blue)))
plot!(p3, dt, mean(SBKFSR_OC, dims=2) ./ maximum(mean(SBKFSR_RV, dims=2)), color = :blue, line=(1.5, [:dot]), label = L"\textrm{Overlap correction SBK/FSR}", marker=([:circle :d],2,0,stroke(2,:blue)))
xlabel!(p3, L"\Delta t\textrm{[sec]}")
ylabel!(p3, L"\rho_{\Delta t}^{ij}")

# savefig(p3, "Plots/Empirical/EmpComparison.svg")


# No error bars
p1 = plot(dt, mean(SBKFSR_RV, dims=2), legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (-0, 2.5), size = (600, 500))
plot!(p1, dt, mean(SBKFSR_FT, dims=2), color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p1, dt, mean(SBKFSR_OC, dims=2), color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p1, [mean(SBKFSR_HY)], color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p1, L"\Delta t\textrm{[sec]}")
ylabel!(p1, L"\rho(\Delta t)")

# savefig(p1, "Plots/Empirical/SBKFSRcorrectionNOerrorbars.svg")

p2 = plot(dt, mean(NEDABG_RV, dims=2), legend = :topright, color = :red, line=(1, [:solid]), label = L"\textrm{Measured}", marker=([:+ :d],1,0,stroke(2,:red)), dpi = 300, ylims = (-0, 2.5), size = (600, 500))
plot!(p2, dt, mean(NEDABG_FT, dims=2), color = :blue, line=(1, [:solid]), label = L"\textrm{Flat trade correction}", marker=([:x :d],1,0,stroke(2,:blue)))
plot!(p2, dt, mean(NEDABG_OC, dims=2), color = :green, line=(1, [:solid]), label = L"\textrm{Overlap correction}", marker=([:circle :d],1,0,stroke(2,:green)))
hline!(p2, [mean(NEDABG_HY)], color = :brown, line=(1, [:dash]), label = L"\textrm{HY}")
xlabel!(p2, L"\Delta t\textrm{[sec]}")
ylabel!(p2, L"\rho(\Delta t)")

# savefig(p2, "Plots/Empirical/NEDABGcorrectionNOerrorbars.svg")

#---------------------------------------------------------------------------
# Compute results for k-skip HY

SBKFSR_kskip = Empirical_kskip(7, 11, JSE_Data)
NEDABG_kskip = Empirical_kskip(8, 9, JSE_Data)

# Save and Load
save("Computed Data/EppsCorrection/Empirical_kskip.jld", "SBKFSR_kskip", SBKFSR_kskip, "NEDABG_kskip", NEDABG_kskip)

ComputedResults_kskip = load("Computed Data/EppsCorrection/Empirical_kskip.jld")
SBKFSR_kskip = ComputedResults_kskip["SBKFSR_kskip"]
NEDABG_kskip = ComputedResults_kskip["NEDABG_kskip"]

# Plots
kskip = collect(1:1:50)

p4 = plot(kskip, mean(SBKFSR_kskip, dims=1)', legend = :bottomright, color = :blue, line=(1, [:solid]), label = L"\textrm{HY SBK/FSR}", marker=([:+ :d],1,0,stroke(2,:blue)), dpi = 300, size = (600, 500))
plot!(p4, kskip, mean(NEDABG_kskip, dims=1)', color = :red, line=(1, [:solid]), label = L"\textrm{HY NED/ABG}", marker=([:circle :d],1,0,stroke(2,:red)), dpi = 300)
xlabel!(p4, L"\textrm{k-skip}")
ylabel!(p4, L"\rho(\textrm{k})")

# savefig(p4, "Plots/Empirical/Empirical_kskip.svg")
