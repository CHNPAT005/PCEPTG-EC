## Author: Patrick Chang
# Script file to investigate the QMLE and MLA using 5 days of JSE data

using LinearAlgebra, Plots, LaTeXStrings, StatsBase, Intervals, JLD, ProgressMeter, Distributions, CSV
#---------------------------------------------------------------------------

cd("/Users/patrickchang1/PCEPTG-EC")

include("../../Functions/Correlation Estimators/Dirichlet/NUFFTcorrDK-FGG.jl")
include("../../Functions/Correlation Estimators/QMLE/QMLEcorr.jl")
include("../../Functions/Correlation Estimators/MLA/KEM.jl")

#---------------------------------------------------------------------------
# Read in the data
days = ["2019-06-24"; "2019-06-25"; "2019-06-26"; "2019-06-27"; "2019-06-28"]
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

# Function to compute the correlations of QMLE
function computecorrs_QMLE(data, T = 28200, dt = collect(1:1:100))
    m = size(data)[1]
    QMLE = zeros(length(dt), m)

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
            startlam = NUFFTcorrDKFGG([p1 p2], [t t])[2][1,1]/T
            QMLE[i,k] = QMLEcorr(p1, p2, dt[i], [log(startlam); log(startlam)])
        end
    end
    return QMLE
end

# Function to compute the correlations of KEM
function computecorrs_KEM(data, T = 28200, dt = collect(1:1:100))
    m = size(data)[1]
    KEMvar = zeros(length(dt), m)

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
            starting = NUFFTcorrDKFGG([p1 p2], [t t])[2]/T
            KEMvar[i,k] = KEM([p1 p2], starting, starting, 300, 1e-5)
        end
    end
    return KEMvar
end

# Streamline everything for computing correlations of QMLE
function Empirical_QMLE(A1::Int64, A2::Int64, Fulldata)
    Data = Matrix{Float64}[] #Actually FSR/SBK
    for i in 1:length(days)
        p = Fulldata[i][:,[1;A1;A2]]
        push!(Data, p)
    end
    Data_fixed = fixdata(Data)
    res = computecorrs_QMLE(Data_fixed)
    return res
end

# Streamline everything for computing correlations of KEM
function Empirical_KEM(A1::Int64, A2::Int64, Fulldata)
    Data = Matrix{Float64}[] #Actually FSR/SBK
    for i in 1:length(days)
        p = Fulldata[i][:,[1;A1;A2]]
        push!(Data, p)
    end
    Data_fixed = fixdata(Data)
    res = computecorrs_KEM(Data_fixed)
    return res
end

#---------------------------------------------------------------------------
# Compute results for QMLE

SBKFSR_QMLE = Empirical_QMLE(7, 11, JSE_Data)
NEDABG_QMLE = Empirical_QMLE(8, 9, JSE_Data)

# Save and Load
save("Computed Data/Supp/Empirical_QMLE.jld", "SBKFSR_QMLE", SBKFSR_QMLE, "NEDABG_QMLE", NEDABG_QMLE)

ComputedResults_QMLE = load("Computed Data/Supp/Empirical_QMLE.jld")
SBKFSR_QMLE = ComputedResults_QMLE["SBKFSR_QMLE"]
NEDABG_QMLE = ComputedResults_QMLE["NEDABG_QMLE"]

# Plots
dt = collect(1:1:100)

p1 = plot(dt, mean(SBKFSR_QMLE, dims=2), legend = :bottomright, color = :blue, line=(1, [:solid]), label = L"\textrm{QMLE SBK/FSR}", marker=([:+ :d],1,0,stroke(2,:blue)), dpi = 300, ylims = (0, 0.7), size = (600, 500))
plot!(p1, dt, mean(NEDABG_QMLE, dims=2), color = :red, line=(1, [:solid]), label = L"\textrm{QMLE NED/ABG}", marker=([:circle :d],1,0,stroke(2,:red)), dpi = 300)
xlabel!(p1, L"\Delta t\textrm{[sec]}")
ylabel!(p1, L"\rho_{\Delta t}^{ij}")

# savefig(p1, "Plots/Supp/Empirical_QMLE.svg")

#---------------------------------------------------------------------------
# Compute results for KEM

SBKFSR_KEM = Empirical_KEM(7, 11, JSE_Data)
NEDABG_KEM = Empirical_KEM(8, 9, JSE_Data)

# Save and Load
save("Computed Data/Supp/Empirical_KEM.jld", "SBKFSR_KEM", SBKFSR_KEM, "NEDABG_KEM", NEDABG_KEM)

ComputedResults_KEM = load("Computed Data/Supp/Empirical_KEM.jld")
SBKFSR_KEM = ComputedResults_KEM["SBKFSR_KEM"]
NEDABG_KEM = ComputedResults_KEM["NEDABG_KEM"]

# Plots
dt = collect(1:1:100)

p2 = plot(dt, mean(SBKFSR_KEM, dims=2), legend = :bottomright, color = :blue, line=(1, [:solid]), label = L"\textrm{KEM SBK/FSR}", marker=([:+ :d],1,0,stroke(2,:blue)), dpi = 300, ylims = (0, 0.7), size = (600, 500))
plot!(p2, dt, mean(NEDABG_KEM, dims=2), color = :red, line=(1, [:solid]), label = L"\textrm{KEM NED/ABG}", marker=([:circle :d],1,0,stroke(2,:red)), dpi = 300)
xlabel!(p2, L"\Delta t\textrm{[sec]}")
ylabel!(p2, L"\rho_{\Delta t}^{ij}")

# savefig(p2, "Plots/Supp/Empirical_KEM.svg")
