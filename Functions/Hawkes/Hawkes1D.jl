## Author: Patrick Chang
# Function to simulate a univariate self exciting point process
# Using the method outlined in Toke and Pomponio 2012.

using Plots, LaTeXStrings, Optim

#---------------------------------------------------------------------------
### SIMUATION
# The simulation fuction is based on Dharmesh Sing's implementation from Java
#---------------------------------------------------------------------------

# lambda0 = base-line intensity (double float)
# alpha = excitation (double float)
# beta = rate of relaxation (double float)
# T = time horizon of simulation (double float)

function simulateHawkes1D(lambda0, alpha, beta, T)

    # Initialize
    if beta < alpha
        return println("WARNING: Unstable, alpha must be less than beta")
    end

    samples = []
    dlambda = 0.0
    lambda_star = lambda0
    t = 0.0

    ## First Event
    U = rand()
    s = - log(U) / lambda_star
    if s<=T
        samples = append!(samples, s)
        dlambda = alpha
        t = s
    else
        return samples
    end

    ## General Routine
    while true
        lambda_star = lambda0 + dlambda * exp(-beta * (s-t))
        U = rand()
        s = s - (log(U) / lambda_star)
        if s > T
            return samples
        end
        D = rand()
        if D <= ((lambda0 + dlambda * exp(-beta * (s-t))) / lambda_star)
            samples = append!(samples, s)
            dlambda = dlambda * exp(-beta * (s - t)) + alpha
            t = s
        end
    end
end

#---------------------------------------------------------------------------
## Supporting Functions to extract useful information from the Simulation

# Extract the Intensity fuction given the simulation path
function Intensity1D(T, history, lambda0, alpha, beta)
    l = repeat([lambda0], length(T))
    for i in 1:length(T)
        for j in 1:length(history)
            if T[i] > history[j]
                l[i] += alpha * exp(-beta*(T[i] - history[j]))
            end
        end
    end
    return l
end

#---------------------------------------------------------------------------
### CALIBRATION
#---------------------------------------------------------------------------
## Supporting function to calculate R^{nm}(l) used as a Recursion
# outlined in Toke and Pomponio 2012.

function recursion1D(history, beta)
    history = append!([0.0], history)
    N = length(history)
    R = zeros(N, 1)
    for i in 2:N
        R[i] = exp(-beta * (history[i] - history[i-1])) * (1 + R[i-1])
    end
    return R[2:end]
end

# Computes the integrated intensity from [0,T]
function Λ1D(history, T, lambda0, alpha, beta)
    Λ = lambda0 * T

    for i in 1:length(history)
        Λ += (alpha/beta) * (1 - exp(-beta * (T - history[i])))
    end
    return Λ
end

#---------------------------------------------------------------------------
## Functions to compute the log-likelihood to optimize using MLE

function loglikeHawkes1D(history, lambda0, alpha, beta, T)
    ll = T - Λ1D(history, T, lambda0, alpha, beta)
    R = recursion1D(history, beta)

    for i in 1:length(history)
        d = lambda0 + (alpha * R[i])
        ll += log(d)
    end
    return ll
end

function calibrateHawkes1D(param)
    return -loglikeHawkes1D(t, param[1], param[2], param[3], T)
end
