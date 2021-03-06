## Author: Patrick Chang
# Script file to simulate multivariate Merton Model
# includes an example to demonstrate the code

#------------------------------------------------------------------------------

## Building the Merton Model using the method of Paul Glasserman in his book
#   Monte Carlo Methods in Financial Engineering
#   This method allows me to build the path in 1 unit time increments
#   and there is no compensator in this method

using Random
using LinearAlgebra
using Distributions
using PoissonRandom
using Plots

function Merton(n, mu, sigma, lambda, a, b; kwargs...)
    # n - simulation length
    # mu - vector input of the drift component for the underlying GBM
    # sigma - covariance matrix of the stocks
    # lambda - vector input of the poisson process intensity
    # a - vector input of mean of normal jump size
    # b - vector input of std dev of normal jump size

    # all inputs must have appropriate dimensions

    k = size(sigma)[1]

    kwargs = Dict(kwargs)

    if haskey(kwargs, :startprice)
        startprice = kwargs[:startprice]
    else
        startprice = fill(100.0, (k,1))
    end

    if haskey(kwargs, :seed)
        seed = kwargs[:seed]
    else
        seed = 1
    end

    mu = reshape(mu, k, 1)
    sigma = reshape(sigma, k, k)
    sigma2 = reshape(diag(sigma), k, 1)
    a = reshape(a, k, 1)
    b = reshape(b, k, 1)

    X = zeros(n, k)
    X[1,:] = log.(startprice)

    A = cholesky(sigma).L
    d = mu - sigma2./2 - lambda.*(exp.(a - 0.5*b.^2) .- 1)

    Random.seed!(seed)
    Z = randn(k, n-1)

    for i in 2:n
        M = zeros(k, 1)
        N = zeros(k, 1)

        for j in 1:k
            Random.seed!(seed+j+i+n)
            N[j] = pois_rand(lambda[j])
        end

        Random.seed!(seed+i)
        Z2 = randn(k)
        M = a.*N + b.*sqrt.(N).*Z2
        z = Z[:,i-1]
        X[i,:] = X[i-1,:] + d + A*z + M
    end
    return exp.(X)
end

#------------------------------------------------------------------------------

# n = 500
# mu = [0.01/86400, 0.01/86400]
# sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
#         sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]
# a = [0.0, 0.0]
# b= [100/86400, 100/86400]
# lambda = [0.2, 1]
#
# P1 = Merton(n, mu, sigma, lambda, a, b)
# P2 = Merton(n, mu, sigma, lambda, a, b, seed = 2)
#
# # important thing to note about plotting is that [mxn] matrix is m series with n data points
# p1 = plot(1:n, P1, label = ["Line 1" "Line 2"])
# title!(p1, "Correlated Merton Model")
# xlabel!(p1, "Seconds")
# ylabel!(p1, "Price")
#
# p2 = plot(1:n, P2, label = ["Line 1" "Line 2"])
# title!(p2, "Correlated Merton Model")
# xlabel!(p2, "Seconds")
# ylabel!(p2, "Price")
#
# plot(p1, p2, layout = (1,2), legend = false, lw = 2)
#
# #plt = plot(1, P[1,:], xlim = (0,500), ylim = (94, 102), title = "Correlated GBM", legend = false)
# @gif for i in 1:500
#     plot(1:i, P1[1:i,:], xlim = (0,500), ylim = (94, 102), title = "Correlated Merton Model", legend = false)
# end every 10
