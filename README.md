# Discriminating between samples from correlated continuous and discrete random events

## Authors:
- Patrick Chang
- Etienne Pienaar
- Tim Gebbie

## Link to resources:

Link to paper: 

Dataset DOI: 10.25375/uct.12315092

## Steps for Replication:
- Change directories for all the files under [/Scripts/Epps](https://github.com/CHNPAT005/PCEPTG-EC/tree/master/Scripts/Epps). Currently the directories are set as: `cd("/Users/patrickchang1/PCEPTG-EC")`. Change this to where you have stored the file `PCEPTG-EC`. 

	- Run [/Scripts/Epps/EppsCorrection](https://github.com/CHNPAT005/PCEPTG-EC/blob/master/Scripts/Epps/EppsCorrection) to reproduce Figures 1-7.
	
 - To reproduce the Empirical analysis - download the processed dataset from ZivaHub and put the csv files into the folder `/Real Data`.
 	- Run [/Scripts/Epps/Empirical](https://github.com/CHNPAT005/PCEPTG-EC/blob/master/Scripts/Epps/Empirical) to reproduce Figures 8 and 9.

- We have included the plots under `/Plots` and Computed results under `/Computed Data` if one does not wish to re-run everything.

## Using the functions for other purposes:
### Hawkes

We have included a variety of functions for a M-variate Hawkes process with single exponential kernel.

#### Simulation Example

The simulation function requires 4 input variables:
- lambda0: the constant base-line intensity
- alpha: MxM matrix of alphas in the exponential kernel
- beta: MxM matrix of betas in the exponential kernel
- T: the time horizon of the simulation

```julia

include("Functions/Hawkes/Hawkes")

# Setting the parameters
lambda0 = [0.016 0.016]
alpha = [0 0.023; 0.023 0]
beta = [0 0.11; 0.11 0]
T = 3600

# Simulation
t = simulateHawkes(lambda0, alpha, beta, T)

```

#### Calibration Example

The calibration requires the user to decide on how many parameters to estimate and write a small function to initialise the input matrix of lambda0, alpha and beta and invoke the log-likelihood.

The calibration uses the Optim package in Julia.

```julia

include("Functions/Hawkes/Hawkes")

# Function to be used in optimization for the above simulation
function calibrateHawkes(param)
    lambda0 = [param[1] param[1]]
    alpha = [0 param[2]; param[2] 0]
    beta = [0 param[3]; param[3] 0]
    return -loglikeHawkes(t, lambda0, alpha, beta, T)
end

# Optimize the parameters using Optim
res = optimize(calibrateHawkes, [0.01, 0.015, 0.15])
par = Optim.minimizer(res)

```

### Estimators

The estimators include the Malliavin-Mancino estimator and the Hayashi-Yoshida estimator.
For details on usage of the Malliavin-Mancino estimator, please refer to our previous work: [Malliavin-Mancino estimators implemented with non-uniform fast Fourier transforms](https://github.com/CHNPAT005/PCEPTG-MM-NUFFT).

The Hayashi-Yoshida estimator takes in the vectors of prices (can be asynchronous), along with their associated trade times.

#### Example

```julia

include("Functions/Correlation Estimators/HY/HYcorr")
include("Functions/SDEs/GBM")

# Create some data
mu = [0.01/86400, 0.01/86400]
sigma = [0.1/86400 sqrt(0.1/86400)*0.35*sqrt(0.2/86400);
        sqrt(0.1/86400)*0.35*sqrt(0.2/86400) 0.2/86400]

P = GBM(10000, mu, sigma, seed = 10)
P1 = P[:,1] ; P2 = P[:,2]
t1= collect(1:1:10000); t2= collect(1:1:10000)

# Obtain results
output = HYcorr(P1,P2,t1,t2)

# Extract results
cor1 = output1[1]
cov1 = output1[2]

```




