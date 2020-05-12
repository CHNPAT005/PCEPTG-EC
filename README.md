# Discriminating between samples from correlated continuous and discrete random events

## Authors:
- Patrick Chang
- Etienne Pienaar
- Tim Gebbie

## Link to resources:

Link to paper: https://arxiv.org/abs/2003.02842

Link to the Dataset: [Link](https://zivahub.uct.ac.za/articles/Malliavin-Mancino_estimators_implemented_with_the_non-uniform_fast_Fourier_transform_Dataset/11903442)

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

include("../Functions/Hawkes/Hawkes")

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

include("../Functions/Hawkes/Hawkes")

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

