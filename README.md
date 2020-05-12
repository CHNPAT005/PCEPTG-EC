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