# Machine Learning with Modified Fuchs Model (Improved)
This is a repository that contains all the code used from the paper *Towards Automated Learning with Ultra-Intense Laser Systems Operating in the kHz Repetition Rate Regime*. It contains the ML model training python scripts, plotting/analysis jupyter notebooks, and dataset generation files. This is a continuation on a [previous work](https://arxiv.org/abs/2307.16036) that is currently (October 2024) under review at [Contributions to Plasma Physics](https://onlinelibrary.wiley.com/journal/15213986). 
- Please use the attached `environment.yml` or `environment_nobuilds.yml` files to create a conda environment that can be used to run all the code

## Data Set Generation
The synthetic modified Fuchs data sets are generated with the following notebook in the *datasets* directory
- `Fuchs_Data_Generation_v5.0.ipynb`
 
For Appendix B dataset generation
- `campaign_1_generation.ipynb`
- `campaign_2_generation.ipynb`

## Campaigns
Use `campaign_1_generation.cpp` and `campaign_2_generation.cpp` to generate the split campaign datasets described in Appendix B in the *campaigns* directory

## Hyperparameter Optimization
The hyperparameters were chosen with `GridSearchCV` and the relevant files are in the *hyperparameters* directory

## Training
The training scripts are done across different data splits and noise levels in the *train* directory. Additionally, the split campaign training results are found here too.

## Saving the Models
The NN and SVGP models can be saved (along with the min-max scaler transformers) using the *models* directory

## Optimization Notebooks
The optimization task from *Section 5* is found in the *optimize* directory. The saved models from the *models* directory are used for the optimization tasks outlined in the jupyter notebooks. 

## Correspondence
This code was jointly developed by Ronak Desai and [Jack Felice](https://github.com/Felice27)
- Please email `desai.458@osu.edu` for any questions
