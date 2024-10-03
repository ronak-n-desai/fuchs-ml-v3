import numpy as np
import torch
from skorch.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tqdm.auto import tqdm, trange
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import sys
sys.path.append('../')
from helpful_functions import make_datasets
torch.manual_seed(0)
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from skorch.probabilistic import GPRegressor
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, LRScheduler

noise = 10
spectrum = False
if spectrum:
    identifier = 'spectrum'
    num_outputs = 25
    output_list = ['Bin ' + str(i) for i in range(25)]
else:
    identifier = 'threeEns'
    num_outputs = 3
    output_list = ["Max Proton Energy", "Total Proton Energy", "Avg Proton Energy"]
percentagepoints = 100
mod_type = 'svgp_'
info = 'noise_'+str(noise)
desc = mod_type + info
date_str = datetime.today().strftime('%m-%d-%H-%M-%S')
result_dir = 'results/' + desc + '_' + date_str
input_list = ['Intensity', 'Target Thickness', 'Focal Distance', 'Contrast']
if not os.path.exists('results'):
    os.mkdir('results')
os.mkdir(result_dir)
filename = f'../datasets/fuchs_v5_0_seed-2_train_1525000_noise_{noise}_threeEns_.h5'
df = pd.read_hdf(filename, key='df').fillna(0)
#df.loc[:, output_list] = df.loc[:, output_list].replace(0, 1e-16)
datype = np.float32
num_inputs = 4

X = np.array(df[input_list],dtype=datype)
y = np.array(df[output_list],dtype=datype)
X_train, y_train, X_val, y_val, input_transformer, output_transformer = make_datasets(X, y, random_state=42)

train_ds = Dataset(X_train, y_train)
valid_ds = Dataset(X_val, y_val)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MultitaskSVGPModel(ApproximateGP):
    def __init__(self, inducing_points, num_latents, num_tasks):
        # Let's use a different set of inducing points for each latent function
        #inducing_points = torch.rand(num_latents, 16, 3)
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        
        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )
        
        super(MultitaskSVGPModel, self).__init__(variational_strategy)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
        
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
model = GPRegressor(
    module=MultitaskSVGPModel,
    max_epochs = 50,
    module__num_tasks = num_outputs,
    device=device,
    criterion = gpytorch.mlls.VariationalELBO,
    criterion__num_data = y_train.shape[0],
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=True,
    train_split = predefined_split(valid_ds),
    batch_size = 1024,
    lr = 0.01,
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_outputs),
    callbacks=[('early_stopping', EarlyStopping(patience=5,monitor='valid_loss'))]
)

inducing_points_size_list = [2000]
param_grid = {
    'module__num_latents': [8],
    'module__inducing_points': [torch.Tensor(X_train[:n, :]) for n in inducing_points_size_list],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, refit=False, n_jobs=1, cv=3,scoring='neg_mean_squared_error',verbose=1)
grid_result = grid.fit(X_train, y_train)
 
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Best number of latent points:", grid_result.best_params_['module__inducing_points'].size(0))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("\n%f (%f) with: %r" % (mean, stdev, param))
grid_result_df = pd.DataFrame(grid_result.cv_results_)
print("Dataframe", grid_result_df)
grid_result_df.to_csv(f'{result_dir}/grid_search_results.csv', index=False)