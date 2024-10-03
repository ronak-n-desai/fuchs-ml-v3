import numpy as np
import torch
from skorch.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet, NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
#import ipywidgets
from tqdm.auto import tqdm, trange
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from skorch.callbacks import EarlyStopping, LRScheduler
import sys
sys.path.append('../')
from helpful_functions import InputLogTransformer, OutputLogTransformer, build_neural_network, make_datasets
torch.manual_seed(0)

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
mod_type = 'nn_'
info = 'noise_'+str(noise)
desc = mod_type + info
date_str = datetime.today().strftime('%m-%d-%H-%M-%S')
if not os.path.exists('results'):
    os.mkdir('results')
result_dir = 'results/' + desc + '_' + date_str
input_list = ['Intensity', 'Target Thickness', 'Focal Distance', 'Contrast']
os.mkdir(result_dir)
filename = f'../datasets/fuchs_v5_0_seed-2_train_1525000_noise_{noise}_threeEns_.h5'
df = pd.read_hdf(filename, key='df')
datype = np.float32
num_inputs = 4

X = np.array(df[input_list],dtype=datype)
y = np.array(df[output_list],dtype=datype)
X_train, y_train, X_val, y_val, input_transformer, output_transformer = make_datasets(X, y, random_state=42)

device = 'cuda:0' if torch.cuda.is_available else 'cpu'

class TestingModel(nn.Module):
    '''
    Model of variable architecture for hyperparameter tuning
    '''
    def __init__(self, n_hidden = 1,n_neurons=64,activation=nn.LeakyReLU(), n_outputs = 25):
        super().__init__()
        self.norms = []
        self.layers = []
        self.acts = []
        self.norm0 = nn.BatchNorm1d(4)
        self.layer0 = nn.Linear(4,n_neurons)
        for i in range(1,n_hidden+1):
            self.norms.append(nn.BatchNorm1d(n_neurons))
            self.acts.append(activation)
            self.add_module(f"norm{i}", self.norms[-1])
            self.add_module(f"act{i}", self.acts[-1])
            if (i != n_hidden):
                self.layers.append(nn.Linear(n_neurons, n_neurons))
                self.add_module(f"layer{i}", self.layers[-1])
        self.output = nn.Linear(n_neurons, n_outputs)

    def forward(self, x):
        '''
          Forward pass
        '''
        x = self.layer0(self.norm0(x))
        for norm, layer, act in zip(self.norms, self.layers, self.acts):
            x = act(layer(norm(x)))
        return self.output(x)
    
model = NeuralNetRegressor(
    module=TestingModel,
    max_epochs = 50,
    module__n_outputs = num_outputs,
    #module__n_hidden=2,
    #module__n_neurons = 64,
    #module__activation=nn.LeakyReLU(),
    device=device,
    criterion = nn.MSELoss(),
    #optimizer__lr = 1e-3,
    iterator_train__shuffle=True,
    #train_split = None,
    callbacks=[('early_stopping', EarlyStopping(patience=7,monitor='valid_loss')),
    ('lr_scheduler', LRScheduler(policy='ExponentialLR',gamma=0.9))]
)

param_grid = {
    'module__n_hidden':[8],
    'module__n_neurons':[128],
    'module__activation':[nn.LeakyReLU()],
    'optimizer':[optim.Adam],
    'optimizer__lr':[1e-2],
    'callbacks__lr_scheduler__gamma':[0.90, 0.95],
    'callbacks__early_stopping__patience':[10],
    'batch_size':[2**10, 2**12, 2**14]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, refit=False, n_jobs=1, cv=3,scoring='neg_mean_squared_error',verbose=1)
grid_result = grid.fit(X_train, y_train)
 
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("\n%f (%f) with: %r" % (mean, stdev, param))
grid_result_df = pd.DataFrame(grid_result.cv_results_)
print("Dataframe", grid_result_df)
grid_result_df.to_csv(f'{result_dir}/grid_search_results.csv', index=False)
