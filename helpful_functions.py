import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet, NeuralNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.auto import tqdm, trange
from sklearn.base import BaseEstimator, TransformerMixin
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import Callback
if torch.cuda.is_available():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from skorch.probabilistic import GPRegressor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_transform(X): # transforms first and last columns of data.  Alternatively, this could be implemented via operations on DataFrames
    X1 = X.copy()
    X1[:, 0] = np.log(X1[:, 0])
    X1[:, -1] = np.log(X1[:, -1])
    return X1
def log_inverse(X):
    X1 = X.copy()
    X1[:, 0] = np.exp(X1[:,0])
    X1[:, -1] = np.exp(X1[:, -1])
    return X1

class OutputLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = StandardScaler()
    def fit(self, y):
        y_copy = y.copy()
        y_copy = np.log(y_copy)
        self._estimator.fit(y_copy)
        
        return self
    def transform(self, y):
        y_copy = y.copy()
        y_copy = np.log(y_copy)
        return self._estimator.transform(y_copy)
    def inverse_transform(self, y):
        y_copy = y.copy()
        y_reverse = np.exp(self._estimator.inverse_transform(y_copy))
        
        return y_reverse

class InputLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = StandardScaler()
    def fit(self, X):
        X_copy = X.copy()
        X_copy = log_transform(X_copy)
        self._estimator.fit(X_copy)
        
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy = log_transform(X_copy)
        return self._estimator.transform(X_copy)
    def inverse_transform(self, X):
        X_copy = X.copy()
        X_reverse = log_inverse(self._estimator.inverse_transform(X_copy))
        
        return X_reverse
    
class OutputLogPolyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = MinMaxScaler()
    def fit(self, y):
        y_copy = y.copy()
        y_copy = np.log(y_copy)
        self._estimator.fit(y_copy)
        
        return self
    def transform(self, y):
        y_copy = y.copy()
        y_copy = np.log(y_copy)
        return self._estimator.transform(y_copy)
    def inverse_transform(self, y):
        y_copy = y.copy()
        y_reverse = np.exp(self._estimator.inverse_transform(y_copy))
        
        return y_reverse

class InputLogPolyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = MinMaxScaler()
    def fit(self, X):
        X_copy = X.copy()
        X_copy = log_transform(X_copy)
        self._estimator.fit(X_copy)
        
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy = log_transform(X_copy)
        return self._estimator.transform(X_copy)
    def inverse_transform(self, X):
        X_copy = X.copy()
        X_reverse = log_inverse(self._estimator.inverse_transform(X_copy))
        
        return X_reverse
    
class LDIAModel(nn.Module):
    '''
    Perceptron Model of variable architecture for hyperparameter tuning
    '''
    def __init__(self, n_hidden = 1,n_neurons=64,activation=nn.LeakyReLU(),n_inputs=4, n_outputs=25):
        super().__init__()
        self.norms = []
        self.layers = []
        self.acts = []
        self.norm0 = nn.BatchNorm1d(n_inputs)
        self.layer0 = nn.Linear(n_inputs,n_neurons)
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
    
def build_neural_network(max_epochs=50, n_hidden=3, n_neurons=64, n_inputs = 4, n_outputs = 25, activation=nn.LeakyReLU(), device=torch.device('cuda'),loss_fn=nn.MSELoss(), optimizer=optim.Adam, lr=1e-3, shuffled=True, batch_size=1024, patience=5, gamma=0.85,valid_ds=None,compiled=False):
    return NeuralNetRegressor(
    module=LDIAModel,
    max_epochs = max_epochs,
    module__n_hidden=n_hidden,
    module__n_neurons = n_neurons,
    module__n_inputs = n_inputs,
    module__n_outputs = n_outputs,
    module__activation=activation,
    device=device,
    criterion = loss_fn,
    optimizer=optimizer,
    optimizer__lr = lr,
    iterator_train__shuffle=shuffled,
    batch_size=batch_size,
    callbacks=[('early_stopping', EarlyStopping(patience=patience,monitor='valid_loss')),
    ('lr_scheduler', LRScheduler(policy='ExponentialLR',gamma=gamma))],
    train_split=predefined_split(valid_ds),
    compile = compiled
)

def make_datasets(X, y, *, train_size=0.8, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size,random_state=random_state)
    input_transformer = InputLogTransformer()
    output_transformer = OutputLogTransformer()
    X_train = input_transformer.fit_transform(X_train)
    X_val = input_transformer.transform(X_val)
    y_train = output_transformer.fit_transform(y_train)
    y_val = output_transformer.transform(y_val)
    #train_ds = Dataset(X_train, y_train)
    #valid_ds = Dataset(X_val, y_val)
    return X_train, y_train, X_val, y_val, input_transformer, output_transformer

def make_poly_datasets(X, y, *, train_size=0.8, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size,random_state=random_state)
    input_transformer = InputLogPolyTransformer()
    output_transformer = OutputLogPolyTransformer()
    X_train = input_transformer.fit_transform(X_train)
    X_val = input_transformer.transform(X_val)
    y_train = output_transformer.fit_transform(y_train)
    y_val = output_transformer.transform(y_val)
    #train_ds = Dataset(X_train, y_train)
    #valid_ds = Dataset(X_val, y_val)
    return X_train, y_train, X_val, y_val, input_transformer, output_transformer

class GPUMemoryLogger(Callback):
    def __init__(self, verbose=False):
        self.gpu_memory = []
        self.verbose = verbose

    def on_epoch_end(self, net, **kwargs):
        memory_stats = torch.cuda.memory_stats()
        # Calculate available GPU memory
        total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 **2)
        gpu_memory_usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        available_memory = total_memory - gpu_memory_usage
        self.gpu_memory.append(gpu_memory_usage)
        if self.verbose:
            print(f"GPU memory usage: {gpu_memory_usage:.2f} MB")
            print(f"Available memory: {available_memory:.2f} MB")
            #print("Available memory: {:.2f} MB".format(available_memory))
    def get_memory_usage(self):
        return self.gpu_memory
    
def correct_predictions(y_train_true, y_train_pred, y_test_pred):
    """
    Corrects predictions for log-transformed data due to tendency to underpredict because
    of how log-transformation shifts the mean of the data.

    Parameters
    ----------
    y_train_true : np.ndarray
        True values of the training data.
    y_train_pred : np.ndarray
        Predicted values of the training data.
    y_test_pred : np.ndarray
        Predicted values of the test data.

    Returns
    -------
    y_test_corrected : np.ndarray
        Corrected test predictions.
    """
    n_outputs = y_train_true.shape[1] # number of output dimensions
    correction_factors = [None]*n_outputs
    for i in range(n_outputs):
        correction_factors[i] = np.mean(y_train_true[:,i]/y_train_pred[:,i])
    y_test_corrected = np.zeros(y_test_pred.shape)
    for i in range(n_outputs):
        y_test_corrected[:,i] = y_test_pred[:,i]*correction_factors[i]
    return y_test_corrected
    # Possible ideas: check if the corrected data has lower error.  Possibly implement as class that automatically makes correction factors 1
    # if the corrected data has higher error.

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