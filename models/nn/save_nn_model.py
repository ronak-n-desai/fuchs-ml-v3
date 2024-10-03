import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet, NeuralNetRegressor
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.auto import tqdm, trange
from sklearn.base import BaseEstimator, TransformerMixin
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import joblib
import sys
import os
sys.path.append('../../')
from helpful_functions import InputLogTransformer, OutputLogTransformer, build_neural_network, make_datasets
import time
import joblib
from datetime import datetime

spectrum = False
noise = 30
if spectrum:
    num_outputs = 25
    identifier = 'spectrum'
    output_list = ['Bin ' + str(i) for i in range(25)] # training outputs
else:
    num_outputs = 3
    identifier = 'threeEns'
    output_list = ['Max Proton Energy', 'Total Proton Energy', 'Avg Proton Energy']

df = pd.read_hdf(f'../../datasets/fuchs_v5_0_seed-2_train_1525000_noise_{noise}_threeEns_.h5', key='df').fillna(0)
test_df = pd.read_hdf(f'../../datasets/fuchs_v5_0_seed-2_test_250000_noise_0_{identifier}_.h5', key = 'df').fillna(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datype = np.float32
num_inputs = 4
num_outputs = 3
batch_size = 2**13
patience=10

input_list = ['Intensity', 'Target Thickness', 'Focal Distance', 'Contrast'] # independent variables

df = df.sample(frac=1,random_state=42).reset_index(drop=True)
X = np.array(df[input_list],dtype=datype)
y = np.array(df[output_list],dtype=datype)
data_fractions = np.linspace(0, 1, 7)[1:]
num_pts_tot = X.shape[0]

X_test = np.array(test_df[input_list],dtype=datype)
y_test = np.array(test_df[output_list],dtype=datype)

# Network architecture variables determined by hyperparameter optimization
n_hidden = 12
n_neurons = 64
lr = 1e-2
gamma = 0.9

np.random.seed(42)
torch.manual_seed(42)
start_time = time.time()
num_pts = num_pts_tot
num_train_pts = int(0.8 * num_pts_tot + 0.5)
print("training", num_train_pts)
X_train, y_train, X_val, y_val, input_transformer, output_transformer = make_datasets(X[:num_pts, :], y[:num_pts, :], random_state=42)
train_ds = Dataset(X_train, y_train)
valid_ds = Dataset(X_val, y_val)
model = build_neural_network(n_hidden = n_hidden, n_neurons = n_neurons, lr=lr, gamma=gamma, valid_ds=valid_ds,device=device, n_inputs = num_inputs, n_outputs = num_outputs, batch_size=batch_size,patience=patience)
model.fit(train_ds, y=None)
y_pred = output_transformer.inverse_transform(model.predict(valid_ds.X))
y_train_pred = output_transformer.inverse_transform(model.predict(train_ds.X))
y_train_true = output_transformer.inverse_transform(y_train)
correction_factors = []
for i in range(num_outputs):
    α = np.mean(y_train_true[:, i]/y_train_pred[:, i])
    correction_factors.append(α)
y_train_pred_corrected = y_train_pred.copy()
for i in range(num_outputs):
    y_train_pred_corrected[:, i] *= correction_factors[i]
X_train = input_transformer.inverse_transform(train_ds.X)
X_val = input_transformer.inverse_transform(valid_ds.X)
y_train = output_transformer.inverse_transform(train_ds.y)
y_val = output_transformer.inverse_transform(valid_ds.y)
y_train_pred = output_transformer.inverse_transform(model.predict(train_ds.X))
X_test_scaled = input_transformer.transform(X_test)
y_test_pred = output_transformer.inverse_transform(model.predict(X_test_scaled))
y_test_pred_corrected = y_test_pred.copy()
for i in range(num_outputs):
    y_test_pred_corrected[:, i] *= correction_factors[i]
print("Model performance on training data:", mape(y_train, y_train_pred, multioutput='raw_values')*100)
print("Model performance on validation data", mape(y_val, y_pred, multioutput='raw_values')*100)
mape_uncorrected = mape(y_test, y_test_pred, multioutput='raw_values')*100
mse_uncorrected = mse(y_test, y_test_pred, multioutput='raw_values')
print("Model performance on blind randomized testing data:", mape_uncorrected)
mape_corrected = mape(y_test, y_test_pred_corrected, multioutput='raw_values')*100
mse_corrected = mse(y_test, y_test_pred_corrected, multioutput='raw_values')
print("Corrected model performance on blind randomized testing data:", mape_corrected)
end_time = time.time()
elapsed_time = end_time - start_time
print("Time", elapsed_time)

# Save Model
stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_name = "Neural Net " + stamp + ".pth"
out_name = "Output Transformer " + stamp + ".sav"
in_name = "Input Transformer " + stamp + ".sav"

torch.save(model.module_.state_dict(), model_name)
joblib.dump(input_transformer, in_name)
joblib.dump(output_transformer, out_name)
