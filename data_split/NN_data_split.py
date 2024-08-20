import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet, NeuralNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm.auto import tqdm, trange
from sklearn.base import BaseEstimator, TransformerMixin
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import joblib
from helpful_functions import InputLogTransformer, OutputLogTransformer, build_neural_network, make_datasets
import time

spectrum = False
noise = 10
if spectrum:
    num_outputs = 25
    identifier = 'spectrum'
    output_list = ['Bin ' + str(i) for i in range(25)] # training outputs
else:
    num_outputs = 3
    identifier = 'threeEns'
    output_list = ['Max Proton Energy', 'Total Proton Energy', 'Avg Proton Energy']
dfA = pd.read_hdf(f'datasets/fuchs_v5_0_seed-2_train_1274091_noise_{noise}_{identifier}_campaign2.h5', key = 'df')
dfB = pd.read_hdf(f'datasets/fuchs_v5_0_seed-2_train_campaign1_pct_100_noise_{noise}.h5', key='df')
#df = pd.concat([dfA, dfB], ignore_index=True).fillna(0)
df = pd.read_hdf(f'datasets/fuchs_v5_0_seed-2_train_1525000_noise_{noise}_threeEns_.h5', key='df').fillna(0)
df.loc[:, output_list] = df.loc[:, output_list].clip(1e-2, None)
test_df = pd.read_hdf(f'datasets/fuchs_v5_0_seed-2_test_1000000_noise_0_{identifier}_.h5', key = 'df').fillna(0)
test_df.loc[:, output_list] = test_df.loc[:, output_list].clip(1e-2, None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datype = np.float32
num_inputs = 4
num_outputs = 3
batch_size = 2**14
patience=7

input_list = ['Intensity', 'Target Thickness', 'Focal Distance', 'Contrast'] # independent variables
time_list = []
mape_list = []
point_list = []
uncorrected_list = []
# testing_list = ['Max Exact Energy', 'Total Exact Energy', 'Avg Exact Energy'] # testing set outputs, deprecated

#shuffle dataset to allow for random sampling
df = df.sample(frac=1,random_state=42).reset_index(drop=True)
X = np.array(df[input_list],dtype=datype)
y = np.array(df[output_list],dtype=datype)
#X_train0, y_train0, X_val0, y_val0, input_transformer, output_transformer = make_datasets(X, y, random_state=42)
data_fractions = np.linspace(0, 1, 7)[1:]
num_pts_tot = X.shape[0]

X_test = np.array(test_df[input_list],dtype=datype)
y_test = np.array(test_df[output_list],dtype=datype)

# Network architecture variables determined by hyperparameter optimization
n_hidden = 12
n_neurons = 64
lr = 1e-2
gamma = 0.85

for frac in data_fractions:
    np.random.seed(42)
    torch.manual_seed(42)
    t0 = time.time()
    num_pts = int(num_pts_tot * frac + 0.5)
    num_train_pts = int(0.8 * num_pts + 0.5)
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
    print("Model performance on training data:", mean_absolute_percentage_error(y_train, y_train_pred)*100)
    print("Model performance on validation data", mean_absolute_percentage_error(y_val, y_pred)*100)
    mape_uncorrected = mean_absolute_percentage_error(y_test, y_test_pred)*100
    print("Model performance on blind randomized testing data:", mape_uncorrected)
    mape = mean_absolute_percentage_error(y_test, y_test_pred_corrected)*100
    print("Corrected model performance on blind randomized testing data:", mape)
    t1 = time.time()
    del_t = t1 - t0
    print("Time", del_t)
    time_list.append(del_t)
    mape_list.append(mape)
    point_list.append(num_train_pts)
    uncorrected_list.append(mape_uncorrected)

df = pd.DataFrame({'points': point_list, 'mape': uncorrected_list, 'corrected': mape_list, 'time': time_list})
df.to_csv(f'results/NN_data_split_patience_{patience}.csv', index=False)