import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge as cpu_Ridge
from sklearn.preprocessing import PolynomialFeatures
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
sys.path.append('../../')
from helpful_functions import make_poly_datasets
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
dfA = pd.read_hdf(f'../../datasets/fuchs_v5_0_seed-2_train_campaign1__noise_{noise}.h5', key = 'df')
dfB = pd.read_hdf(f'../../datasets/fuchs_v5_0_seed-2_train_campaign2__noise_{noise}.h5', key='df')
df = pd.concat([dfA, dfB], ignore_index=True).sample(frac=0.60).fillna(0)
test_df = pd.read_hdf(f'../../datasets/fuchs_v5_0_seed-2_test_250000_noise_0_{identifier}_.h5', key = 'df').fillna(0)

datype = np.float32
num_inputs = 4
num_outputs = 3

input_list = ['Intensity', 'Target Thickness', 'Focal Distance', 'Contrast'] # independent variables
time_list = []
max_mape_corrected_list = []
max_mse_corrected_list = []
max_mape_uncorrected_list = []
max_mse_uncorrected_list = []
tot_mape_corrected_list = []
tot_mse_corrected_list = []
tot_mape_uncorrected_list = []
tot_mse_uncorrected_list = []
avg_mape_corrected_list = []
avg_mse_corrected_list = []
avg_mape_uncorrected_list = []
avg_mse_uncorrected_list = []
point_list = []

df = df.sample(frac=1,random_state=42).reset_index(drop=True)
X = np.array(df[input_list],dtype=datype)
y = np.array(df[output_list],dtype=datype)
data_fractions = np.linspace(0, 1, 7)[1:]
num_pts_tot = X.shape[0]

X_test = np.array(test_df[input_list],dtype=datype)
y_test = np.array(test_df[output_list],dtype=datype)

# Polynomial determined by hyperparameter optimization
degree = 1
alpha = 1e1

for frac in data_fractions:
    np.random.seed(42)
    torch.manual_seed(42)
    start_time = time.time()
    num_pts = int(num_pts_tot * frac + 0.5)
    num_train_pts = int(0.8 * num_pts + 0.5)
    print("training", num_train_pts)
    X_train, y_train, X_val, y_val, input_transformer, output_transformer = make_poly_datasets(X[:num_pts, :], y[:num_pts, :], random_state=42)
    poly = PolynomialFeatures(degree=degree)
    ridge = cpu_Ridge(alpha=alpha)
    model = make_pipeline(poly, ridge)
    model.fit(X_train, y_train)
    y_pred = output_transformer.inverse_transform(model.predict(X_val))
    y_train_pred = output_transformer.inverse_transform(model.predict(X_train))
    y_train_true = output_transformer.inverse_transform(y_train)
    correction_factors = []
    for i in range(num_outputs):
        α = np.mean(y_train_true[:, i]/y_train_pred[:, i])
        correction_factors.append(α)
    y_train_pred_corrected = y_train_pred.copy()
    for i in range(num_outputs):
        y_train_pred_corrected[:, i] *= correction_factors[i]
    X_train_true = input_transformer.inverse_transform(X_train)
    y_val_true = output_transformer.inverse_transform(y_val)
    y_train_pred = output_transformer.inverse_transform(model.predict(X_train))
    X_test_scaled = input_transformer.transform(X_test)
    y_test_pred = output_transformer.inverse_transform(model.predict(X_test_scaled))
    y_test_pred_corrected = y_test_pred.copy()
    for i in range(num_outputs):
        y_test_pred_corrected[:, i] *= correction_factors[i]
    print("Model performance on training data:", mape(y_train, y_train_pred, multioutput='raw_values')*100)
    print("Model performance on validation data", mape(y_val, y_pred, multioutput='raw_values')*100)
    try: 
        mape_uncorrected = mape(y_test, y_test_pred, multioutput='raw_values')*100
    except ValueError as e:
        print(e)
        mape_uncorrected = np.full_like(y_test[0], 1e6, dtype=np.float32)
    try:
        mse_uncorrected = mse(y_test, y_test_pred, multioutput='raw_values')
    except ValueError as e:
        print(e)
        mse_uncorrected = np.full_like(y_test[0], 1e6, dtype=np.float32)
    print("Model performance on blind randomized testing data:", mape_uncorrected)
    try:
        mape_corrected = mape(y_test, y_test_pred_corrected, multioutput='raw_values')*100
    except ValueError as e:
        print(e)
        mape_corrected = np.full_like(y_test[0], 1e6, dtype=np.float32)
    try:
        mse_corrected = mse(y_test, y_test_pred_corrected, multioutput='raw_values')
    except ValueError as e:
        print(e)
        mse_corrected = np.full_like(y_test[0], 1e6, dtype=np.float32)
    print("Corrected model performance on blind randomized testing data:", mape_corrected)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time", elapsed_time)
    time_list.append(elapsed_time)
    point_list.append(num_train_pts)
    max_mape_corrected_list.append(mape_corrected[0])
    max_mse_corrected_list.append(mse_corrected[0])
    max_mape_uncorrected_list.append(mape_uncorrected[0])
    max_mse_uncorrected_list.append(mse_uncorrected[0])
    tot_mape_corrected_list.append(mape_corrected[1])
    tot_mse_corrected_list.append(mse_corrected[1])
    tot_mape_uncorrected_list.append(mape_uncorrected[1])
    tot_mse_uncorrected_list.append(mse_uncorrected[1])
    avg_mape_corrected_list.append(mape_corrected[2])
    avg_mse_corrected_list.append(mse_corrected[2])
    avg_mape_uncorrected_list.append(mape_uncorrected[2])
    avg_mse_uncorrected_list.append(mse_uncorrected[2])

df = pd.DataFrame({'points': point_list, 'time': time_list,
                'max mape corrected': max_mape_corrected_list, 'max mape uncorrected': max_mape_uncorrected_list, 
                'max mse corrected': max_mse_corrected_list, 'max mse uncorrected': max_mse_uncorrected_list, 
                'tot mape corrected': tot_mape_corrected_list, 'tot mape uncorrected': tot_mape_uncorrected_list, 
                'tot mse corrected': tot_mse_corrected_list, 'tot mse uncorrected': tot_mse_uncorrected_list, 
                'avg mape corrected': avg_mape_corrected_list, 'avg mape uncorrected': avg_mape_uncorrected_list, 
                'avg mse corrected': avg_mse_corrected_list, 'avg mse uncorrected': avg_mse_uncorrected_list})
df.to_csv(f'results/POLY_data_split_deg{degree}.csv', index=False)