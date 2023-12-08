#%%
import pandas as pd 
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt

linux = False
import os
#%% Import scripts
#from xgboost import XGBRegressor
directory = os.getcwd()
if directory[0] == "/":
    linux = True
else:
    linux = False
#%% Import scripts
import sys
if linux == True:  
    directory_functions = str(directory +"/functions/")
    sys.path.insert(0, directory_functions) # linux
else:
    directory_functions = str(directory +"\\functions\\")
    sys.path.insert(0, directory_functions) # linux
    
import functions_ensemble as fe
#import functions_feature_extraction_c as ffe_c
import functions_feature_extraction as ffe
import functions_xgboost_c as fxgb_c
import functions_signal_processing_analysis_c as fspa_c
#import functions_causal_feature_selection as fcfs
import functions_feature_selection as ffs
import functions_paths as fpt
import functions_assist as fa
import load_save_data as fld
import functions_specific as fsp

#%% Functions

def standardize(matrix):
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    return (matrix - means) / stds

def scale_0_1(matrix):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    return (matrix - mins) / (maxs - mins)

#%% Main program
path = directory + "\\CSV"
# Load complete dataset
name_csv = 'time_dataset_complete'
df_time = fld.load_csv(name_csv,path,linux=linux)
feature_name_time = df_time.columns

name_csv = 'frequency_dataset_complete'
df_freq = fld.load_csv(name_csv,path,linux=linux)
feature_name_freq = df_freq.columns

name_csv = 'connectivity_dataset_complete'
df_conn = fld.load_csv(name_csv,path,linux=linux)
feature_name_conn = df_conn.columns

name_csv = 'pearson_dataset_complete'
df_pear = fld.load_csv(name_csv,path,linux=linux)
feature_name_pear = df_pear.columns

#%% Fix datasets
df_arr_time = np.array(df_time)
x = df_arr_time[:,:-1]

df_arr_freq = np.array(df_freq)
x2 = df_arr_freq[:,:-1]

df_arr_conn = np.array(df_pear)
x3 = df_arr_conn[:,:-1]

df_arr_pear = np.array(df_pear)
x4 = df_arr_pear[:,:-1]

y = df_arr_time[:,-1]

# Normalize datasets
x = scale_0_1(x)
x2 = scale_0_1(x2)

#%% Make predictions with each subset
tic = time.perf_counter() 
for i in range(5):
    subset_time,subset_freq,subset_conn = fe.get_subsets(algo="knn")
    pred_ensemble = fe.ensemble_pred(x,x2,x3,y,subset_time,subset_freq,subset_conn,algo="knn",test_size=0.4)

    subset_time,subset_freq,subset_conn = fe.get_subsets(algo="lr")
    pred_ensemble = fe.ensemble_pred(x,x2,x3,y,subset_time,subset_freq,subset_conn,algo="lr",test_size=0.4)

    subset_time,subset_freq,subset_conn = fe.get_subsets(algo="xgb")
    pred_ensemble = fe.ensemble_pred(x,x2,x3,y,subset_time,subset_freq,subset_conn,algo="xgb",test_size=0.4)

toc = time.perf_counter()
print("\n\nTime elapsed: ",(toc-tic), "seconds")


# %%
