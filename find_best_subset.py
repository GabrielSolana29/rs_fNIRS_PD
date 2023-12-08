#%%
import pandas as pd 
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
feature_name_conn = df_freq.columns

name_csv = 'pearson_dataset_complete'
df_pear = fld.load_csv(name_csv,path,linux=linux)
feature_name_pear = df_conn.columns

#%% Functions

def standardize(matrix):
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    return (matrix - means) / stds

def scale_0_1(matrix):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    return (matrix - mins) / (maxs - mins)

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

#%% create datasets

tic = time.perf_counter() 

df_arr_time = np.array(df_time)
x = df_arr_time[:,:-1]

df_arr_freq = np.array(df_freq)
x2 = df_arr_freq[:,:-1]

df_arr_conn = np.array(df_conn)
x3 = df_arr_conn[:,:-1]

df_arr_pear = np.array(df_pear)
x4 = df_arr_pear

y = df_arr_time[:,-1]

# Normalize datasets
x = scale_0_1(x)
x2 = scale_0_1(x2)
#%% Feature selection
#best_subset_lr_time,score_lr_time = ffs.wrapper_feature_subset_selection(x,y,classifier='lr',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
#best_subset_lr_freq, score_lr_freq = ffs.wrapper_feature_subset_selection(x2,y,classifier='lr',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
#best_subset_lr_conn, score_lr_conn = ffs.wrapper_feature_subset_selection(x3,y,classifier='lr',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
best_subset_lr_pear,score_lr_pear = ffs.wrapper_feature_subset_selection(x4,y,classifier='lr',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)


#best_subset_xgb_time,score_xgb_time = ffs.wrapper_feature_subset_selection(x,y,classifier='xgb',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
#best_subset_xgb_freq,score_xgb_freq = ffs.wrapper_feature_subset_selection(x2,y,classifier='xgb',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
#best_subset_xgb_conn,score_xgb_conn = ffs.wrapper_feature_subset_selection(x3,y,classifier='xgb',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
best_subset_xgb_pear,score_xgb_pear = ffs.wrapper_feature_subset_selection(x4,y,classifier='xgb',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)

#best_subset_knn_time,score_knn_time = ffs.wrapper_feature_subset_selection(x,y,classifier='knn',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
#best_subset_knn_freq,score_knn_freq = ffs.wrapper_feature_subset_selection(x2,y,classifier='knn',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
best_subset_knn_conn,score_knn_conn = ffs.wrapper_feature_subset_selection(x3,y,classifier='knn',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)


#%%
#lr = LogisticRegression(class_weight='balanced', penalty = 'l2', random_state=42, n_jobs=-1,max_iter=1000).fit(x2[0:35],y[0:35])
#predictions = lr.predict(x2)

#%%
"""
print("\n\nBest score lr time: ", score_lr_time,"\n")
print("\n\nBest score lr frequency: ", score_lr_freq,"\n")
print("\n\nBest score lr connectivity: ", score_lr_conn,"\n")
print("\nBest score xgb time: ", score_xgb_time,"\n")
print("\nBest score xgb frequency: ", score_xgb_freq,"\n")
print("\nBest score xgb connectivity: ", score_xgb_conn,"\n")
print("\n\nBest score knn time: ", score_knn_time,"\n")
print("\n\nBest score knn frequency: ", score_knn_freq,"\n")
print("\n\nBest score knn connectivity: ", score_knn_conn,"\n")

toc = time.perf_counter()
print("\n\nTime elapsed: ",(toc-tic), "seconds")
"""
#%%
best_subset_knn_conn = pd.DataFrame(best_subset_knn_conn)
#best_subset_knn_time = pd.DataFrame(best_subset_knn_time)
#best_subset_knn_freq = pd.DataFrame(best_subset_knn_freq)


best_subset_lr_conn = pd.DataFrame(best_subset_lr_conn)
#best_subset_lr_time = pd.DataFrame(best_subset_lr_time)
#best_subset_lr_freq = pd.DataFrame(best_subset_lr_freq)


best_subset_xgb_conn = pd.DataFrame(best_subset_xgb_conn)
#best_subset_xgb_time = pd.DataFrame(best_subset_xgb_time)
#best_subset_xgb_freq = pd.DataFrame(best_subset_xgb_freq)

#best_subset_knn_time.to_csv('best_subset_knn_time.csv', index=False)
best_subset_knn_conn.to_csv('best_subset_knn_conn.csv', index=False)
#best_subset_knn_freq.to_csv('best_subset_knn_freq.csv', index=False)

#best_subset_xgb_time.to_csv('best_subset_xgb_time.csv', index=False)
best_subset_xgb_conn.to_csv('best_subset_xgb_conn.csv', index=False)
#best_subset_xgb_freq.to_csv('best_subset_xgb_freq.csv', index=False)

#best_subset_lr_time.to_csv('best_subset_lr_time.csv', index=False)
best_subset_lr_conn.to_csv('best_subset_lr_conn.csv', index=False)
#best_subset_lr_freq.to_csv('best_subset_lr_freq.csv', index=False)


# %%
