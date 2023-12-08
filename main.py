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
    
import functions_feature_extraction_c as ffe_c
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
name_csv = 'features_dataset_complete'
df = fld.load_csv(name_csv,path,linux=linux)
feature_name = df.columns

#%% Feature selection 

tic = time.perf_counter() 

df_arr = np.array(df)
x = df_arr[:,2:]
y = df_arr[:,1]
#best_subset_lr,score_lr = ffs.wrapper_feature_subset_selection(x,y,classifier='lr',scoring='accuracy',n_jobs=-1,cv=5,xgb_optimize=True,verbose=True)
best_subset_xgb,score_xgb = ffs.wrapper_feature_subset_selection(x,y,classifier='xgb',scoring='accuracy',n_jobs=-1,cv=5,xgb_optimize=True,verbose=True)
#best_subset_knn,score_knn = ffs.wrapper_feature_subset_selection(x,y,classifier='knn',scoring='accuracy',n_jobs=-1,cv=5,xgb_optimize=True,verbose=True)

#print("\n\nBest score lr: ", score_lr,"\n")
print("\nBest score xgb: ", score_xgb,"\n")
#print("\nBest score knn: ", score_knn,"\n")

toc = time.perf_counter()
print("\n\nTime elapsed: ",(toc-tic), "seconds")










