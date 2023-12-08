#%%
# import pandas as pd 
import numpy as np
import time
import scipy.io

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
import functions_xgboost_c as fxgb_c
import functions_signal_processing_analysis_c as fspa_c
import functions_causal_feature_selection as fcfs
import functions_feature_selection as ffs
import functions_paths as fpt
import functions_assist as fa
import functions_specific as fsp

#%% functions

def load_tsv_dataset(dataset_name,linux:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + dataset_name + ".tsv" # linux
    else:        
        complete_name = path_name + "\\" + dataset_name + ".tsv" # Windows
    
    df = pd.read_csv(
        complete_name,  
        sep='\t',
        na_values=['NA', '?'])
    
    df = df
    
    return df

# clean dataset according to the tsv "motion" which when zero indicates movement an so the sample must be discarded
def clean_dataset(df,dataset_name,linux:bool=False):
    dataset_name = dataset_name[0:23] + "motion_" + dataset_name[23:]
    tsv = load_tsv_dataset(dataset_name,linux=linux)    
    tsv = np.array(tsv)
    list_delete = []
    cont = 0
    for i in range(len(tsv)):
        if tsv[i,0] == 0:
            list_delete.append(i)
            cont += 1
    
    if cont > 0:    
        df_clean = np.delete(df, list_delete, axis = 0)
    else:
        df_clean = df
        
    return df_clean
    

#%% Main program

#%% Create Dataset
##Create control csv
df_list = []
no_datasets = 20
 

for i in range(no_datasets):
    tic = time.perf_counter()
    name_short = "1_resting_seg_1_export_"
    if i+1 <= 9:        
        name_tsv = str(name_short + "0" + str(i+1) + "_c")    
        print(name_tsv)
    else:
        name_tsv = str(name_short + str(i+1) + "_c")    
        print(name_tsv)
    
    patient = load_tsv_dataset(name_tsv,linux=linux)
    patient = np.array(patient)
    #patient = clean_dataset(patient,name_tsv)
    df_list.append(patient[:,0:69])
    

for i in range(no_datasets):
    if i ==0 :    
        comp_dataset_c = df_list[0]
        
    else:
        current = df_list[i]
        comp_dataset_c = np.vstack((comp_dataset_c,current))

df = load_tsv_dataset(name_tsv,linux=linux) 
feature_name = df.columns   

## Create pd csv
df_list = []
no_datasets = 20

for i in range(no_datasets):
    tic = time.perf_counter()
    name_short = "1_resting_seg_1_export_"
    if i+1 <= 9:        
        name_tsv = str(name_short + "0" + str(i+1) + "_pd")    
        print(name_tsv)
    else:
        name_tsv = str(name_short + str(i+1) + "_pd")    
        print(name_tsv)
        
    patient = load_tsv_dataset(name_tsv,linux=linux)    
    patient = np.array(patient)
    #patient = clean_dataset(patient,name_tsv)
    df_list.append(patient[:,0:69])


for i in range(no_datasets):
    if i ==0 :    
        comp_dataset_pd = df_list[0]
        
    else:
        current = df_list[i]
        comp_dataset_pd = np.vstack((comp_dataset_pd,current))

 
## Combine both datasets
comp_dataset = np.vstack((comp_dataset_pd,comp_dataset_c))
comp_dataset = pd.DataFrame(comp_dataset)
comp_dataset.columns = feature_name
comp_dataset.to_csv('complete_dataset.csv', index=False)


#%%
### Create Clean Dataset this removes artifacts from movement, but time series is not complete to measure peaks and valleys etc.

##Create control csv
df_list = []
no_datasets = 20
 

for i in range(no_datasets):
    tic = time.perf_counter()
    name_short = "1_resting_seg_1_export_"
    if i+1 <= 9:        
        name_tsv = str(name_short + "0" + str(i+1) + "_c")    
        print(name_tsv)
    else:
        name_tsv = str(name_short + str(i+1) + "_c")    
        print(name_tsv)
    
    patient = load_tsv_dataset(name_tsv,linux=linux)
    patient = np.array(patient)
    patient = clean_dataset(patient,name_tsv)
    df_list.append(patient[:,0:69])
    

for i in range(no_datasets):
    if i ==0 :    
        comp_dataset_c = df_list[0]
        
    else:
        current = df_list[i]
        comp_dataset_c = np.vstack((comp_dataset_c,current))

df = load_tsv_dataset(name_tsv,linux=linux) 
feature_name = df.columns   

## Create pd csv
df_list = []
no_datasets = 20

for i in range(no_datasets):
    tic = time.perf_counter()
    name_short = "1_resting_seg_1_export_"
    if i+1 <= 9:        
        name_tsv = str(name_short + "0" + str(i+1) + "_pd")    
        print(name_tsv)
    else:
        name_tsv = str(name_short + str(i+1) + "_pd")    
        print(name_tsv)
        
    patient = load_tsv_dataset(name_tsv,linux=linux)    
    patient = np.array(patient)
    patient = clean_dataset(patient,name_tsv)
    df_list.append(patient[:,0:69])


for i in range(no_datasets):
    if i ==0 :    
        comp_dataset_pd = df_list[0]
        
    else:
        current = df_list[i]
        comp_dataset_pd = np.vstack((comp_dataset_pd,current))

 
## Combine both datasets
comp_dataset = np.vstack((comp_dataset_pd,comp_dataset_c))
comp_dataset = pd.DataFrame(comp_dataset)
comp_dataset.columns = feature_name
comp_dataset.to_csv('complete_clean_dataset.csv', index=False)


