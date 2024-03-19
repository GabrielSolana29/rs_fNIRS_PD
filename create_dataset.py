#%%
# import pandas as pd 
import numpy as np
import time
import pandas as pd

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
    
import functions_paths as fpt

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

#%% Main program

#%% Create Dataset
##Create Control csv
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
