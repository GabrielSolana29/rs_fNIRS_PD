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
    
import functions_feature_extraction as ffe
import load_save_data as fld

#%% function to get information from patient
def get_patient(df,number,control:bool=False):    
    if control == True:
        number = number + .1    
    for i in range(np.shape(df)[0]):    
        key_ini = 0
        if df.id[i] == number:            
            key_ini = 1             
        if key_ini == 1:
            cont = 0
            while(1):
                if i+cont < np.shape(df)[0]:
                    if df.id[i+cont] == number:
                        cont += 1
                    else:
                        break
                else:
                    break                
            df = np.array(df)
            patient = df[i:i+cont,:]
            return patient[:,:-2]
        
#%% Main program
path = directory + "\\CSV\\"
# Load complete dataset
name_csv = 'complete_dataset'
df = fld.load_csv(name_csv,path,linux=linux)
feature_name = df.columns
# Load clean_dataset
name_csv = 'complete_clean_dataset'
df_clean = fld.load_csv(name_csv,path,linux=linux)

# Get the correlation features from the patietns
name_csv = 'pearson_dataset_complete'
pearson_dataset = fld.load_csv(name_csv,path,linux=linux)
feature_name = pearson_dataset.columns

#%% Feature extraction
no_patients = 20
control = False
cleaned = False
columns = 66
plots = False

no_features_connectivity = 12
no_features_pearson = 693
no_features = ((columns) * 12) + no_features_connectivity + no_features_pearson

tic = time.perf_counter()  

features_arr = np.zeros((no_patients,no_features))

if cleaned == True:
    dataset = df_clean
else:
    dataset = df

# Extract features from each patient
for i in range(1,no_patients+1):
    patient = get_patient(dataset,i,control=control) 
    feature_list = []
    
    # Extract features from each column
    sampling_rate = 0
    for column in range(1,columns+1):
        signal = np.array(patient[:,column],dtype=float)
        
        sampling_rate = np.ceil(1 / (patient[1,0] - patient[0,0]))
        # Return 11 features 
        features = ffe.extract_features_from_points(signal,sampling_rate)        
        
        feature_list.append(features)
        if plots == True:    
            plt.plot(signal)        
    
    # Extract connectivity features (from all columns)
    features_connectivity = ffe.extract_features_connectivity(patient,sampling_rate)
    vec_features = 0
    
    for j in range(len(feature_list)):
        if j==0:            
            vec_features = feature_list[0]
        else:
            vec_features = np.hstack((vec_features,feature_list[j]))
                
    vec_features = np.hstack((vec_features,features_connectivity))    
    
    
    # Pearson correlation features  
    # Append the features from pearson corr to the feature vector
    patient_pearson_corr = pearson_dataset.iloc[i-1,:-1]
    for j in range(len(patient_pearson_corr)):
        vec_features = np.hstack((vec_features,patient_pearson_corr[j]))
                
    features_arr[i-1,:] = vec_features                
        

# Save Dataset of features
name_features = ffe.list_name_features(66)
features_arr = pd.DataFrame(features_arr)
features_arr.columns = name_features

features_arr.to_csv("features_dataset_pd_pearson.csv",index=False)
        


#%% Feature extraction controls 

no_patients = 20
control = True
cleaned = False
columns = 66
plots = False

no_features_connectivity = 12
no_features_pearson = 693
no_features = ((columns) * 12) + no_features_connectivity + no_features_pearson

tic = time.perf_counter()  

features_arr = np.zeros((no_patients,no_features))

if cleaned == True:
    dataset = df_clean
else:
    dataset = df

# Extract features from each patient
for i in range(1,no_patients+1):
    patient = get_patient(dataset,i,control=control) 
    feature_list = []
    
    # Extract features from each column
    sampling_rate = 0
    for column in range(1,columns+1):
        signal = np.array(patient[:,column],dtype=float)
        
        sampling_rate = np.ceil(1 / (patient[1,0] - patient[0,0]))
        # Return 11 features 
        features = ffe.extract_features_from_points(signal,sampling_rate)        
        
        feature_list.append(features)
        if plots == True:    
            plt.plot(signal)        
    
    # Extract connectivity features (from all columns)
    features_connectivity = ffe.extract_features_connectivity(patient,sampling_rate)
    vec_features = 0
    
    for j in range(len(feature_list)):
        if j==0:            
            vec_features = feature_list[0]
        else:
            vec_features = np.hstack((vec_features,feature_list[j]))
                
    vec_features = np.hstack((vec_features,features_connectivity))  
    
    
    # Pearson correlation features  
    # Append the features from pearson corr to the feature vector
    patient_pearson_corr = pearson_dataset.iloc[i-1,:-1]
    for j in range(len(patient_pearson_corr)):
        vec_features = np.hstack((vec_features,patient_pearson_corr[j]))
                
    features_arr[i-1,:] = vec_features   
            
        

# Save Dataset of features
name_features = ffe.list_name_features(66)
features_arr = pd.DataFrame(features_arr)
features_arr.columns = name_features

features_arr.to_csv("features_dataset_controls_pearson.csv",index=False)

toc = time.perf_counter()
print("\nTime elapsed: ",(toc-tic), "seconds")


#%% Create separate datasets
################################################################
################################################################
################################################################

path = directory + "\\CSV\\"
# Load complete dataset
name_csv = 'complete_dataset'
df = fld.load_csv(name_csv,path,linux=linux)
feature_name = df.columns
dataset = df
#%% Feature extraction
no_patients = 20
control = True
cleaned = False
columns = 66
plots = False

tic = time.perf_counter()  
features_time_l =[]
features_freq_l =[]
features_con_l = []

# Extract features from each patient
for i in range(1,no_patients+1):
    patient = get_patient(dataset,i,control=control) 
    
    # Extract features from each column
    sampling_rate = 0
    for column in range(1,columns+1):
        signal = np.array(patient[:,column],dtype=float)
        
        sampling_rate = np.ceil(1 / (patient[1,0] - patient[0,0]))
        # Return 11 features 
        features_time = ffe.extract_features_time(signal,sampling_rate)        
        features_freq = ffe.extract_features_frequency(signal,sampling_rate)
        
        if column == 1:
            features_time_conca = features_time
            features_freq_conca = features_freq
        else:
            features_time_conca = np.hstack((features_time_conca,features_time))
            features_freq_conca = np.hstack((features_freq_conca,features_freq))
        
        if plots == True:    
            plt.plot(signal)       
             
    features_time_l.append(features_time_conca)
    features_freq_l.append(features_freq_conca)

    # Extract connectivity features (from all columns)
    features_connectivity = ffe.extract_features_connectivity(patient,sampling_rate)
    features_con_l.append(features_connectivity)  
    
         
#%% # Save Dataset of features
name_features,name_features_freq,name_features_connectivity = ffe.list_name_features_2(66)
features_freq_l = pd.DataFrame(features_freq_l)
features_freq_l.columns = name_features
features_time_l = pd.DataFrame(features_time_l)
features_time_l.columns = name_features_freq
features_con_l = pd.DataFrame(features_con_l)
features_con_l.columns = name_features_connectivity

features_time_l.to_csv("time_dataset_controls.csv",index=False)
features_freq_l.to_csv("frequency_dataset_controls.csv",index=False)
features_con_l.to_csv("connectivity_dataset_controls.csv",index=False)

toc = time.perf_counter()
print("\nTime elapsed: ",(toc-tic), "seconds")

# %%

path = directory + "\\CSV\\"
# Load complete dataset
name_csv = 'complete_dataset'
df = fld.load_csv(name_csv,path,linux=linux)
feature_name = df.columns
dataset = df
#%% Feature extraction
no_patients = 20
control = False
cleaned = False
columns = 66
plots = False

tic = time.perf_counter()  
features_time_l =[]
features_freq_l =[]
features_con_l = []
# Extract features from each patient
for i in range(1,no_patients+1):
    patient = get_patient(dataset,i,control=control) 
    
    # Extract features from each column
    sampling_rate = 0
    for column in range(1,columns+1):
        signal = np.array(patient[:,column],dtype=float)
        
        sampling_rate = np.ceil(1 / (patient[1,0] - patient[0,0]))
        # Return 11 features 
        features_time = ffe.extract_features_time(signal,sampling_rate)        
        features_freq = ffe.extract_features_frequency(signal,sampling_rate)
        
        if column == 1:
            features_time_conca = features_time
            features_freq_conca = features_freq
        else:
            features_time_conca = np.hstack((features_time_conca,features_time))
            features_freq_conca = np.hstack((features_freq_conca,features_freq))
        
        if plots == True:    
            plt.plot(signal)       
             
    features_time_l.append(features_time_conca)
    features_freq_l.append(features_freq_conca)

    # Extract connectivity features (from all columns)
    features_connectivity = ffe.extract_features_connectivity(patient,sampling_rate)
    features_con_l.append(features_connectivity)  

    
         
#%% # Save Dataset of features
name_features,name_features_freq,name_features_connectivity = ffe.list_name_features_2(66)
features_freq_l = pd.DataFrame(features_freq_l)
features_freq_l.columns = name_features
features_time_l = pd.DataFrame(features_time_l)
features_time_l.columns = name_features_freq
features_con_l = pd.DataFrame(features_con_l)
features_con_l.columns = name_features_connectivity

features_time_l.to_csv("time_dataset_pd.csv",index=False)
features_freq_l.to_csv("frequency_dataset_pd.csv",index=False)
features_con_l.to_csv("connectivity_dataset_pd.csv",index=False)

toc = time.perf_counter()
print("\nTime elapsed: ",(toc-tic), "seconds")
