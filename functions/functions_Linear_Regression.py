import numpy as np
import os
import pandas as pd
import pickle as pk
import functions_feature_extraction as ffe
import functions_paths as fpt
import load_save_data as fld
from sklearn.linear_model import LinearRegression
import functions_signal_processing_analysis as fspa

#%% Linear regression 
"""
#test example
current_imf = 1
x,y = ffe.get_xtrain_ytrain(xtrain_list_complete,ytrain_list,current_imf,no_points_feat)
xtrain = x[:len(y),:]
ytrain = y
model = train_lr(xtrain,ytrain,verbosity=True)
plt.plot(model.predict(xtrain))

name="TEST_LR"
save_lr_model(name,model,linux=linux)
new_model = load_lr_model(name,linux=linux)
plt.plot(new_model.predict(xtrain))
"""

def train_save_lr(xtrain_list,ytrain_list,name_csv,no_imf,no_points_feat,linux:bool=False,verbosity:bool=False):

    # Get the data to train the model in tensor form 
    xtrain,ytrain = ffe.get_xtrain_ytrain(xtrain_list,ytrain_list,no_imf)            
    model = train_lr(xtrain,ytrain,verbosity=verbosity)
    
    name_model ="lr_imf" + str(no_imf) + "_" + str(name_csv)
    save_lr_model(name_model,model,linux=linux)
    

def train_lr(xtrain,ytrain,verbosity:bool=False):
    model = LinearRegression(n_jobs=-1)
    model.fit(xtrain,ytrain)
    
    if verbosity == True:
        r_sq = model.score(xtrain, ytrain)
        print('\nCoefficient of determination:', r_sq)
        y_pred = model.predict(xtrain)        
        cor = fspa.correlation_between_signals(y_pred,ytrain)
        print("\nCorrelation with ytrain: ","\n",cor)
        
    return model        


def load_all_lr_models(no_imfs,name_csv,linux:bool=False,verbosity:bool=True):    
    path_name = fpt.path_saved_models_params(linux=linux) 
    os.chdir(path_name)    
    model_vec = []
           
    for i in range(no_imfs):         
        name_model = "lr_imf"+ str(i) +"_" + str(name_csv)
        model_vec.append(load_lr_model(name_model,linux=linux))
                
    return model_vec


def save_lr_model(name,model,linux:bool=False,verbosity:bool=False):
    
    path = fpt.path_saved_models_params(linux=linux)   
    
    if linux==False:
        file_name =  path + "\\" + name
    else:
        file_name =  path + "/" + name
    
    pk.dump(model, open(file_name, 'wb'))    
    
    if verbosity==True:
        print("Model saved")
    

def load_lr_model(name,linux:bool=False,verbosity:bool=False):
    
    path = fpt.path_saved_models_params(linux=linux)   
    
    if linux==False:
        file_name =  path + "\\" + name
    else:
        file_name =  path + "/" + name
    
    loaded_model = pk.load(open(file_name, 'rb'))
    
    if verbosity==True:
        print("model loaded")
        
    return loaded_model

    