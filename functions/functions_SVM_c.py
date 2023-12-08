import numpy as np
import os
import pandas as pd
import pickle as pk
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%% Import scripts
import functions_feature_extraction_c as ffe_c
import functions_paths as fpt
import load_save_data as fld
import functions_signal_processing_analysis_c as fspa_c
#%% Functions


def train_save_SVM(xtrain,ytrain,name_csv,no_imf,opt:bool=True,epochs:int=500,no_iterations:int=300,linux:bool=False,verbosity:bool=False):    
    
    if opt == True:
        opt_params = optimize_params(xtrain,ytrain, no_iterations_opt=no_iterations)              
        model = train_svm(xtrain,ytrain,epochs=epochs,params=opt_params,verbosity=True)
    else:
        model = train_svm(xtrain,ytrain,epochs=epochs,verbosity=True)
            
    name_model ="svm_imf" + str(no_imf) + "_" + str(name_csv)
    save_SVM_model(name_model,model,linux=linux)
    
    if verbosity == True:
        predictions = model.predict(xtrain)
        ac = accuracy_score(ytrain, predictions)
        print("\nAccuracy of the svm model: ", ac, "\n")


def optimize_params(xtrain,ytrain,no_jobs:int = -1, test_size:int=.2, cache_size:int=8000,no_iterations_opt:int=250):
     #split the data   
    svm_opt = svm.SVC(cache_size=cache_size) #### THIS IS NEW 
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=test_size, random_state=12)
    #eval_set = [(X_train, y_train), (X_test, y_test)]
    #eval_metric = ["rmse"]
    eval_metric = ["mphe"]
    #eval_metric = ["mae"]
    # Find the best parameters                             
    params = {'kernel':('linear','poly', 'rbf'), 'C':np.arange(0.01,5,.01),'gamma':np.arange(0.02,20,.01)}
    # Define the parameters of the search
    search = RandomizedSearchCV(svm_opt, param_distributions=params, n_iter=no_iterations_opt, cv=5, verbose=1, n_jobs=no_jobs, return_train_score=True, scoring='accuracy')
    # Search    
    #search.fit(X_train, y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=False)          
    search.fit(xtrain, ytrain)          
    # Save the optimized parameters        
    opt_params = search.best_params_

    return opt_params


def train_svm(xtrain,ytrain,epochs:int=200,test_size:int=.2,params:dict={},kernel:str="rbf",verbosity:bool=False):    
            
    if bool(params) == False:
        #If parameters are not provided, default parameters are given        
        params = {'kernel':('rbf'), 'C':1}                
        
    print("\nTraining svm_model...")    
    #set the optimized parameters        
    svm_model = svm.SVC(**params)     
                
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=test_size, random_state=12)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    svm_model.fit(xtrain,ytrain)

    if verbosity==True:
        print("\optimized_params:",params)
        pred = svm_model.predict(xtrain)
        ac = accuracy_score(ytrain, pred)
        print("\nAccuracy: ",ac)
        print("\nFinished training svm model")    
    
    return svm_model
    
       
def load_all_SVM_models(no_imfs,name_csv,linux:bool=False,verbosity:bool=True):    
    path_name = fpt.path_saved_models_params(linux=linux) 
    os.chdir(path_name)    
    model_vec = []
           
    for i in range(no_imfs):         
        name_model = "svm_imf"+ str(i) +"_" + str(name_csv)
        model_vec.append(load_SVM_model(name_model,linux=linux))
                
    return model_vec


def save_SVM_model(name,model,linux:bool=False,verbosity:bool=False):
    
    path = fpt.path_saved_models_params(linux=linux)   
    
    if linux==False:
        file_name =  path + "\\" + name
    else:
        file_name =  path + "/" + name
    
    pk.dump(model, open(file_name, 'wb'))    
    
    if verbosity==True:
        print("Model saved")
    

def load_SVM_model(name,linux:bool=False,verbosity:bool=False):
    
    path = fpt.path_saved_models_params(linux=linux)   
    
    if linux==False:
        file_name =  path + "\\" + name
    else:
        file_name =  path + "/" + name
    
    loaded_model = pk.load(open(file_name, 'rb'))
    
    if verbosity==True:
        print("model loaded")
        
    return loaded_model