import numpy as np
import os
import pandas as pd
import pickle as pk
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

#%% Import scripts
import functions_feature_extraction as ffe
import functions_paths as fpt
import load_save_data as fld
import functions_signal_processing_analysis as fspa


def train_save_SVM(xtrain_list,ytrain_list,name_csv,no_imf,no_points_feat,opt:bool=True,linux:bool=False,verbosity:bool=False):    
    
    xtrain,ytrain = ffe.get_xtrain_ytrain(xtrain_list,ytrain_list,no_imf)    
    
    if opt == True:
        opt_params = optimize_params(xtrain,ytrain,no_jobs = -1,no_iterations_opt = 2500)            
        model = train_regression_svm(xtrain,ytrain,params=opt_params,verbosity=verbosity,epochs=1000)
    else:
        model = train_regression_svm(xtrain,ytrain,verbosity=verbosity)
    
    name_model ="svm_imf" + str(no_imf) + "_" + str(name_csv)
    save_SVM_model(name_model,model,linux=linux)
    

def optimize_params(xtrain,ytrain,no_jobs:int = -1, no_iterations_opt:int=250):
     #split the data   
    svm_opt = SVR()
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)
    #eval_set = [(X_train, y_train), (X_test, y_test)]
    #eval_metric = ["rmse"]
    eval_metric = ["mphe"]
    #eval_metric = ["mae"]
    # Find the best parameters                             
    params = {'kernel':('linear','poly', 'rbf'), 'C':np.arange(0.01,20,.01),'epsilon':np.arange(0.02,20,.01)}
    # Define the parameters of the search
    search = RandomizedSearchCV(svm_opt, param_distributions=params, n_iter=no_iterations_opt, cv=5, verbose=1, n_jobs=no_jobs, return_train_score=True, scoring='neg_mean_squared_error')
    # Search    
    #search.fit(X_train, y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=False)          
    search.fit(xtrain, ytrain)          
    # Save the optimized parameters        
    opt_params = search.best_params_

    return opt_params


def train_regression_svm(xtrain,ytrain,epochs:int=200,params:dict={},verbosity:bool=False):    
            
    if bool(params) == False:
        #If parameters are not provided, default parameters are given        
        params = {'kernel':('rbf'), 'C':.05}                
        
    print("\nTraining svm_model...")    
    #set the optimized parameters        
    svm_model = SVR(**params) 
            
    #if erly_stpping_rounds >= 10:        
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    svm_model.fit(xtrain,ytrain)
    #else:
    #    svm_model.fit(xtrain,ytrain,verbose=verbosity)                                      
        
    if verbosity==True:
        print("\optimized_params:",params)
        pred = svm_model.predict(xtrain)
        fspa.correlation_between_signals(pred,ytrain,verbosity=True)
        print("\nFinished training")
    
    
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



"""
n_samples, n_features = 1000, 5
rng = np.random.RandomState(0)
ytrain = rng.randn(n_samples)
xtrain = rng.randn(n_samples, n_features)
svm = SVR(C=1.0, epsilon=0.2,kernel='rbf')
svm.fit(xtrain,ytrain)

name = "TEST_SVM"
save_SVM_model(name,svm,linux=linux,verbosity=True)

model_2 = load_SVM_model(name,linux=linux)
model_2.predict(xtrain)

X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)
#The function for tuning parameters is 
no_iterations = 1000
no_jobs = -1
svm = SVR()

opt_params = optimize_params(xtrain,ytrain,no_jobs = -1,no_iterations_opt = 2000)
svc = train_regression_svm(xtrain,ytrain,epochs=200,params=opt_params,verbosity=True) 
svc.predict(xtrain)
"""

