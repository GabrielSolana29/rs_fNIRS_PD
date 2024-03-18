#%% Train XgBoost
import numpy as np
from xgboost import XGBClassifier
import pickle
import os
#from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score,KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import functions_paths as fpt

def optimize_params(xtrain,ytrain,no_jobs:int = -1, no_iterations_opt:int=250):
     #split the data
    eval_metric = 'error'
   
    xgb_opt = XGBClassifier(objective='binary:logistic',eval_metric=eval_metric,booster='gbtree')
    #X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)           
                
    params = {
        "colsample_bytree": np.arange(0.3,1.03,.1),                        
        "min_child_weight": np.arange(0,6,1),
        "gamma": np.arange(0,1.2,.2),
        "learning_rate": np.arange(0.01,0.25,.02),
        "max_depth": np.arange(2,9,1), 
        "n_estimators": np.arange(100,160,5),
        "subsample": np.arange(0.4,1.05,.1),
        "lambda": np.arange(0,3.25,.2),            
        "alpha": np.arange(0,3.25,.2)                       
    }                

    # Define the parameters of the search
    search = RandomizedSearchCV(xgb_opt, param_distributions=params, n_iter=no_iterations_opt, cv=5, verbose=1, n_jobs=no_jobs, return_train_score=True)#, scoring='binary:logistic')
    # Search    
    #search.fit(X_train, y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=False)          
    search.fit(xtrain, ytrain,verbose=False)          
    # Save the optimized parameters        
    opt_params = search.best_params_
    
    opt_params['seed'] = 0    
    opt_params['eval_metric']= 'error'
    opt_params['booster'] = 'gbtree'    
    opt_params['base_score'] = .5  
    opt_params['colsample_bytree'] = 1             
    #set the optimized parameters    
    return opt_params


def train_classification_xgboost(xtrain,ytrain,epochs:int=300,params:dict={},verbosity:bool=False,erly_stpping_rounds:int=10,gpu:bool=False):    
            
    if bool(params) == False:
        #If parameters are not provided, default parameters are given        
        params['seed'] = 0        
        params['eval_metric']= 'error'        
        params['n_estimators'] = epochs
        params['booster'] = 'gbtree'    
        params['base_score'] = .5
        params['colsample_bytree'] = 1                 
        
    params['booster'] = 'gbtree'  
    params['n_estimators'] = epochs   
    params['objective'] = 'binary:logistic'
    #params['use_label_encoder'] = False
    if gpu==True:
        params['tree_method'] = 'gpu_hist'
        
    print("\nTraining xgboost_model...")    
    #set the optimized parameters        
    xgb_model = XGBClassifier(**params) 
            
    if erly_stpping_rounds >= 10:        
        X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.30, random_state=12)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xgb_model.fit(xtrain,ytrain,early_stopping_rounds=erly_stpping_rounds,eval_set=eval_set,verbose = verbosity)
    else:
        xgb_model.fit(xtrain,ytrain,verbose=verbosity)    
    
                  
    if verbosity == True:            
        #fspa.correlation_between_signals(predictions,ytrain,verbosity=True)
        print("\nFinished training")
    
    return xgb_model


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


def save_params(params,name_params,lunx:bool=False,verbosity:bool=True,linux:bool=False):
    path_name = fpt.path_saved_models_params(linux=linux)  
    os.chdir(path_name) 
        
    # Save parameters
    name_params = name_params + ".pkl"
    f = open(name_params,"wb")
    pickle.dump(params,f)
    f.close()
    if verbosity == True:
        print("\nParams saved")

def save_model(xgb_model,name_model,linux:bool=False,verbosity:bool=True):    
    path_name = fpt.path_saved_models_params(linux=linux)   
    os.chdir(path_name) 
    
    name_save = name_model + ".txt"
    xgb_model.save_model(name_save)
    if verbosity == True:
        print("\nmodel saved")


def load_model(name_model,linux:bool=False,verbosity:bool=True):  
    path_name = fpt.path_saved_models_params(linux=linux)      
    os.chdir(path_name) 
    
    model = XGBClassifier()
    try:
        name_load = name_model + ".txt"
        model.load_model(name_load)
        if verbosity == True:            
            print("\nModel loaded")
        
    except EOFError:
        print("\nFile with trained model is empty ")    
    except:
        print("\nFile with trained xgboost model not found")
                
    return model


def load_params(name_params,linux:bool=False,verbosity:bool=True):
    path_name = fpt.path_saved_models_params(linux=linux)  
    os.chdir(path_name) 
        
    # Load parameters
    name_params_p = name_params + ".pkl"
    fl = open(name_params_p,"rb")    
    try:
        new_params = pickle.load(fl)    
        fl.close()
        if verbosity == True:
            print("\nParams loaded")
    except EOFError:
        print("\nFile with optimized parameters is empty ")    
        fl.close()            
    
    return new_params


def train_save_xgboost_models_params(no_imf,xtrain,ytrain,name_csv,epochs:int=500,no_iterations_opt:int=300,no_points_feature_extraction:int=10,linux:bool=False,verbosity:bool=False,gpu:bool=False):
    current_imf = no_imf
    name_model = "xgboost_imf"+ str(current_imf) +"_" + str(name_csv)
    name_params ="params_imf"+ str(current_imf) +"_" + str(name_csv) 
    # Get the data to train the model 
    #xtrain,ytrain = ffe.get_xtrain_ytrain(xtrain_list,ytrain_list,no_imf)        
    # Find optimal parameters for xgb model
    opt_params = optimize_params(xtrain,ytrain,no_iterations_opt=no_iterations_opt)
    save_params(opt_params, name_params,linux = linux)
    # Train the regression xgb model     
    
    xgb_model = train_classification_xgboost(xtrain,ytrain,params=opt_params,epochs=epochs,verbosity=verbosity,gpu=gpu)
    
    # Save xgboost model        
    save_model(xgb_model, name_model,linux = linux)
    
    if verbosity == True:
        print("Model ",no_imf," trained and saved")
        predictions = xgb_model.predict(xtrain)        
        ac = accuracy_score(ytrain, predictions)
        print("\nAccuracy imf ", no_imf, " : ", ac,"\n\n")
        
        
def load_all_xgboost_models(no_imfs,name_csv,linux:bool=False,verbosity:bool=False):    
    model_vec = []
    for i in range(0,no_imfs):    
        name_model = "xgboost_imf"+ str(i) +"_" + str(name_csv)
        model = load_model(name_model,linux = linux, verbosity=verbosity)                        
        model_vec.append(model)
        
    return model_vec


def load_all_xgboost_params(no_imfs,name_csv,linux:bool=False,verbosity:bool=False):
    params_vec = []
    for i in range(0,no_imfs):
        name_params ="params_imf"+ str(i) +"_" + str(name_csv) 
        params = load_params(name_params,linux = linux,verbosity=verbosity)
        params_vec.append(params)
        
    return params_vec
        
