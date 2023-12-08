#%% Train XgBoost
import numpy as np
from xgboost import XGBRegressor
import pickle
import os
#from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score,KFold, RandomizedSearchCV, train_test_split
import functions_feature_extraction as ffe
import functions_paths as fpt
import functions_signal_processing_analysis as fspa


def display_scores_mse(scores):
    print("\n10 fold CrossValidation \nMean: {0:.3f}\nStd: {1:.3f}".format(np.mean(scores), np.std(scores)))

def optimize_params(xtrain,ytrain,no_jobs:int = -1, no_iterations_opt:int=250):
     #split the data
   
    xgb_opt = XGBRegressor()
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)
    #eval_set = [(X_train, y_train), (X_test, y_test)]
    #eval_metric = ["rmse"]
    eval_metric = ["mphe"]
    #eval_metric = ["mae"]
    # Find the best parameters          
                   
    params = {
        "colsample_bytree": np.arange(0.3,1.03,.1),                        
        "min_child_weight": np.arange(0,6,1),
        "gamma": np.arange(0,1.2,.2),
        "learning_rate": np.arange(0.01,0.25,.02),
        "max_depth": np.arange(2,9,1), 
        "n_estimators": np.arange(100,155,5),
        "subsample": np.arange(0.4,1.05,.1),
        "lambda": np.arange(0,3.25,.2),            
        "alpha": np.arange(0,3.25,.2)               
        
    }                

    # Define the parameters of the search
    search = RandomizedSearchCV(xgb_opt, param_distributions=params, n_iter=no_iterations_opt, cv=5, verbose=1, n_jobs=no_jobs, return_train_score=True, scoring='neg_mean_squared_error')
    # Search    
    #search.fit(X_train, y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=False)          
    search.fit(xtrain, ytrain,eval_metric=eval_metric,verbose=False)          
    # Save the optimized parameters        
    opt_params = search.best_params_
    
    opt_params['seed'] = 0
    #opt_params['eval_metric']= 'rmse'
    opt_params['eval_metric']= 'mphe'
    #opt_params['eval_metric']= 'mae'
    opt_params['booster'] = 'gbtree'    
    opt_params['base_score'] = .5  
    opt_params['colsample_bytree'] = 1             
    #set the optimized parameters
    
    return opt_params
   
def train_regression_xgboost(xtrain,ytrain,epochs:int=200,params:dict={},verbosity:bool=False,erly_stpping_rounds:int=10,gpu:bool=False):    
            
    if bool(params) == False:
        #If parameters are not provided, default parameters are given        
        params['seed'] = 0
        #params['eval_metric']= 'rmse'
        params['eval_metric']= 'mphe'
        #params['eval_metric']= 'mae'
        params['n_estimators'] = epochs
        params['booster'] = 'gbtree'    
        params['base_score'] = .5
        params['colsample_bytree'] = 1                 
        
    params['booster'] = 'gbtree'  
    params['n_estimators'] = epochs   
    
    if gpu==True:
        params['tree_method'] = 'gpu_hist'
        
    print("\nTraining xgboost_model...")    
    #set the optimized parameters        
    xgb_model = XGBRegressor(**params) 
            
    if erly_stpping_rounds >= 10:        
        X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xgb_model.fit(xtrain,ytrain,early_stopping_rounds=erly_stpping_rounds,eval_set=eval_set,verbose = verbosity)
    else:
        xgb_model.fit(xtrain,ytrain,verbose=verbosity)    
    
                  
    if verbosity == True:            
        #fspa.correlation_between_signals(predictions,ytrain,verbosity=True)
        print("\nFinished training")
    
    return xgb_model

    
"""    
def regression_arq(xtrain,ytrain,optimize:bool=True,no_jobs:int = -1,xgb_params:dict={},show_cv:bool=True,no_iterations_opt:int=250):        
    #split the data
   
    xgb_opt = XGBRegressor()
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.20, random_state=12)
    #eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_metric = ["rmse"]
    # Find the best parameters          
    if optimize == True:                    
        params = {
            "colsample_bytree": np.arange(0.3,1.03,.1),                        
            "min_child_weight": np.arange(0,6,1),
            "gamma": np.arange(0,1.2,.2),
            "learning_rate": np.arange(0.01,0.25,.02),
            "max_depth": np.arange(2,9,1), 
            "n_estimators": np.arange(100,155,5),
            "subsample": np.arange(0.4,1.05,.1),
            "lambda": np.arange(0,3.25,.2),            
            "alpha": np.arange(0,3.25,.2)               
            
        }                

        # Define the parameters of the search
        search = RandomizedSearchCV(xgb_opt, param_distributions=params, n_iter=no_iterations_opt, cv=5, verbose=1, n_jobs=no_jobs, return_train_score=True, scoring='neg_mean_squared_error')
        # Search    
        #search.fit(X_train, y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=False)          
        search.fit(xtrain, ytrain,eval_metric=eval_metric,verbose=False)          
        # Save the optimized parameters        
        opt_params = search.best_params_
        
    else:
        opt_params = xgb_params
    
    opt_params['seed'] = 0
    opt_params['eval_metric']= 'rmse'
    opt_params['booster'] = 'gbtree'    
    opt_params['base_score'] = .5  
    opt_params['colsample_bytree'] = 1             
    #set the optimized parameters
    xgb_model = XGBRegressor(**opt_params)            
    #xgb_model.fit(X_train, y_train,eval_metric=eval_metric,eval_set=eval_set,verbose=False)    
    xgb_model.fit(xtrain,ytrain,eval_metric=eval_metric,verbose=False)
    predictions = xgb_model.predict(xtrain)
    #training score
    #training_score = xgb_model.score(xtrain,ytrain)   
     
    #10 fold cross validation
    kfold = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(xgb_model, xtrain, ytrain, scoring="neg_mean_squared_error", cv=kfold)
    display_scores_mse(np.sqrt(-scores))    
    #print("parameters 2:",xgb_model.get_params())
    print("\nCorrelation: \n", np.corrcoef(ytrain,predictions))
    return predictions,opt_params,xgb_model
"""

def classification_arq(optimize:bool=True):
    return


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def convert_0_1(pred):
    new_vec = np.zeros(len(pred))
    for i in range(len(pred)):
        if pred[i] > 0.5:
            new_vec[i] = 1.0
        else:
            new_vec[i] = 0.0
    return new_vec

def metrics_predictions(pred,ytrain):
    tp=0;tn=0;fp=0;fn=0;
    for i in range(len(pred)):
        if pred[i] == 1.0 and ytrain[i] == 1.0:
            tp = tp + 1
        elif pred[i] == 0.0 and ytrain[i] == 1.0:
            fn = fn + 1
        elif pred[i] == 1.0 and pred[i] == 0.0:
            fp = fp + 1
        elif pred[i] == 0.0 and pred[i] == 0.0:
            tn = tn + 1
                
    accuracy = safe_div((tp + tn),(tp + fp + tn + fn))
    sensitivity = safe_div(tp,(tp + fn))
    specificity = safe_div(tn,(tn + fp))
    precision = safe_div(tp,(tp + fp))
    new_vec = np.zeros(4)
    new_vec[0]=accuracy;new_vec[1]=sensitivity;new_vec[2]=specificity;new_vec[3]=precision;
    
    return new_vec

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

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
    
    model = XGBRegressor()
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


def train_save_xgboost_models_params(no_imf,xtrain_list,ytrain_list,name_csv,epochs:int=500,no_iterations_opt:int=300,no_points_feature_extraction:int=10,linux:bool=False,verbosity:bool=False,gpu:bool=False):
    current_imf = no_imf
    name_model = "xgboost_imf"+ str(current_imf) +"_" + str(name_csv)
    name_params ="params_imf"+ str(current_imf) +"_" + str(name_csv) 
    # Get the data to train the model 
    xtrain,ytrain = ffe.get_xtrain_ytrain(xtrain_list,ytrain_list,no_imf)        
    # Find optimal parameters for xgb model
    opt_params = optimize_params(xtrain,ytrain,no_iterations_opt=no_iterations_opt)
    save_params(opt_params, name_params,linux = linux)
    # Train the regression xgb model     
    
    xgb_model = train_regression_xgboost(xtrain,ytrain,params=opt_params,epochs=epochs,verbosity=verbosity,gpu=gpu)
    
    # Save xgboost model        
    save_model(xgb_model, name_model,linux = linux)
    
    if verbosity == True:
        print("Model ",no_imf," trained and saved")
        predictions = xgb_model.predict(xtrain)
        print("\nCorrelation imf ", no_imf, " : " , "\n", np.corrcoef(ytrain,predictions),"\n\n")
        
        
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
        
        

















