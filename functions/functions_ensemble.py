import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from random import randrange

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
    
import load_save_data as fld
import functions_xgboost_c as fxgb
#%% Functions 

def get_subsets(algo:str="knn"):
    
    path_best_features = directory + "\\saved_model_params"    
    if algo=="knn":            
        name_s = "best_subset_knn"
    elif algo=="lr":
        name_s = "best_subset_lr"
    elif algo=="xgb":
        name_s = "best_subset_xgb"
    else:
        print("Cannot open with that algorithm ")
        return
        
    name_subset = name_s + "_freq"
    subset_freq = fld.load_csv(name_subset,path_best_features,linux=linux)
    subset_freq = subset_freq.values
    subset_freq = np.ravel(subset_freq)

    name_subset = name_s + "_time"
    subset_time = fld.load_csv(name_subset,path_best_features,linux=linux)
    subset_time = subset_time.values
    subset_time = np.ravel(subset_time)

    name_subset = name_s + "_pearson"
    subset_conn = fld.load_csv(name_subset,path_best_features,linux=linux)
    subset_conn = subset_conn.values
    subset_conn = np.ravel(subset_conn)

    return subset_time, subset_freq, subset_conn


def ensemble_pred(x,x2,x3,y,subset_time,subset_freq,subset_conn,algo:str='knn',test_size:float=0.3,random_state_s:bool=True, xgb_optimize:bool=True,verbose:bool=True):       
    
    if random_state_s:
        ran_state = randrange(100)
    else:
        ran_state = 12
        
    # Create training and evaluation sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=ran_state)
    X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y, test_size=test_size, random_state=ran_state)
    X3_train, X3_test, y3_train, y3_test = train_test_split(x3, y, test_size=test_size, random_state=ran_state)

    x_train = X_train[:,subset_time]
    x_test = X_test[:,subset_time]
    
    x2_train = X2_train[:,subset_freq]
    x2_test = X2_test[:,subset_freq]
    
    x3_train = X3_train[:,subset_conn]
    x3_test = X3_test[:,subset_conn]
    
    if algo == "knn":                 
        clsf1 = KNeighborsClassifier(n_neighbors=3)
        clsf2 = KNeighborsClassifier(n_neighbors=3)
        clsf3 = KNeighborsClassifier(n_neighbors=3)  
        
    elif algo == "lr": 
        clsf1 = LogisticRegression(class_weight='balanced', penalty = 'l2', random_state=42, n_jobs=-1,max_iter=1000)
        clsf2 = LogisticRegression(class_weight='balanced', penalty = 'l2', random_state=42, n_jobs=-1,max_iter=1000)
        clsf3 = LogisticRegression(class_weight='balanced', penalty = 'l2', random_state=42, n_jobs=-1,max_iter=1000)
        
    elif algo == "xgb":
        if xgb_optimize:
            params1 = fxgb.optimize_params(x,y,no_iterations_opt=100,no_jobs=-1)
            params2 = fxgb.optimize_params(x2,y,no_iterations_opt=100,no_jobs=-1)
            params3 = fxgb.optimize_params(x3,y,no_iterations_opt=100,no_jobs=-1)
            xgb_model1 = XGBClassifier(**params1) 
            xgb_model2 = XGBClassifier(**params2) 
            xgb_model3 = XGBClassifier(**params3) 
            if verbose==True:
                print("\n\nFinish optimizing parameters for xgboost")
        else:
            xgb_model1 = XGBClassifier()                    
            xgb_model2 = XGBClassifier()                    
            xgb_model3 = XGBClassifier()                    
            
        clsf1=xgb_model1           
        clsf2=xgb_model2
        clsf3=xgb_model3              
            
    clsf1.fit(x_train,y_train)
    pred1 = clsf1.predict(x_test)
    
    clsf2.fit(x2_train,y_train)
    pred2 = clsf2.predict(x2_test)            
    
    clsf3.fit(x3_train,y_train)
    pred3 = clsf3.predict(x3_test)


    # Make the predictions based on the three votes
    pred_ensemble = np.zeros(len(y_test))

    for i in range(len(y_test)):
        cont = 0
        if pred1[i] == 1:
            cont += 1
        if pred2[i] == 1:
            cont += 1
        if pred3[i] == 1:
            cont += 1
                    
        if cont >= 2:
            pred_ensemble[i] = 1
        
        
    # check how many predictions where correct
    cont = 0
    for i in range(len(y_test)):
        if y_test[i] == pred_ensemble[i]:
            cont += 1

    if verbose:
        print(" Results with ", algo, ": ", cont, " correct out of ", len(y_test))

    return pred_ensemble