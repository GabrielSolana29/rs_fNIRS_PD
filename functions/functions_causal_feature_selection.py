#Feature selection 
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
#from econml.dml import CausalForestDML
#from sklearn.linear_model import MultiTaskLassoCV,LassoCV
import time
import math
#from sklearn.model_selection import cross_val_score,KFold, RandomizedSearchCV, train_test_split
import random
import shap
import functions_feature_selection as ffs
from scipy.special import softmax

def causal_forest_feature_selection(df,name_outcome,most_important_feature:str="",n_estimators:int=5000,optimize_parameters:bool=True,shap_values:bool=False,verbose:bool=False):
    tic = time.perf_counter()
    #train, test = train_test_split(df, test_size=0.2)
    feature_list = list(df.columns.values)
    
    if name_outcome in feature_list:
        feature_list.remove(name_outcome)        
    
    if most_important_feature == "":
        treatment = random.choice(feature_list)
    else:        
        treatment = most_important_feature
        
    if treatment != "target" and treatment != "Class" and treatment != "class":            
        feature_list.remove(treatment)
        
    covariates = feature_list
    Y = df[name_outcome] # outcome
    X = df[covariates]
    T = df[treatment]
    W = None # Counfounders            
    #X_test = test[covariates]
    
    #Initialize
    #c_forest = CausalForestDML(criterion='het',n_estimators=n_estimators,cv=5,model_t=LassoCV(max_iter=10000), model_y=LassoCV(max_iter=10000))        
    c_forest = CausalForestDML(criterion='het',n_estimators=n_estimators,cv=5)  
    
    if optimize_parameters == True:        
        c_forest.tune(Y, T, X=X, W = W)    
             
    c_forest.fit(Y, T, X=X, W=W)
    # estimate the CATE with the test set 
    #c_forest.const_marginal_ate(X_test)

    if verbose == True:        
        toc = time.perf_counter()
        print("\nTime elapsed: ",(toc-tic), "seconds")            
        
    heterogeneous_effect_feature_importance = c_forest.feature_importances_
    
    # sort from smallest contribution to largest contribution with shap values 
    sort_index = np.argsort(heterogeneous_effect_feature_importance)    
    feature_importance = []
    for i in range(len(heterogeneous_effect_feature_importance)):
        feature_importance.append((heterogeneous_effect_feature_importance[sort_index[i]],feature_list[sort_index[i]]))
        # Include the treatment
    feature_importance.append((1,treatment))
    
    if shap_values == True:                  
        shap_values_c = c_forest.shap_values(X)                
        shap.summary_plot(shap_values_c[name_outcome][treatment].values,covariates)
        shap.summary_plot(shap_values_c[name_outcome][treatment].values,covariates,plot_type="bar")
        shap_values_c = shap_values_c[name_outcome][treatment].values
        
        # Calculate the feature importance (mean absolute shap value) for each feature
        importances = []    
        for i in range(np.shape(shap_values_c)[1]):
            importances.append(np.mean(np.abs(shap_values_c[:, i])))
                    
        # Calculate the normalized version
        importances_norm = softmax(importances)            
        shap_sorted = np.argsort(importances_norm)
        
        # return the names of the features (inices may be diferent because we remove the treatment)
        shap_features = []       
        for i in range(np.shape(shap_sorted)[0]):
            shap_features.append(feature_list[shap_sorted[i]])                        
        # Include the treatment
        shap_features.append(treatment)                 
        return feature_importance, shap_features
            
    return feature_importance


def causal_feature_selection_pipeline(x,y,feature_list:list=[],target_name:str="target",filter_treatment:str='info_gain',variance_threshold:float=0.0,wrapper:bool=True,w_algorithm:str='lr',drop_percentage:int=50,shap_value:bool=False,n_jobs:int=-1,verbose:bool=True):        
    
    if verbose == True:
        tic = time.perf_counter()
        
    #%%  ####---- If features did not include its name, assign a name ----####    
    if bool(feature_list) == False:
        for i in range(x.shape[1]):
            name = "x" + str(i)
            feature_list.append(name)
    else:
        if target_name in feature_list:
            feature_list.remove(target_name)        
            
    #%%  ####---- Remove features with less variance than threshold ----####    
    x,features_removed = ffs.variance_threshold(x,threshold=variance_threshold)
        
    removed_features = []          
    feature_list = list(feature_list)
    for i in range(len(features_removed)):
        removed_features.append(feature_list[features_removed[i]])     
        
    for i in range(len(removed_features)):                
        feature_list.remove(removed_features[i]) 
         
    #%%  ####---- Filtering stage ----####    
    if filter_treatment == 'info_gain':
        mutual_info_gain_r = ffs.mutual_information_gain_rank(x,y,plot=verbose)
        treatment = feature_list[mutual_info_gain_r[0]]
        
    elif filter_treatment == 'fisher_rank':
        fisher_r = ffs.fisher_rank(x,y,plot=verbose)
        treatment = feature_list[fisher_r[0]]
        
    elif filter_treatment == 'MAD':
        mad_r = ffs.mean_absolute_difference(x, plot=verbose)
        treatment = feature_list[mad_r[0]]
        
    ### WARNING DISPERSION RATIO METHOD CHANGES THE DATA SO NOT RECOMMENDED AT ALL ###    
    #elif filter_treatment == 'dispersion_ratio':
    #    dr_r = ffs.dispersion_ratio(x,plot=verbose)
    #    treatment = feature_list[dr_r[0]]
        
    elif filter_treatment == 'ensemble':
        mif_r = ffs.most_impactful_feature(x,y)
        treatment = feature_list[mif_r]
        
    #%%  ####---- Causal forest feature selection ----####    
    df = pd.DataFrame(x)
    df.insert(loc=np.shape(df)[1], column= target_name, value=y)
    feature_list.append(target_name)
    df.columns = feature_list
    
    if shap_value==True:
        feature_importance,shap_values = causal_forest_feature_selection(df, target_name,n_estimators=10000, most_important_feature=treatment, shap_values=shap_value, optimize_parameters=True)            
    else:
        feature_importance = causal_forest_feature_selection(df, target_name,n_estimators=10000, most_important_feature=treatment, shap_values=shap_value, optimize_parameters=True)            
    
    if verbose == True:
        print("\n\n######### Causal inference complete ###########\n\n")        
        
    #%% ####---- Drop the a percentage of non relevant features ----####
    if shap_value == True:
        ### Use the shap feature importance from the causal forest
        f_i = np.array(shap_values)
        no_f_delete = int(math.floor((drop_percentage * len(shap_values)) / 100))        
        list_delete = list(f_i[0:no_f_delete])
        for i in range(len(list_delete)):
            df.drop(list_delete[i], inplace=True, axis=1)  
            
        df.drop(target_name, inplace=True, axis=1)
        x = np.array(df)
        
    else:
        ### Use the heterogeneous feature importance from the causal forest 
        f_i = np.array(feature_importance)
        no_f_delete = int(math.floor((drop_percentage * len(feature_importance)) / 100))        
        list_delete = list(f_i[0:no_f_delete,1])
        for i in range(len(list_delete)):
            df.drop(list_delete[i], inplace=True, axis=1)  
            
        df.drop(target_name, inplace=True, axis=1)
        x = np.array(df)
        
    #%% ####---- Wrapper feature subset selection ----####
    if verbose == True:
        print("\n\n##### Begin Wrapper Feature Subset selection ######")
        
    if wrapper == True:
        best_subset = ffs.wrapper_feature_subset_selection(x,y,direction='bidirectional',classifier=w_algorithm,n_jobs=n_jobs,verbose=verbose)
    else:
        best_subset = feature_importance
        
    #%%% ###--- Create new subset ---###    
    feature_list = []
    for i in range(len(best_subset)):
        feature_list.append(df.columns[best_subset[i]])
    
    best_subset_x = np.zeros((x.shape[0],len(best_subset)))
    for i in range(len(best_subset)):
        best_subset_x[:,i] = x[:,best_subset[i]]
    
    df = pd.DataFrame(best_subset_x)
    df.columns = feature_list
    
    if verbose == True:
        print("\n\n##### Wrapper Feature Subset selection complete ######")
        
    #%% Return
    if verbose == True:
        toc = time.perf_counter()
        print("\nTime elapsed: ",(toc-tic), "seconds")
        
    if shap_value == True:
        return df, feature_importance, shap
    else:
        return df, feature_importance


def construct_dataset(feature_importance,df,drop_no_features:int=0,min_contribution:float=0,drop_percentage:float=0):
    
    if min_contribution > 0:
        f_i = np.array(feature_importance)[::-1]
        for i in range(len(f_i)):
            if float(f_i[i,0]) < min_contribution:
                list_delete = list(f_i[i:,1])  
                for j in range(len(list_delete)):
                    df.drop(list_delete[j], inplace=True, axis=1)  
                                               
                return df
            
    elif drop_no_features > 0:        
        f_i = np.array(feature_importance)
        list_delete = list(f_i[0:drop_no_features,1])
        for i in range(len(list_delete)):
            df.drop(list_delete[i], inplace=True, axis=1)        
            
    elif drop_percentage > 0:
        f_i = np.array(feature_importance)
        no_f_delete = int(math.floor((drop_percentage * len(feature_importance)) / 100))        
        list_delete = list(f_i[0:no_f_delete,1])
        for i in range(len(list_delete)):
            df.drop(list_delete[i], inplace=True, axis=1)  
    
    return df


def plot_feature_heterogeneity_importance(feature_importance,no_values:int=20):
    
    fi_value = []
    fi_name = []
    for i in range(np.shape(feature_importance)[0]):    
        fi_value.append(feature_importance[i][0])
        fi_name.append(feature_importance[i][1])                
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    # Example data
    fi_name = fi_name[-(no_values+1):-1][::-1]
    fi_value = fi_value[-(no_values+1):-1][::-1]
    y_pos = fi_name
    
    ax.barh(y_pos, fi_value, align='edge')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.yaxis
    ax.set_xlabel('Feature importance')
    ax.set_title('Parameter heterogeneity')
    
    plt.show()
















