from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import mutual_info_classif,VarianceThreshold
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functions_xgboost_c as fxgb

def plot_ranks(ranks,max_features:int=15,title:str=""):
    leng = len(ranks)
    names = []        
    
    if leng > max_features:
        leng = max_features
        ranks_p = ranks[0:max_features]
        for i in range(leng):
            names.append("x"+str(ranks[i]))        
    else:
        for i in range(leng):
            names.append("x"+str(ranks[i]))  
                      
    plt.barh(names[::-1],np.arange(leng),color = 'steelblue')  
    plt.ylabel("Features")     
    plt.xlabel("importance")   
    plt.title(title)
    plt.show()


def fisher_rank(x,y, plot:bool=False,no_feat_plt:int=15):
    # we want the higher fisher score, the function returns the rank in descending order (the most important feature is the first in the vector)
    ranks = fisher_score.fisher_score(x, y)        
    if plot == True:        
        plot_ranks(ranks,no_feat_plt,title="Fisher Ranks")             
            
    return ranks


def mutual_information_gain_rank(x,y,plot:bool=False,no_feat_plt:int=15):
    
    mi = mutual_info_classif(x,y)
    ranks = np.argsort(mi)[::-1]
    if plot == True:        
        plot_ranks(ranks,no_feat_plt,title="Mutual information gain")                              
     
    return ranks


def variance_threshold(x,threshold:float=0):
    #eliminate 0 variance features
    v_threshold = VarianceThreshold(threshold=threshold)
    v_threshold.fit(x)
    v_vec = v_threshold.get_support()
    new_x = x
    features_removed = []
    # remove columns that contain the features that we do not need
    cont = 0
    for i in range(x.shape[1]):        
        if v_vec[i] == False:
            features_removed.append(i)
            new_x = np.delete(new_x,i-cont,1)
            cont = cont + 1
            
        
    return new_x,features_removed


def mean_absolute_difference(x,plot:bool=False,no_feat_plt:int=15):
    mad = np.sum(np.abs(x -np.mean(x,axis=0)), axis=0)/x.shape[0]
    ranks = np.argsort(mad)[::-1]
    if plot == True:
        plot_ranks(ranks,max_features=no_feat_plt,title="MAD")        
        
    return ranks


def dispersion_ratio(x,plot:bool=False,no_feat_plt:int=15):
    x += 1 # to avoid 0 denominators
    # Artithmetic mean
    am = np.mean(x,axis=0)
    # Geometric mean
    gm = np.power(np.prod(x,axis=0),1/x.shape[0])
    # Ratio of arithmetic mean and geometric mean
    disp_ratio = am/gm
    
    ranks = np.argsort(disp_ratio)[::-1]
    
    if plot == True:
        plot_ranks(ranks,max_features=no_feat_plt,title="Dispersion ratio")        
    
    return ranks


def most_impactful_feature(x,y):
    fisher_r = fisher_rank(x, y,plot=True)
    mutual_info_gain_r = mutual_information_gain_rank(x,y,plot=True)
    mad_r = mean_absolute_difference(x, plot=True)
    #dr_r = dispersion_ratio(x,plot=True)
    
    vec = np.zeros(len(fisher_r))    
    
    for i in range(len(vec)):        
        vec[fisher_r[i]] += i
        vec[mutual_info_gain_r[i]] += i
        vec[mad_r[i]] += i
        #vec[dr_r[i]] += i
                
    best_feature = np.argsort(vec,axis=0)[0]
    
    return best_feature

def wrapper_feature_subset_selection(x,y,direction:str='forward',classifier:str='lr',scoring='accuracy',n_jobs:int=-1,cv:int=5,xgb_optimize:bool=True,verbose:bool=False):
    # Parameters
    if x.shape[0]>5:
        cv_c = cv
    else:
        cv_c = 0
        
    if direction == 'forward':
        forward_f = True
        floating_f = False
    elif direction == 'backward':
        forward_f = False
        floating_f = False
    elif direction == 'bidirectional':
        forward_f = True
        floating_f = True
        
    # Learning algorithm
    if classifier == "lr":
        lr = LogisticRegression(class_weight='balanced',solver='lbfgs',random_state=42, n_jobs=n_jobs,max_iter=1000)
        lr.fit(x,y)
        sfs = SFS(lr,k_features='best',forward=forward_f,floating=floating_f,scoring= scoring,cv=cv_c,n_jobs=n_jobs,verbose=verbose)
        
    elif classifier == "knn":
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x,y)
        sfs = SFS(knn,k_features='best',forward=forward_f,floating=floating_f,scoring=scoring,cv=cv_c,n_jobs=n_jobs,verbose=verbose)
        
    elif classifier == "xgb":      
        if xgb_optimize == True:
            params = fxgb.optimize_params(x,y,no_iterations_opt=100,no_jobs=n_jobs)
            xgb_model = XGBClassifier(**params) 
            if verbose==True:
                print("\n\nFinish optimizing parameters for xgboost")
        else:
            xgb_model = XGBClassifier()
            
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=12)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xgb_model.fit(x,y,early_stopping_rounds=10,eval_set=eval_set,verbose = verbose)        
        sfs = SFS(xgb_model,k_features='best',forward=forward_f,floating=floating_f,scoring=scoring,cv=cv_c,n_jobs=n_jobs,verbose=verbose)
        
    elif classifier == "rf":        
        sfs = SFS(lr,k_features='best',forward=forward_f,floating=floating_f,scoring=scoring,cv=cv_c,n_jobs=n_jobs,verbose=verbose)
        
    elif classifier == "svm":        
        sfs = SFS(lr,k_features='best',forward=forward_f,floating=floating_f,scoring=scoring,cv=cv_c,n_jobs=n_jobs,verbose=verbose)
    else:
        print("The wrapper does not support that classifier")
        return 0 
        
    sfs.fit(x,y)
    best_subset = np.array(list(map(int,sfs.k_feature_names_)))
    
    if verbose == True:
        print("Best subset score:",sfs.k_score_)    

    return best_subset,sfs.k_score_




#%% Test functions
"""
### Select the data
X = np.array([[81, 2, 5, 60, 1, 2], [35, 8, 4, 9, 2, 4 ], [6, 8, 50, 54, 1,8], [8, 21, 35, 41,9,4],[92,6,4,7,46,7]])
y = np.array([0, 1, 0, 0,1])
# or
from sklearn.datasets import load_breast_cancer as LBC
cancer = LBC()
X = cancer['data']
y = cancer['target']

### Functions
# Remove zero variance features 
X,features_removed = variance_threshold(X,threshold=0)    
x=X
# Feature selection functions
fisher_r = fisher_rank(X, y,plot=True)
mutual_info_gain_r = mutual_information_gain_rank(X,y,plot=True)
mad_r = mean_absolute_difference(x, plot=True)
dr_r = dispersion_ratio(x,plot=True)
best_subset_f = wrapper_feature_subset_selection(x,y,direction='bidirectional',classifier="knn",verbose=True)
treatment_feature = most_impactful_feature(x,y)
"""