import pandas as pd 
import numpy as np
import time
from sklearn import preprocessing
import scipy.io
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from feature_selection_ga import FeatureSelectionGA, FitnessFunction
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score    
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shap
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import mutual_info_classif,VarianceThreshold

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
    
import functions_feature_selection as ffs
import functions_paths as fpt
import load_save_data as fld
#%% Needed functions
def eliminate_correlated_features(X,threshold_):
    # Remove correlated columns
    columns_to_drop = np.zeros(5)
    while len(columns_to_drop) > 0:                
        # Calculate the correlation matrix
        correlation_matrix = X.corr()

        # Find highly correlated pairs (threshold can be adjusted)
        threshold = threshold_
        highly_correlated = (correlation_matrix.abs() > threshold) & (correlation_matrix != 1)

        # Get a set of columns to drop
        columns_to_drop = set()

        for col in highly_correlated:
            correlated_cols = list(highly_correlated.index[highly_correlated[col]])
            if correlated_cols:
                columns_to_drop.add(min(correlated_cols, key=len))  # Drop the column with the shorter name

        # Drop the columns
        X = X.drop(columns=columns_to_drop)
        
    return X 

def plot_correlation_matrix(x):
    # Plot the correlation matrix    
    f = plt.figure(figsize=(19, 15))
    plt.matshow(x.corr(), fignum=f.number)
    plt.xticks(range(x.select_dtypes(['number']).shape[1]), x.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(x.select_dtypes(['number']).shape[1]), x.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)

def load_dataset(dataset_name,linux:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + dataset_name + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + dataset_name + ".csv" # Windows
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    
    df = df
    class_vec = df.Class
    #subject_vec = df.Id
        
    #df = df.drop('Id', axis=1)
    df = df.drop('Class', axis=1)
    
    return class_vec,df

def load_preprocess_dataset(name_csv,variance_t:bool=False,remove_feature_list:list=[],linux:bool=False):
    ytrain,xtrain = load_dataset(name_csv,linux = linux)
    feature_list = xtrain.columns 
    xtrain = np.array(xtrain)       
    removed_features = []    
    ### Pre processing 
    if variance_t ==True:
        # remove features with 0 variance 
        xtrain,features_removed = ffs.variance_threshold(xtrain)
        # Remove the items from the lists        
        feature_list = list(feature_list)
        for i in range(len(features_removed)):
            removed_features.append(feature_list[features_removed[i]])
            
    if len(remove_feature_list):           
        df = pd.DataFrame(xtrain)        
        df.columns = feature_list[:-1]        
        for i in range(len(feature_list)):            
            if feature_list[i] in remove_feature_list:
                df = df.drop(feature_list[i],1)
                                  
        xtrain = np.array(df)
        feature_list = list(df.columns)
        feature_list.append("target")                             
            
    #min_max_scaler = preprocessing.MinMaxScaler()
    #xtrain = min_max_scaler.fit_transform(xtrain)
    
    return xtrain,ytrain,feature_list,removed_features

def scale_0_1(matrix):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    return (matrix - mins) / (maxs - mins)

def eliminate_elements(base_list, elements_to_eliminate):
    return [x for x in base_list if x not in elements_to_eliminate]

def calculate_accuracy_cross_validation(model,x,y,cv:int=5):            
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
    mean_accuracy = np.mean(cv_scores)
    return mean_accuracy   

def find_shared_features_filters(fisher,mutual_inf_gain,mad,perc):
    # At least two filters should share the feature in the top percentage 
    no_features = int(np.floor((perc*len(fisher))/100))
    fisher_local = np.array(fisher)[:no_features]
    mutual_inf_local = np.array(mutual_inf_gain)[:no_features]
    mad_local = np.array(mad)[:no_features]
    list_common_elements = []
    for i in range(len(fisher)):
        if np.isin(i,mad_local) and np.isin(i,mutual_inf_local) or np.isin(i,fisher_local):
            list_common_elements.append(i)
        
        elif np.isin(i,mutual_inf_local) and np.isin(i,mad_local) or np.isin(i,fisher_local):
            list_common_elements.append(i)
            
        elif np.isin(i,fisher) and np.isin(i,mad_local) or np.isin(i,mutual_inf_local):
            list_common_elements.append(i)
            
    return list_common_elements

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


def feature_selection_analysis(name_csv,directory,eliminate_correlation:bool=True,correlation_threshold:float=0.8,alpha_elastic_net:float=0.1,l1_ratio_elastic_net:float=0.1,n_ga:int=50,crs_prob_ga:float=0.05,mut_prob_ga:float=0.01,n_gen_ga:int=20,standardize_data:bool=True,verbose:bool=False):
    path = directory + "\\CSV"
    
    if eliminate_correlation:        
        xtrain,ytrain,feature_list,removed_features_ = load_preprocess_dataset(name_csv,variance_t=True,linux = linux)
        if standardize_data:
            # Perfrorm standarization 
            scaler = preprocessing.StandardScaler().fit(xtrain)
            xtrain = scaler.transform(xtrain)
        # remove features with zero variance
        feature_list_ = eliminate_elements(feature_list, removed_features_)
        X = pd.DataFrame(xtrain)
        X.columns=feature_list_
        # Find the features that are not highly correlated
        x = eliminate_correlated_features(X,correlation_threshold)        
        features_uncorrelated = x.columns
        y = np.array(ytrain)                        
           
    else:
        xtrain,ytrain,feature_list,removed_features_ = load_preprocess_dataset(name_csv,variance_t=False,linux = linux)
        if standardize_data:
            # Perfrorm standarization 
            xtrain = preprocessing.StandardScaler().fit(xtrain)
        X = pd.DataFrame(xtrain)
        X.columns=feature_list
        x = X
        y = np.array(ytrain)        
        
    #########################################################################################################################
    # Perform Elastic Net (Which includes L1 and L2)
    # Create the elastic net
    elastic_net = ElasticNet(alpha=alpha_elastic_net, l1_ratio=l1_ratio_elastic_net)
    # Fit the elastic net
    elastic_net.fit(x,y)
    # Access the coefficients (these represent feature importance)
    feature_importance_elastic_net = elastic_net.coef_
    
    #########################################################################################################################
    # Perform SVM with regularization 
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC    
    # Penalties The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
    
    #Create the svm
    model_svm = LinearSVC(penalty='l1', dual=False)
    # Fit the svm
    model_svm.fit(x,y)
    # Access the coefficientes 
    feature_importance_svm_l1 = model_svm.coef_

    #Create the svm
    model_svm = LinearSVC(penalty='l2', dual=False)
    # Fit the svm
    model_svm.fit(x,y)
    # Access the coefficientes 
    feature_importance_svm_l2 = model_svm.coef_
    
    #########################################################################################################################
    # Perform SVM with anova to find the performance with the different features used
    # https://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html#sphx-glr-auto-examples-svm-plot-svm-anova-py

    # Create a feature-selection transform, a scaler and an instance of SVM that we
    # combine together to have a full-blown estimator

    clf = Pipeline(
        [
            ("anova", SelectPercentile(f_classif)),
        # ("scaler", StandardScaler()),
            ("svc", SVC(gamma="auto")),
        ]
    )    
    score_means = list()
    score_stds = list()
    percentiles = (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)
        this_scores = cross_val_score(clf, x, y)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    
    sort_score_means = np.argsort(score_means)[::-1]
    best_score_means = score_means[sort_score_means[0]]
    best_percentile = percentiles[sort_score_means[0]]

    #########################################################################################################################
    # Genetic algorithms
    # https://pypi.org/project/feature-selection-ga/    
    class FitnessFunction:
        def __init__(self,n_total_features,n_splits = 10, alpha=0.01, *args,**kwargs):
            """
                Parameters
                -----------
                n_total_features :int
                    Total number of features N_t.
                n_splits :int, default = 5
                    Number of splits for cv
                alpha :float, default = 0.01
                    Tradeoff between the classifier performance P and size of
                    feature subset N_f with respect to the total number of features
                    N_t.

                verbose: 0 or 1
            """
            self.n_splits = n_splits
            self.alpha = alpha
            self.n_total_features = n_total_features

        def calculate_fitness(self,model,x,y):
            #alpha = self.alpha
            #total_features = self.n_total_features
            cv_scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
            mean_accuracy = np.mean(cv_scores)
            fitness = mean_accuracy                        
            return fitness
         
    model_lg_ga = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    ff = FitnessFunction(n_total_features= x.shape[1], n_splits=5, alpha=0.05)
    fsga = FeatureSelectionGA(model_lg_ga,np.array(x),y, ff_obj = ff,verbose=1)
    # Number of individuals evaluated n_ga     
    # Cross over probability crs_prob_ga     
    # Mutation probability mut_prob_ga    
    # Number of generations n_gen_ga     
    fsga.generate(n_ga,crs_prob_ga,mut_prob_ga,n_gen_ga)
    best_subset_genetic_algorithm = fsga.best_ind
    # evalueate to find the accuracy     
    indices_features = np.where(np.array(best_subset_genetic_algorithm) == 1)[0]    
    ac_x = np.array(x)[:,indices_features]
    accuracy_genetic_algorithm= calculate_accuracy_cross_validation(model_lg_ga,ac_x,y,cv=10)    
    
    #########################################################################################################################
    # Filter based fs
    plot = verbose
    fisher_r = fisher_rank(np.array(x), y,plot=plot)
    mutual_info_gain_r = mutual_information_gain_rank(np.array(x),y,plot=plot)
    mad_r = mean_absolute_difference(np.array(x), plot=plot)
    
    accuracy_filters = 0
    for i in range(10):    
        list_common = find_shared_features_filters(fisher_r,mutual_info_gain_r,mad_r,(i+1))
        bsf = x.iloc[:,list_common]
        model_lg_filters = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
        acc = calculate_accuracy_cross_validation(model_lg_filters,bsf,y,cv=10)   
        if acc > accuracy_filters:
            accuracy_filters = acc
            feature_percentage = i+1
            list_indices_filters = list_common
            best_subset_filters = bsf
        
    #########################################################################################################################
    # XGBoost
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)
    xgb = XGBRegressor(n_estimators=100,eta=.01)
    xgb.fit(x,y)
    ## First method for computing feature importance
    # Compute the feature importance. the default is gain, but it can be done with weight (the # of times that the feature is used to split data)
    xgb.feature_importances_
    sorted_idx_gain = xgb.feature_importances_.argsort()
    
    key = True
    cont = 0
    sorted_idx_ = sorted_idx_gain[::-1]
    while(key):
        if xgb.feature_importances_[sorted_idx_[cont]] != 0:
            cont += 1
        else:        
            key = False     
    
    sorted_idx_xgbst_gain = sorted_idx_[:cont]
    best_subset_xgb_gain = x.iloc[:,sorted_idx_xgbst_gain]        
    model_lg_xgb_gain = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_xgb_gain = calculate_accuracy_cross_validation(model_lg_xgb_gain,best_subset_xgb_gain,y,cv=10)    

    ## Second method for computing feature importance
    # Use of permutation. This allows for randomly shuffle each feature and compute the change in the models performacne.
    # The features which impact the performance the most are the most important ones.
    perm_importance = permutation_importance(xgb, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()    
        
    ## The third method is with SHAP values. It is model agnostic and it estimates the how does each feature contribute to the prediction.
    # It can be used to plot more interpretation plots        
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    
    #########################################################################################################################        
    # Wrapper feature subset selection 
    best_subset_lr_pear,score_lr_pear = ffs.wrapper_feature_subset_selection(np.array(x),y,classifier='lr',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
                
    #########################################################################################################################
    # Ranking based Best subset    
    new_list = []
    # Get the 10 percent of the best features from the elastic net
    perc = int(np.floor((len(sorted_indices_elastic_net_) * 10)/100))
    # Create the list with the features from the different ranking systems
    for i in range(np.shape(x)[1]):
        if np.isin(i,list_indices_filters ) or np.isin(i,sorted_idx_xgbst_gain) or np.isin(i,sorted_indices_elastic_net_[:perc]) or np.isin(i,sorted_indices_svm_l1_):
            new_list.append(i)

    ranking_best_subset = x.iloc[:,new_list]
    model_lg_ranking = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_ranking = calculate_accuracy_cross_validation(model_lg_ranking,ranking_best_subset,y,cv=10)   

    ## Wrapper with ranking based best subset
    best_subset_lr_ranking_wrapper,score_lr_ranking_wrapper = ffs.wrapper_feature_subset_selection(np.array(ranking_best_subset),y,classifier='lr',scoring='accuracy',n_jobs=20,cv=5,xgb_optimize=True,verbose=True)
    
    ## Genetic algorithm with ranking_best_subset
    model_lg_ga_ranking = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    ff_ranking = FitnessFunction(n_total_features= ranking_best_subset.shape[1], n_splits=5, alpha=0.05)
    fsga_ranking = FeatureSelectionGA(model_lg_ranking,np.array(ranking_best_subset),y, ff_obj = ff_ranking,verbose=1)
    # Number of individuals evaluated 
    n = 500
    # Cross over probability
    crs_prob = 0.05
    # Mutation probability
    mut_prob = 0.03
    # Number of generations 
    n_gen = 50
    fsga_ranking.generate(n,crs_prob,mut_prob,n_gen)
    best_subset_genetic_algorithm_ranking = fsga_ranking.best_ind
    # evaluate to find the accuracy 
    indices_features_ranking = np.where(np.array(best_subset_genetic_algorithm_ranking) == 1)[0]
    ac_x_ranking = np.array(ranking_best_subset)[:,indices_features_ranking]
    accuracy_genetic_algorithm_ranking = calculate_accuracy_cross_validation(model_lg_ga_ranking,ac_x_ranking,y,cv=10)    
    
    #########################################################################################################################
    ## Analysis 
    # Elastic net
    sorted_indices_elastic_net = np.argsort(feature_importance_elastic_net)[::-1]
    sorted_vector_elastic_net = feature_importance_elastic_net[sorted_indices_elastic_net]
    # Keep the non zero elements of the vector 
    nonzero_indices_elastic_net = []
    for i in range(len(sorted_indices_elastic_net)):
        if sorted_vector_elastic_net[i] != 0:
            nonzero_indices_elastic_net.append(i)
        if sorted_vector_elastic_net[i] != 0:
            nonzero_indices_elastic_net.append(i)

    sorted_indices_elastic_net_ = sorted_indices_elastic_net[nonzero_indices_elastic_net]
    importance_feature_subset_elastic_net = sorted_vector_elastic_net[nonzero_indices_elastic_net]
    feature_subset_elastic_net = x.iloc[:,sorted_indices_elastic_net_]
    model_lg_elastic = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_elastic_net = calculate_accuracy_cross_validation(model_lg_elastic,feature_subset_elastic_net,y,cv=10)    

    # SVM L1 y L2    
    sorted_indices_svm_l1 = np.argsort(feature_importance_svm_l1)[::-1]
    sorted_vector_svm_l1 = feature_importance_elastic_net[sorted_indices_svm_l1]
    
    sorted_indices_svm_l2 = np.argsort(feature_importance_svm_l2)[::-1]
    sorted_vector_svm_l2 = feature_importance_elastic_net[sorted_indices_svm_l2]
    # Keep the non zero elements of the vector and 
    nonzero_indices_l1 = []
    nonzero_indices_l2 = []
    for i in range(np.shape(sorted_indices_svm_l1)[1]):
        if sorted_vector_svm_l1[-1,i] != 0:
            nonzero_indices_l1.append(i)
        if sorted_vector_svm_l2[-1,i] != 0:
            nonzero_indices_l2.append(i)
        
    sorted_indices_svm_l1_ = sorted_indices_svm_l1[-1,nonzero_indices_l1]
    importance_feature_subset_svm_l1 = sorted_vector_svm_l1[-1,nonzero_indices_l1]
    feature_subset_svm_l1 = x.iloc[:,sorted_indices_svm_l1_]
    
    sorted_indices_svm_l2_ = sorted_indices_svm_l2[-1,nonzero_indices_l2]
    importance_feature_subset_svm_l2 = sorted_vector_svm_l2[-1,nonzero_indices_l2]
    feature_subset_svm_l2 = x.iloc[:,sorted_indices_svm_l2_]
    
    model_lg_svm_l1 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_svm_l1 = calculate_accuracy_cross_validation(model_lg_svm_l1,feature_subset_svm_l1,y,cv=10)    
    model_lg_svm_l2 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_svm_l2 = calculate_accuracy_cross_validation(model_lg_svm_l2,feature_subset_svm_l2,y,cv=10)    
        
    if verbose:
        print("\n\nFeature Selection Report with Logistic Regression:\n\n")
        print("\n- Elastic net accuracy score: ", accuracy_elastic_net)
        print("\n- SVM with l1 regularization accuracy score: ", accuracy_svm_l1)
        print("\n- SVM with l2 regularization accuracy score: ", accuracy_svm_l2)
        print("\n- Filter based subset score with", feature_percentage, "% of the features: ", accuracy_filters)
        print("\n- XGB with best gain features score: ", accuracy_xgb_gain)
        print("\n- Ranking Feature-based subset score: ", accuracy_ranking)
        
        print("\n- Genetic Algorithm fitness/accuracy score: ", fsga.final_fitness )
        print("\n- Wrapper Feature Subset Selection accuracy score: ", score_lr_pear)
        
        print("\n- Wrapper with Ranking Feature-based subset score: ", score_lr_ranking_wrapper)
        print("\n- Genetic Algorithm with Ranking Feature-based subset fitness/accuracy score: ", fsga_ranking.final_fitness[0][1] )
                
        print("\n- Best percentile of features from ANOVA-SVM analyisis \n          -Percentile:",best_percentile,"-Score:",best_score_means)                
        
        print("\n Correlation reduction analysis:")
         # Plot the correlation matrix
        plot_correlation_matrix(x)
        # Do the VIF analysis
        vif_data = pd.DataFrame()
        vif_data["feature"] = x.columns
        vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]            
                
        print("\n ANOVA based analysis with SVM:")
        plt.errorbar(percentiles, score_means, np.array(score_stds))
        plt.title("Performance of the SVM-Anova varying the percentile of features selected")
        plt.xticks(np.linspace(0, 100, 11, endpoint=True))
        plt.xlabel("Percentile")
        plt.ylabel("Accuracy Score")
        plt.axis("tight")
        plt.show()
        
        print("\n Plots of the XGBoost based analysis:")
        
        print("\n 1- Gain based:")
        plt.barh(features_uncorrelated, xgb.feature_importances_)
        plt.barh(features_uncorrelated[sorted_idx_gain], xgb.feature_importances_[sorted_idx_gain])
        plt.xlabel("Xgboost Feature Importance")
        
        print("\n 2- Permutation importance:")
        plt.barh(features_uncorrelated[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance")
        
        print("\n 3- SHAP based:")
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        shap.summary_plot(shap_values, X_test)

    return best_subset_genetic_algorithm_ranking
           

def filter_based_genetic_algorithm_feature_subset_selection(name_csv,eliminate_correlation:bool=True,correlation_threshold:float=0.8,alpha_elastic_net:float=0.1,l1_ratio_elastic_net:float=0.1,n_ind_ga:int=500,n_gen_ga:int=50,crs_prob_ga:float=0.05,mut_prob_ga:float=0.01,standardize_data:bool=True,verbose:bool=False):    
    if eliminate_correlation:        
        xtrain,ytrain,feature_list,removed_features_ = load_preprocess_dataset(name_csv,variance_t=True,linux = linux)
        if standardize_data:
            # Perfrorm standarization 
            scaler = preprocessing.StandardScaler().fit(xtrain)
            xtrain = scaler.transform(xtrain)
        # remove features with zero variance
        feature_list_ = eliminate_elements(feature_list, removed_features_)
        X = pd.DataFrame(xtrain)
        X.columns=feature_list_
        # Find the features that are not highly correlated
        x = eliminate_correlated_features(X,correlation_threshold)        
        features_uncorrelated = x.columns
        y = np.array(ytrain)                        
            
    else:
        xtrain,ytrain,feature_list,removed_features_ = load_preprocess_dataset(name_csv,variance_t=False,linux = linux)
        if standardize_data:
            # Perfrorm standarization 
            xtrain = preprocessing.StandardScaler().fit(xtrain)
        X = pd.DataFrame(xtrain)
        X.columns=feature_list
        x = X
        y = np.array(ytrain)        
        
    #########################################################################################################################
    # Perform Elastic Net (Which includes L1 and L2)
    # Create the elastic net
    elastic_net = ElasticNet(alpha=alpha_elastic_net, l1_ratio=l1_ratio_elastic_net)
    # Fit the elastic net
    elastic_net.fit(x,y)
    # Access the coefficients (these represent feature importance)
    feature_importance_elastic_net = elastic_net.coef_

    #########################################################################################################################
    # Perform SVM with regularization 
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC    
    # Penalties The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.    
    #Create the svm
    model_svm = LinearSVC(penalty='l1', dual=False)
    # Fit the svm
    model_svm.fit(x,y)
    # Access the coefficientes 
    feature_importance_svm_l1 = model_svm.coef_        
    #########################################################################################################################
    # Filter based fs
    plot = verbose
    fisher_r = ffs.fisher_rank(np.array(x), y,plot=plot)
    mutual_info_gain_r = ffs.mutual_information_gain_rank(np.array(x),y,plot=plot)
    mad_r = ffs.mean_absolute_difference(np.array(x), plot=plot)

    accuracy_filters = 0
    for i in range(10):    
        list_common = find_shared_features_filters(fisher_r,mutual_info_gain_r,mad_r,(i+1))
        bsf = x.iloc[:,list_common]
        model_lg_filters = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
        acc = calculate_accuracy_cross_validation(model_lg_filters,bsf,y,cv=10)   
        if acc > accuracy_filters:
            accuracy_filters = acc
            feature_percentage = i+1
            list_indices_filters = list_common
            best_subset_filters = bsf
        
    #########################################################################################################################
    # XGBoost
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)
    xgb = XGBRegressor(n_estimators=100,eta=.01)
    xgb.fit(x,y)
    ## First method for computing feature importance
    # Compute the feature importance. the default is gain, but it can be done with weight (the # of times that the feature is used to split data)
    xgb.feature_importances_
    sorted_idx_gain = xgb.feature_importances_.argsort()

    key = True
    cont = 0
    sorted_idx_ = sorted_idx_gain[::-1]
    while(key):
        if xgb.feature_importances_[sorted_idx_[cont]] != 0:
            cont += 1
        else:        
            key = False     

    sorted_idx_xgbst_gain = sorted_idx_[:cont]
    best_subset_xgb_gain = x.iloc[:,sorted_idx_xgbst_gain]        
    model_lg_xgb_gain = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_xgb_gain = calculate_accuracy_cross_validation(model_lg_xgb_gain,best_subset_xgb_gain,y,cv=10)    

    #########################################################################################################################
    ## Analysis 
    # Elastic net
    sorted_indices_elastic_net = np.argsort(feature_importance_elastic_net)[::-1]
    sorted_vector_elastic_net = feature_importance_elastic_net[sorted_indices_elastic_net]
    # Keep the non zero elements of the vector 
    nonzero_indices_elastic_net = []
    for i in range(len(sorted_indices_elastic_net)):
        if sorted_vector_elastic_net[i] != 0:
            nonzero_indices_elastic_net.append(i)
        if sorted_vector_elastic_net[i] != 0:
            nonzero_indices_elastic_net.append(i)

    sorted_indices_elastic_net_ = sorted_indices_elastic_net[nonzero_indices_elastic_net]
    importance_feature_subset_elastic_net = sorted_vector_elastic_net[nonzero_indices_elastic_net]
    feature_subset_elastic_net = x.iloc[:,sorted_indices_elastic_net_]
    model_lg_elastic = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_elastic_net = calculate_accuracy_cross_validation(model_lg_elastic,feature_subset_elastic_net,y,cv=10)    

    # SVM L1 y L2    
    sorted_indices_svm_l1 = np.argsort(feature_importance_svm_l1)[::-1]
    sorted_vector_svm_l1 = feature_importance_elastic_net[sorted_indices_svm_l1]
        
    # Keep the non zero elements of the vector and 
    nonzero_indices_l1 = []    
    for i in range(np.shape(sorted_indices_svm_l1)[1]):
        if sorted_vector_svm_l1[-1,i] != 0:
            nonzero_indices_l1.append(i)
        
    sorted_indices_svm_l1_ = sorted_indices_svm_l1[-1,nonzero_indices_l1]
    importance_feature_subset_svm_l1 = sorted_vector_svm_l1[-1,nonzero_indices_l1]
    feature_subset_svm_l1 = x.iloc[:,sorted_indices_svm_l1_]

    model_lg_svm_l1 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_svm_l1 = calculate_accuracy_cross_validation(model_lg_svm_l1,feature_subset_svm_l1,y,cv=10)    

    #########################################################################################################################
    # Ranking based Best subset    
    new_list = []
    # Get the 10 percent of the best features from the elastic net
    perc = int(np.floor((len(sorted_indices_elastic_net_) * 10)/100))
    # Create the list with the features from the different ranking systems
    for i in range(np.shape(x)[1]):
        if np.isin(i,list_indices_filters ) or np.isin(i,sorted_idx_xgbst_gain) or np.isin(i,sorted_indices_elastic_net_[:perc]) or np.isin(i,sorted_indices_svm_l1_):
            new_list.append(i)

    ranking_best_subset = x.iloc[:,new_list]
    model_lg_ranking = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_ranking = calculate_accuracy_cross_validation(model_lg_ranking,ranking_best_subset,y,cv=10)   

    ## Genetic algorithm with ranking_best_subset    
    # https://pypi.org/project/feature-selection-ga/    
    class FitnessFunction:
        def __init__(self,n_total_features,n_splits = 10, alpha=0.01, *args,**kwargs):
            """
                Parameters
                -----------
                n_total_features :int
                    Total number of features N_t.
                n_splits :int, default = 5
                    Number of splits for cv
                alpha :float, default = 0.01
                    Tradeoff between the classifier performance P and size of
                    feature subset N_f with respect to the total number of features
                    N_t.

                verbose: 0 or 1
            """
            self.n_splits = n_splits
            self.alpha = alpha
            self.n_total_features = n_total_features

        def calculate_fitness(self,model,x,y):
            #alpha = self.alpha
            #total_features = self.n_total_features
            cv_scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')
            mean_accuracy = np.mean(cv_scores)
            fitness = mean_accuracy                        
            return fitness
    model_lg_ga_ranking = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    ff_ranking = FitnessFunction(n_total_features= ranking_best_subset.shape[1], n_splits=5, alpha=0.05)
    fsga_ranking = FeatureSelectionGA(model_lg_ranking,np.array(ranking_best_subset),y, ff_obj = ff_ranking,verbose=1)
    # Number of individuals evaluated     
    # Cross over probability    
    # Mutation probability
    # Number of generations 
    fsga_ranking.generate(n_ind_ga,crs_prob_ga,mut_prob_ga,n_gen_ga)
    best_subset_genetic_algorithm_ranking = fsga_ranking.best_ind
    # evaluate to find the accuracy 
    indices_features_ranking = np.where(np.array(best_subset_genetic_algorithm_ranking) == 1)[0]
    ac_x_ranking = np.array(ranking_best_subset)[:,indices_features_ranking]
    accuracy_genetic_algorithm_ranking = calculate_accuracy_cross_validation(model_lg_ga_ranking,ac_x_ranking,y,cv=10)  

    if verbose:
        print("\n- Genetic Algorithm with Ranking Feature-based subset fitness/accuracy score: ", fsga_ranking.final_fitness[0][1] )   

    return best_subset_genetic_algorithm_ranking,model_lg_ga_ranking  


# Load complete dataset
name_csv = "complete_dataset_complete"
best_subset_ga, model_logistic_regression = filter_based_genetic_algorithm_feature_subset_selection(name_csv,n_gen_ga=100,n_ind_ga=500,crs_prob_ga=0.05,mut_prob_ga=0.03,verbose=True)

