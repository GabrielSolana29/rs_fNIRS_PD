#%% Import libraries and scripts
import pandas as pd 
import numpy as np
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import LinearSVC
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from feature_selection_ga import FeatureSelectionGA, FitnessFunction
import shap
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Import scripts
linux = False
import os
directory = os.getcwd()
if directory[0] == "/":
    linux = True
else:
    linux = False
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
import functions_feature_extraction as ffe

#%% Functions
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
    
    return xtrain,ytrain,feature_list,removed_features


def standardize(matrix):
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    return (matrix - means) / stds


def scale_0_1(matrix):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    return (matrix - mins) / (maxs - mins)


def eliminate_elements(base_list, elements_to_eliminate):
    return [x for x in base_list if x not in elements_to_eliminate]


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


def merge_datasets(name_csv1,name_csv2,name_merged_csv):
    df_1 = fld.load_csv(name_csv1,path,linux=linux)    
    df_2 = fld.load_csv(name_csv2,path,linux=linux)            
    # Append the second DataFrame to the first one
    merged_df = pd.concat([df_1,df_2], ignore_index=True)        
    merged_df['Class'] = ['1'] * 20 + ['0'] * 20
    name_csv = str("CSV\\" + name_merged_csv + ".csv")
    merged_df.to_csv(name_csv,index=False)


def merge_all_datasets(df_1,df_2,df_3,name_csv):        
    df1 = fld.load_csv(df_1,path,linux=linux)    
    df2 = fld.load_csv(df_2,path,linux=linux)     
    df3 = fld.load_csv(df_3,path,linux=linux)             
    df1 = df1.drop('Class',axis=1)
    df2 = df2.drop('Class',axis=1)
    df3 = df3.drop('Class',axis=1)                
    # Concatenate the two subsets horizontally
    merged_df = pd.concat([df1, df2,df3], axis=1)
    merged_df['Class'] = ['1'] * 20 + ['0'] * 20
    name_csv = str("CSV\\" + name_csv + ".csv")
    merged_df.to_csv(name_csv,index=False)


def generate_csv_dataset(directory,linux:bool=False):        
    path = directory + "\\CSV\\"
    # Load complete dataset
    name_csv = 'complete_dataset'
    df = fld.load_csv(name_csv,path,linux=linux)    
    dataset = df
    # Feature extraction
    no_patients = 20
    control = True    
    columns = 66
    plots = False

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
                    
    ## Save Dataset of features from control participants
    name_features,name_features_freq,name_features_connectivity = ffe.list_name_features_2(66)
    features_freq_l = pd.DataFrame(features_freq_l)
    features_freq_l.columns = name_features
    features_time_l = pd.DataFrame(features_time_l)
    features_time_l.columns = name_features_freq
    features_con_l = pd.DataFrame(features_con_l)
    features_con_l.columns = name_features_connectivity

    features_time_l.to_csv("CSV\\time_dataset_controls.csv",index=False)
    features_freq_l.to_csv("CSV\\frequency_dataset_controls.csv",index=False)
    features_con_l.to_csv("CSV\\connectivity_dataset_controls.csv",index=False)

    ### PD patients ###
    path = directory + "\\CSV\\"
    # Load complete dataset
    name_csv = 'complete_dataset'
    df = fld.load_csv(name_csv,path,linux=linux)    
    dataset = df
    # Feature extraction
    no_patients = 20
    control = False    
    columns = 66
    plots = False
    
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
            
    #   Save Dataset of features from PD patients
    name_features,name_features_freq,name_features_connectivity = ffe.list_name_features_2(66)
    features_freq_l = pd.DataFrame(features_freq_l)
    features_freq_l.columns = name_features
    features_time_l = pd.DataFrame(features_time_l)
    features_time_l.columns = name_features_freq
    features_con_l = pd.DataFrame(features_con_l)
    features_con_l.columns = name_features_connectivity

    features_time_l.to_csv("CSV\\time_dataset_pd.csv",index=False)
    features_freq_l.to_csv("CSV\\frequency_dataset_pd.csv",index=False)
    features_con_l.to_csv("CSV\\connectivity_dataset_pd.csv",index=False)
    
    # Load datasets to merge controls and PD patients            
    name_csv1 = 'time_dataset_pd'
    name_csv2 = 'time_dataset_controls'
    merge_datasets(name_csv1,name_csv2,"time_dataset_complete")
    name_csv3 = 'frequency_dataset_pd'
    name_csv4 = 'frequency_dataset_controls'
    merge_datasets(name_csv3,name_csv4,"frequency_dataset_complete")
    name_csv5 = 'connectivity_dataset_pd'
    name_csv6 = 'connectivity_dataset_controls'
    merge_datasets(name_csv5,name_csv6,"connectivity_dataset_complete")
    
    name1 = "time_dataset_complete"
    name2= "frequency_dataset_complete"
    name3 = "connectivity_dataset_complete"
    merge_all_datasets(name1,name2,name3,"complete_dataset_features")
    

#%% Main program
if __name__ == "__main__":
        
    # Set the Cross Validation folds
    cv = 10
    
    # Generate the necesary datasets for control and PD patients from the file
    # "complete_dataset.csv" stored in the CSV folder
    
    # This function will populate the CSV folder with the necesary files
    generate_csv_dataset(directory,linux=linux)
    path = directory + "\\CSV"
    # Load complete dataset
    name_csv = 'time_dataset_complete'
    df_time = fld.load_csv(name_csv,path,linux=linux)
    feature_name_time = df_time.columns

    name_csv = 'frequency_dataset_complete'
    df_freq = fld.load_csv(name_csv,path,linux=linux)
    feature_name_freq = df_freq.columns

    #%% create datasets
    tic = time.perf_counter() 

    df_arr_time = np.array(df_time)
    x = df_arr_time[:,:-1]

    df_arr_freq = np.array(df_freq)
    x2 = df_arr_freq[:,:-1]

    y = df_arr_time[:,-1]

    # Normalize datasets
    x = scale_0_1(x)
    x2 = scale_0_1(x2)

    # First lets check for correlation between the features 
    name_csv = "complete_dataset_features"    
    xtrain,ytrain,feature_list,removed_features_ = load_preprocess_dataset(name_csv,variance_t=True,linux = linux)
    # Perform standarization 
    scaler = preprocessing.StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    # Remove features with zero variance
    feature_list_ = eliminate_elements(feature_list, removed_features_)
    X = pd.DataFrame(xtrain)
    X.columns=feature_list_
    y = np.array(ytrain)
    #%% Find the features that are not highly correlated
    x_uncorrelated = eliminate_correlated_features(X,.8)
    features_uncorrelated = x_uncorrelated.columns
    # Plot the correlation matrix
    plot_correlation_matrix(x_uncorrelated)

    #%% Do the VIF analysis
    vif_data = pd.DataFrame()
    vif_data["feature"] = x_uncorrelated.columns
    vif_data["VIF"] = [variance_inflation_factor(x_uncorrelated.values, i) for i in range(len(x_uncorrelated.columns))]

    #%% Perform Elastic Net (which includes L1 and L2 regularization)
    # Create the elastic net
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.1)
    # Fit the elastic net
    elastic_net.fit(x_uncorrelated,y)
    # Access the coefficients (these represent feature importance)
    feature_importance_elastic_net = elastic_net.coef_

    #%% Perform SVM with regularization     
    # Penalties The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
    #Create the svm
    model_svm = LinearSVC(penalty='l1', dual=False)
    # Fit the svm
    model_svm.fit(x_uncorrelated,y)
    # Access the coefficientes 
    feature_importance_svm_l1 = model_svm.coef_
    #Create the svm
    model_svm = LinearSVC(penalty='l2', dual=False)
    # Fit the svm
    model_svm.fit(x_uncorrelated,y)
    # Access the coefficientes 
    feature_importance_svm_l2 = model_svm.coef_

    #%% Perform SVM with anova to find the performance with the different features used    
    # Create a feature-selection transform, a scaler and an instance of SVM that we
    # combine to have a full-blown estimator
    clf = Pipeline(
        [
            ("anova", SelectPercentile(f_classif)),        
            ("svc", SVC(gamma="auto")),
        ]
    )    
    score_means = list()
    score_stds = list()
    percentiles = (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)
        this_scores = cross_val_score(clf, x_uncorrelated, y)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds))
    plt.title("Performance of the SVM-Anova varying the percentile of features selected")
    plt.xticks(np.linspace(0, 100, 11, endpoint=True))
    plt.xlabel("Percentile")
    plt.ylabel("Accuracy Score")
    plt.axis("tight")
    plt.show()

    #%% Genetic algorithms
    class FitnessFunction:
        def __init__(self,n_total_features,n_splits = cv, alpha=0.01, *args,**kwargs):
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
            alpha = self.alpha
            total_features = self.n_total_features        
            cv_scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
            mean_accuracy = np.mean(cv_scores)
            P = mean_accuracy
            #fitness = (alpha*(1.0 - P) + (1.0 - alpha)*(1.0 - (x.shape[1])/total_features))
            return P#fitness

    def calculate_accuracy_cross_validation(model,x,y,cv:int=5):             
            model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')   
            cv_scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
            mean_accuracy = np.mean(cv_scores)
            return mean_accuracy
        
    model_lg_ga = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    ff = FitnessFunction(n_total_features= x_uncorrelated.shape[1], n_splits=5, alpha=0.05)
    fsga = FeatureSelectionGA(model_lg_ga,np.array(x_uncorrelated),y, ff_obj = ff,verbose=1)
    # Number of individuals evaluated 
    n = 1000
    # Cross over probability
    crs_prob = 0.05
    # Mutation probability
    mut_prob = 0.03
    # Number of generations 
    n_gen = 150

    fsga.generate(n,crs_prob,mut_prob,n_gen)
    best_subset_genetic_algorithm = fsga.best_ind

    # evalueate to find the accuracy 
    indices_features = np.where(np.array(best_subset_genetic_algorithm) == 1)[0]
    ac_x = np.array(x_uncorrelated)[:,indices_features]
    accuracy_genetic_algorithm = calculate_accuracy_cross_validation(model_lg_ga,ac_x,y,cv=cv)    

    # %% Filter based ffs
    plot = False
    fisher_r = ffs.fisher_rank(np.array(x_uncorrelated), y,plot=plot)
    mutual_info_gain_r = ffs.mutual_information_gain_rank(np.array(x_uncorrelated),y,plot=plot)
    mad_r = ffs.mean_absolute_difference(np.array(x_uncorrelated), plot=plot)

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

    accuracy_filters = 0
    for i in range(10):    
        list_common = find_shared_features_filters(fisher_r,mutual_info_gain_r,mad_r,(i+1))
        bsf = x_uncorrelated.iloc[:,list_common]
        model_lg_filters = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
        acc = calculate_accuracy_cross_validation(model_lg_filters,bsf,y,cv=cv)   
        if acc > accuracy_filters:
            accuracy_filters = acc
            feature_percentage = i+1
            best_subset_filters = bsf
            list_indices_filters = list_common

    #%% PCA    
    pca = PCA()
    pca.fit(x_uncorrelated)
    perc_component_variance = pca.explained_variance_ratio_
    print(perc_component_variance)
    no_components = len(perc_component_variance)

    #%% XGBoost 
    X_train, X_test, y_train, y_test = train_test_split(x_uncorrelated, y, test_size=0.25, random_state=12)

    xgb = XGBRegressor(n_estimators=100,eta=.01)
    xgb.fit(x_uncorrelated,y)

    ## First method for computing feature importance
    # Compute the feature importance. the default is gain, but it can be done with weight (the # of times that the feature is used to split data)
    xgb.feature_importances_
    plt.barh(features_uncorrelated, xgb.feature_importances_)
    sorted_idx_gain = xgb.feature_importances_.argsort()
    plt.barh(features_uncorrelated[sorted_idx_gain], xgb.feature_importances_[sorted_idx_gain])
    plt.xlabel("Xgboost Feature Importance")

    key = True
    cont = 0
    sorted_idx_ = sorted_idx_gain[::-1]
    while(key):
        if xgb.feature_importances_[sorted_idx_[cont]] != 0:
            cont += 1
        else:        
            key = False     

    sorted_idx_xgbst_gain = sorted_idx_[:cont]
    best_subset_xgb_gain = x_uncorrelated.iloc[:,sorted_idx_xgbst_gain]

    ## Second method for computing feature importance
    # Use of permutation. This allows for randomly shuffle each feature and compute the change in the models performacne.
    # The features which impact the performance the most are the most important ones.

    perm_importance = permutation_importance(xgb, X_test, y_test)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features_uncorrelated[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")

    ## The third method is with SHAP values. It is model agnostic and it estimates the how does each feature contribute to the prediction.
    # It can be used to plot more interpretation plots

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    shap.summary_plot(shap_values, X_test)

    #%% Wrapper feature subset selection 
    best_subset_lr_pear,score_lr_pear = ffs.wrapper_feature_subset_selection(np.array(x_uncorrelated),y,classifier='lr',scoring='accuracy',n_jobs=-1,cv=cv,xgb_optimize=True,verbose=True)

    #%% Analysis 

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
    feature_subset_elastic_net = x_uncorrelated.iloc[:,sorted_indices_elastic_net_]
    model_lg_elastic = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_elastic_net = calculate_accuracy_cross_validation(model_lg_elastic,feature_subset_elastic_net,y,cv=10)    

    # SVM L1 y L2
    sorted_indices_svm_l1 = np.argsort(feature_importance_svm_l1)[::-1]
    sorted_vector_svm_l1 = feature_importance_svm_l1[-1,sorted_indices_svm_l1]

    sorted_indices_svm_l2 = np.argsort(feature_importance_svm_l2)[::-1]
    sorted_vector_svm_l2 = feature_importance_svm_l2[-1,sorted_indices_svm_l2]

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
    feature_subset_svm_l1 = x_uncorrelated.iloc[:,sorted_indices_svm_l1_]

    sorted_indices_svm_l2_ = sorted_indices_svm_l2[-1,nonzero_indices_l2]
    importance_feature_subset_svm_l2 = sorted_vector_svm_l2[-1,nonzero_indices_l2]
    feature_subset_svm_l2 = x_uncorrelated.iloc[:,sorted_indices_svm_l2_]

    model_lg_svm_l1 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_svm_l1 = calculate_accuracy_cross_validation(model_lg_svm_l1,feature_subset_svm_l1,y,cv=10)    
    model_lg_svm_l2 = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_svm_l2 = calculate_accuracy_cross_validation(model_lg_svm_l2,feature_subset_svm_l2,y,cv=10)    

    #SVM anova
    # Per percentile 
    sort_score_means = np.argsort(score_means)[::-1]
    best_score_means = score_means[sort_score_means[0]]
    best_percentile = percentiles[sort_score_means[0]]

    ## SUBSET BASED
    #XGBoost
    # Gain
    model_lg_xgb_gain = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_xgb_gain = calculate_accuracy_cross_validation(model_lg_xgb_gain,best_subset_xgb_gain,y,cv=10)    
    
    #%% Ranking based Best subset
    new_list = []
    #Get the 10 percent of the best features from the elastic net
    perc = int(np.floor((len(sorted_indices_elastic_net_) * 10)/100))
    for i in range(np.shape(x_uncorrelated)[1]):
        if np.isin(i,list_indices_filters ) or np.isin(i,sorted_idx_xgbst_gain) or np.isin(i,sorted_indices_elastic_net_[:perc]) or np.isin(i,sorted_indices_svm_l1_):
            new_list.append(i)

    ranking_best_subset = x_uncorrelated.iloc[:,new_list]
    model_lg_ranking = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    accuracy_ranking = calculate_accuracy_cross_validation(model_lg_ranking,ranking_best_subset,y,cv=cv)   

    ## Wrapper with ranking based best subset
    best_subset_lr_ranking_wrapper,score_lr_ranking_wrapper = ffs.wrapper_feature_subset_selection(np.array(ranking_best_subset),y,classifier='lr',scoring='accuracy',n_jobs=20,cv=cv,xgb_optimize=True,verbose=True)

    ## Genetic algorithm with ranking_best_subset
    model_lg_ga_ranking = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
    ff_ranking = FitnessFunction(n_total_features= ranking_best_subset.shape[1], n_splits=cv, alpha=0.05)
    fsga_ranking = FeatureSelectionGA(model_lg_ranking,np.array(ranking_best_subset),y, ff_obj = ff_ranking,verbose=1)
    # Number of individuals evaluated 
    n = 1000
    # Cross over probability
    crs_prob = 0.05
    # Mutation probability
    mut_prob = 0.03
    # Number of generations 
    n_gen = 100

    fsga_ranking.generate(n,crs_prob,mut_prob,n_gen)
    best_subset_genetic_algorithm_ranking = fsga_ranking.best_ind

    # evaluate to find the accuracy 
    indices_features_ranking = np.where(np.array(best_subset_genetic_algorithm_ranking) == 1)[0]
    ac_x_ranking = np.array(ranking_best_subset)[:,indices_features_ranking]
    accuracy_genetic_algorithm_ranking = calculate_accuracy_cross_validation(model_lg_ga_ranking,ac_x_ranking,y,cv=cv)    

    #%% Report
    print("\n\nFeature Selection Report with Logistic Regression:\n")
    print("\n- Elastic net accuracy score: ", accuracy_elastic_net)
    print("\n- SVM with l1 regularization accuracy score: ", accuracy_svm_l1)
    print("\n- SVM with l2 regularization accuracy score: ", accuracy_svm_l2)
    print("\n- Filter based subset score with", feature_percentage, "% of the features: ", accuracy_filters)
    print("\n- XGB with best gain features score: ", accuracy_xgb_gain)
    print("\n- Ranking Feature-based subset score: ", accuracy_ranking)

    print("\n- Genetic Algorithm fitness/accuracy score: ", fsga.final_fitness[0][1] )
    print("\n- Wrapper Feature Subset Selection accuracy score: ", score_lr_pear)

    print("\n- Wrapper with Ranking Feature-based subset score: ", score_lr_ranking_wrapper)
    print("\n- Genetic Algorithm with Ranking Feature-based subset fitness/accuracy score: ", fsga_ranking.final_fitness[0][1] )

    print("\n- Best percentile of features from ANOVA-SVM analyisis \n          -Percentile:",best_percentile,"-Score:",best_score_means)
            


