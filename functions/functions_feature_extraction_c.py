from scipy.signal import find_peaks 
import numpy as np
import pandas as pd
from scipy.stats import kurtosis,skew
import functions_paths as fpt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pywt
import math

def load_timeseries_csv(name_csv,show_plot:bool=True,linux:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + name_csv + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + name_csv + ".csv"
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    
    time_series = df.to_numpy()
    time_series = time_series[0:len(time_series),0]
    show_plot = True
    if show_plot == True:
        plt.title(label = name_csv + " original time signal")
        plt.grid()    
        plt.plot(time_series)
        plt.show()
        
    return time_series


def load_timeseries_classification(name_csv,linux:bool=False,normalize:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + name_csv + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + name_csv + ".csv"
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    
    time_series = df.to_numpy()    
    target = time_series[:,-1]
    time_series_mat = time_series[:,:-1]
    if normalize == True:
        for i in range(np.shape(time_series_mat)[0]):
            time_series_mat[i,:] = normalize_vec(time_series_mat[i,:])
    
    return time_series_mat,target,df.columns


def normalize_vec(vec):
    x = len(vec)
    new_vec = np.zeros(x)
    mi = min(vec)
    ma = max(vec)    
    
    for i in range(0,x):
        new_vec[i] = (vec[i]-mi)/(ma-mi)
        
    return new_vec
        

def normalize_mat(mat):
    x,y = np.shape(mat)
    new_mat = np.zeros((x,y))
    mi = mat.min()
    ma = mat.max()
    
    for i in range(0,x):
        for j in range(0,y):            
            new_mat[i,j] = (mat[i,j]-mi)/(ma-mi)
                
    return new_mat


# This feature measures what is the error between the signal and the dominant frequency by creating a sin signal
def distance_peaks_dominant_frequency(signal,dominant_f):

    time = np.linspace(0,len(signal),len(signal));
    # Amplitude of the sine wave is sine of a variable like time    
    sin_d  = np.sin(2*np.pi*time*dominant_f)
    
    pks = find_peaks(signal)[0]   
    
    #exclude the first and the last peak to avoid outliers with borders
    if len(pks) > 1:
        if pks[0] == 0:
            pks = pks[1:]
        if pks[-1] == len(signal):
            pks = pks[:-1]        
    else:
        return 0,0,0        
        
    if len(pks) > 2:
        pks_distance = np.zeros(len(pks)-1)
        for i in range((len(pks)-1)):
            pks_distance[i] = pks[i+1] - pks[i]    
            
        var_p = np.var(pks_distance)
        avg_p = np.average(pks_distance)            
        # The problem is that sin_d has less than 2 peaks
        pks = find_peaks(sin_d)[0]        
        if len(pks) > 1:
            avg_p_sin = pks[1] - pks[0]
            diff_avg = abs(avg_p_sin - avg_p)            
        else:
            diff_avg = 0
            return var_p,avg_p,diff_avg
    else:
        var_p = 0; avg_p=0; diff_avg=0
        
    return var_p,avg_p,diff_avg


def autocorrelation(signal,verbosity:bool=False):        
    #calculate autocorrelation
    ac_signal = sm.tsa.acf(signal,nlags=int(np.ceil(len(signal)/10)),fft=False)
    ac_mean = np.mean(ac_signal)    
    if verbosity == True:
        print("Autocorrelation signal: ", ac_signal," Mean autocorrelation: ",ac_mean)
        
    return ac_signal,ac_mean    

def energy(signal):
    sum_e = 0
    for i in range(len(signal)):
        sum_e += math.pow(abs(signal[i]),2)
            
    return sum_e

def angle(signal,no_samples_behind,radian:bool=False): 
    n = no_samples_behind + 1           
    height = signal[-1] - signal[-n]
    h_n = height/n    
    
    if h_n == 0:        
        return 0        
    if radian == True:        
        rad = np.arctan(h_n)
        return rad
    else:        
        rad = np.arctan(h_n)
        degrees = (rad * 180) / np.pi
        return degrees
    
    
def divide_signal(signal,no_divisions):
    divide = no_divisions
    size = int(np.floor(len(signal)/divide))
    begin = 0
    end = size
    div_vec = []
    for i in range(divide):        
        if i == divide-1:
            div_vec.append(signal[begin:])
        else:
            div_vec.append(signal[begin:end])
            begin += size
            end += size
            
    return div_vec


def trend_peaks(signal,divide:int=2):
    trend_vec = np.zeros(divide)    
    div_vec = divide_signal(signal,divide)    
    for i in range(divide):                
        trend_vec[i] = np.amax(np.array(div_vec[i]))
                   
    trend = angle(trend_vec,1)
    return trend
    

## defined at 4 levels of decomposition because the signal is expected to have less than 224 samples 
def extract_wavelet(signal,verbosity:bool=False,levels:int=6,wavelet:str='db4'):
    # Returns the max level of decomposition with the signal
    if verbosity==True:
        max_l = pywt.dwt_max_level(len(signal), wavelet)
        print("Max number of decompositions: ", max_l)
    #Get the Low frequency coefficients (cA) and details (cD)
    cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(signal, wavelet)
    mat_coeff = []
    mat_coeff.append(cA4)            
    mat_coeff.append(cD4)
    mat_coeff.append(cD3)
    mat_coeff.append(cD2)
    mat_coeff.append(cD1)
    
    return mat_coeff


def feature_extraction_global(signal,dominant_f):    
    cont = 0
    size = 20
    features=np.zeros(size)
    
    features[cont] = np.mean(signal)
    cont += 1 #1 
    features[cont] = np.std(signal)
    cont += 1 #2
    features[cont] = np.var(signal)
    cont += 1 #3
    features[cont] = skew(signal)
    cont += 1 #4
    features[cont] = kurtosis(signal)
    cont += 1 #5
    var_s,avg_s,diff_avg = distance_peaks_dominant_frequency(signal,dominant_f)
    features[cont] = var_s
    cont += 1 #6 
    features[cont] = avg_s
    cont += 1 #7
    features[cont] = diff_avg
    cont += 1 #8
    ac_signal,ac_mean = autocorrelation(signal,verbosity=False)
    features[cont] = ac_mean
    cont += 1 #9
    features[cont] = trend_peaks(signal,2)
    cont += 1 #10    
    #New wavelet features
    mat_coeff = extract_wavelet(signal[0:200])
    features[cont] = energy(mat_coeff[0])
    cont += 1 #11
    features[cont] = energy(mat_coeff[1])
    cont += 1 #12
    features[cont] = energy(mat_coeff[2])
    cont += 1 #13
    features[cont] = energy(mat_coeff[3])
    cont += 1 #14
    features[cont] = energy(mat_coeff[4])
    cont += 1 #16
    ac_signal,ac_mean = autocorrelation(mat_coeff[0])
    features[cont] = ac_mean
    cont += 1 #17
    ac_signal,ac_mean = autocorrelation(mat_coeff[1])
    features[cont] = ac_mean
    cont += 1 #18
    ac_signal,ac_mean = autocorrelation(mat_coeff[2])
    features[cont] = ac_mean
    cont += 1 #19
    ac_signal,ac_mean = autocorrelation(mat_coeff[3])
    features[cont] = ac_mean
    cont += 1 #20
    ac_signal,ac_mean = autocorrelation(mat_coeff[4])
    features[cont] = ac_mean
    cont += 1 #21
    return features
    
    
def feature_extraction_local(signal,dominant_f,no_divisions:int=4):
    cont = 0
    size = 28
    features = np.zeros(size)
    div = no_divisions
    div_vec = divide_signal(signal,div)
    
    for i in range(div):        
        signal_div = np.array(div_vec[i])
        features[cont] = np.mean(signal_div)
        cont += 1 #1
        features[cont] = np.std(signal_div)
        cont += 1 #2
        features[cont] = np.var(signal_div)
        cont += 1 #3 
        features[cont] = skew(signal_div)
        cont += 1 #4
        features[cont] = kurtosis(signal_div)
        cont += 1 #5
        ac_signal,ac_mean = autocorrelation(signal_div,verbosity=False)
        features[cont] = ac_mean
        cont += 1 #6
        features[cont] = trend_peaks(signal_div,2)
        cont += 1 #7
        
    return features
   
    
def extract_features(imf_mat,dominant_frequencies):       
    features_imfs = []
    for j in range(np.shape(imf_mat)[0]):        
        feat_vec_global = feature_extraction_global(imf_mat[j,:],dominant_frequencies[j])        
        feat_vec_local = feature_extraction_local(imf_mat[j,:],dominant_frequencies[j])
        feat_vec = np.concatenate((feat_vec_global,feat_vec_local))
        features_imfs.append(feat_vec)
        
    size_vec = len(feat_vec)
    feat_vec = np.zeros(len(feat_vec))
    for j in range(len(features_imfs)):
        fi = np.array(features_imfs[j])
        feat_vec = np.concatenate((feat_vec,fi))
    feat_vec = feat_vec[size_vec:]
    
    return feat_vec



def get_headers_features(no_imfs,no_divisions:int=4):
    headers = []
    for i in range(no_imfs):
        # Global
        headers.append("mean_global_imf_" + str(i)) #1
        headers.append("std_global_imf_" + str(i)) #2
        headers.append("var_global_imf_" + str(i)) #3
        headers.append("skew_global_imf_"+ str(i)) #4
        headers.append("kurt_global_imf_" + str(i)) #5
        headers.append("var_s_global_imf_" + str(i)) #6 
        headers.append("avg_s_global_imf_" + str(i)) #7 
        headers.append("diff_avg_s_global_imf_" + str(i)) #8
        headers.append("mean_autocorr_global_imf_" + str(i)) #9
        headers.append("trend_peaks_global_imf_" + str(i)) #10
        headers.append("wav_energy_a4_global_imf_" + str(i)) #11
        headers.append("wav_energy_d4_global_imf_" + str(i)) #12
        headers.append("wav_energy_d3_global_imf_" + str(i)) #13
        headers.append("wav_energy_d2_global_imf_" + str(i)) #14
        headers.append("wav_energy_d1_global_imf_" + str(i)) #15
        headers.append("wav_autocorr_a4_global_imf_" + str(i)) #16
        headers.append("wav_autocorr_d4_global_imf_" + str(i)) #17
        headers.append("wav_autocorr_d3_global_imf_" + str(i)) #18
        headers.append("wav_autocorr_d2_global_imf_" + str(i)) #19
        headers.append("wav_autocorr_d1_global_imf_" + str(i)) #20
        # Local
        for j in range(no_divisions):
            headers.append("mean_local_" + str(j) + "_imf_" + str(i))
            headers.append("std_local_" + str(j) + "_imf_" + str(i))
            headers.append("var_local_" + str(j) + "_imf_" + str(i))
            headers.append("skew_local_" + str(j) + "_imf_" + str(i))
            headers.append("kurt_local_" + str(j) + "_imf_" + str(i))
            headers.append("mean_autocorr_local_" + str(j) + "_imf_" + str(i))
            headers.append("trend_peaks_local_" + str(j) + "_imf_" + str(i))            
    
    return headers


def find_begin_end(no_imf,feature_list):   
    start = 0   
    for i in range(len(feature_list)):                    
        if (feature_list[i])[-1] == str(no_imf):
            if start == 0:
                begin = i
                start = 1
                        
        if (feature_list[i])[-1] == str(no_imf+1):    
            end = i-1
            return begin,end      
        
        if (feature_list[i])[-1] == "t":
            return begin,i-1


# No. of combinations without repetition
#            n!  
#        r!(n − r)!  
def test_combinations(predictions_mat,correct_ans,no_imfs,verbosity:bool=False):
    even = False
    n = np.shape(predictions_mat)[1]
    best_combination_ac = 0
    for i in range(n):
        lenght = i+1
        vec_lin = np.linspace(1, n, num=n, dtype = int)
        comb = combinations(vec_lin, lenght)
        l_comb = list(comb)
                
        for j in range(len(l_comb)):
            avg_col = np.zeros(np.shape(predictions_mat)[0])
            for z in range(np.shape(l_comb)[1]):
                index_c = l_comb[j][z] - 1
                avg_col += predictions_mat[:,index_c]
            
            ### If it is an even number of columns and it is an exactly .5 probability, we believe the class with the individual higher accuracy            
            if np.shape(l_comb)[1] % 2 == 0:
                #individual accuracies 
                even = True
                best_z = 0
                best_col = 0               
                for z in range(np.shape(l_comb)[1]):
                    ac_z = accuracy_score(predictions_mat[:,(l_comb[j][z]-1)], correct_ans) 
                    if ac_z > best_z:
                        best_col = z
                        ac_z = best_z
                        
                        
            for z in range(np.shape(avg_col)[0]):
                if even == True:
                    if avg_col[z]/np.shape(l_comb)[1] > .5:
                        avg_col[z] = 1    
                    elif avg_col[z]/np.shape(l_comb)[1] < .5:
                        avg_col[z] = 0                
                    else:
                        avg_col[z] = predictions_mat[z,best_col]
                        
                else:
                    if avg_col[z]/np.shape(l_comb)[1] >= .5:
                        avg_col[z] = 1    
                    else:
                        avg_col[z] = 0    
            
            ac = accuracy_score(avg_col, correct_ans) 
                        
            if ac > best_combination_ac:
                best_combination_ac = ac
                pr = precision_score(avg_col, correct_ans)
                f1_s = f1_score(avg_col, correct_ans)
                rec = recall_score(avg_col, correct_ans)
                combination = l_comb[j]
                
    if verbosity==True:
        print("\nBest acuracy with the less number of features: ",best_combination_ac,"\n\n")
        print("\nBest Precision with test data: ",pr,"\n\n")                
        print("\nBest Precision with test data: ",f1_s,"\n\n")                
        print("\nBest Recall with test data: ",rec,"\n\n")
    
    return combination
























