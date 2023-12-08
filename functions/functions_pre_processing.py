## Feature extraction 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import blackman
from scipy.stats import kurtosis,skew
import statsmodels.api as sm
from scipy.signal import find_peaks
from PyEMD import CEEMDAN
#from scipy.signal import decimate
import functions_paths as fpt
import functions_signal_processing_analysis_c as fspa
#%% functions

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


def initialize_CEEMDAN(trials_f:int=100,epsilon_f:float=0.005,parallel_f:bool=False,processes_f:int=1,noise_scale_f:float=1.0):
    ## We are adding a noise seed for reproducibility 
    # epsilon:float (default: 0.005) Scale for added noise (ϵ) which multiply std σ: β=ϵ⋅σ    
    ceemdan = CEEMDAN(trials=trials_f, epsilon=epsilon_f, parallel=parallel_f, processes=processes_f)
    ceemdan.noise_seed(0)    
    return ceemdan


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
        #plt.grid()    
        plt.plot(time_series)
        plt.show()
        
    return time_series


def load_timeseries_fmri(name_csv,roi,patient,show_plot:bool=True,linux:bool=False):
    path_name = fpt.path_CSV(linux=linux)
    if linux == True:       
        complete_name = path_name + "/" + name_csv + ".csv" # linux
    else:        
        complete_name = path_name + "\\" + name_csv + ".csv"
    
    df = pd.read_csv(
        complete_name, 
        na_values=['NA', '?'])
    
    time_series = df.to_numpy()
    
    for i in range(np.shape(time_series)[0]):
        if patient == time_series[i,-1]:
            row_p = i 
            break
    row = row_p + (roi-1)        
    time_series = time_series[row,:-1]
    
    show_plot = True
    if show_plot == True:
        plt.title(label = "patient: " + str(patient) + " roi: " + str(roi))
        plt.grid()    
        plt.plot(time_series)
        plt.show()
        
    return time_series


def ceemdan_feature_label_extraction(time_series,limit_no_points_feature_extraction:int=10,percentage:int=99.9999,show_plots:bool=True,features:bool=False,point_seconds:float=1,verbosity:bool=False):
    t = np.linspace(0, 1, len(time_series))
    #% Obtain the IMFs from the time series
    ceemdan_f = initialize_CEEMDAN(trials_f=100,parallel_f=False)
    
    # Execute CEEMDAN on signal
    imf_mat = ceemdan_f.ceemdan(time_series,t)
    if verbosity == True:
        print("\nFinished Ceemdan")    
    percentage = 100
    #% Obtain the vectors with dominant frequencies
    dominant_frequencies = dominant_frequencies_imfs(imf_mat,percentage,show_plots,point_seconds=point_seconds)     
    periods = sinusoid_periods(dominant_frequencies)
    
    no_points_feature_extraction_vec = np.ones(len(imf_mat)) * limit_no_points_feature_extraction
    if verbosity == True:
        print("\nFinished sampling distances")
    #% Get the training labels and features  

    xtrain_list = training_points_imfs_no_points(imf_mat,no_points_feature_extraction_vec)                      
    
    if features == True:
        print("features are not enabled check the function")
    #    features_mat = training_features_imfs(imf_mat,sampling_distances,no_points_feature_extraction=no_points_feature_extraction_vec[0])
    #    return xtrain_list,ytrain_list,sampling_distances,dominant_frequencies,imf_mat,no_points_feature_extraction_vec,features_mat

    if features == False:
        return xtrain_list,periods,dominant_frequencies,imf_mat,no_points_feature_extraction_vec


def dominant_frequencies_imfs(imf_mat,show_plot:bool=False,windowing:bool=True,point_seconds:float=1):
    total_imfs = len(imf_mat)        
    dominant_frequencies_vec = np.zeros(total_imfs)    
        
    for i in range(0,total_imfs-1):
        current_imf = imf_mat[i,:]     
        no_frequencies = 1
        dominant_frequencies_vec[i] = dominant_frequency(current_imf,no_frequencies,show_plot,windowing)                
                
    return dominant_frequencies_vec    
    
        
def dominant_frequency(signal,no_freq:int=1,show_plot:bool=False,windowing:bool=True,point_seconds:float=1):    
    #https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html#d-discrete-fourier-transforms
    # Number of sample points
        
    N = len(signal)
    # sample spacing
    T = point_seconds  # (sampling rate)     
    
    if windowing==True:        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
    else:
        w = 1
        
    signal_f = fft(signal*w)    
    freqs = fftfreq(N, T)[:N//2] 
    positive_real_fs = np.abs(signal_f[0:N//2]) # only positive real values
    freq_step = freqs[1]
   
    # plot positive real half of the spectrum 
    if show_plot == True:
        #plot n values of the spectrum
        n = 50
        plt.plot(freqs[range(0,n)],  positive_real_fs[range(0,n)])    
        if windowing == True:
            plt.title(label="Spectrum " + "with blackman window")
        else:
            plt.title(label="Spectrum " + "without blackman window")
        plt.grid()    
        plt.show()
    
    sorted_indices = np.argsort(positive_real_fs)
    
    ans = np.zeros(no_freq)
    cont = 0
    for i in range(1,no_freq+1):        
        ans[cont] = sorted_indices[(i * -1)] * freq_step
        cont = cont + 1        

    return ans

def sinusoid_periods(dominant_frequency):
    period = np.zeros(len(dominant_frequency))
    
    for i in range(len(dominant_frequency)):
        if dominant_frequency[i] != 0:
            period[i] = ((2 * np.pi) / (dominant_frequency[i])) 
        else:
            period[i] = 0
                        
    return period
        
        
def get_training_label_nyq(signal,sampling_distance):    
    no_training_s = int(len(signal) - sampling_distance)
    ytrain = np.zeros(no_training_s)
    for i in range(no_training_s):
        position = int(i + sampling_distance)
        ytrain[i] = signal[position]
        
    return ytrain


def training_points_imfs_no_points(imf_mat,no_points_feature_extraction_vec):    
    x = len(imf_mat)
    points = []               
    for i in range(x):                    
        signal = imf_mat[i,0:]        
        x_signal = get_n_samples_for_feature_extraction(signal,int(no_points_feature_extraction_vec[i]))
        points.append(x_signal)
    
    return points


def get_n_samples_for_feature_extraction(signal,n):
    x_list = []
    end_for = int(len(signal)-n+1) #from the current point i can predict n days in the future
    for i in range(end_for):          
        start = i
        end = i + n
        x_list.append(signal[range(start,end)])
        
    return x_list

def xtrain_for_training(xtrain_list,sampling_distances):
    ### For training we need to remove the sampling_distance[i] last features of the xtrain_list
    for i in range(len(xtrain_list)):
        xtrain_list[i] = xtrain_list[i][:-int(sampling_distances[i])]
        
    return xtrain_list  
    

