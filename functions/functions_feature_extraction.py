## Feature extraction 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import blackman
from scipy.stats import kurtosis,skew
import statsmodels.api as sm
import seaborn
from scipy.signal import find_peaks
import functions_paths as fpt
#%% functions

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


#%% Functions specific of fnirs feature extraction 


def dominant_frequency(signal,sampling_rate:float=1.0,no_freq:int=1,show_plot:bool=False,windowing:bool=True):    
    # Number of sample points
    N = len(signal)
    # sample spacing
    T = 1.0 / sampling_rate  # Inverse of the sampling rate for the fftfreq according to the scipy documentation    
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


def fft_var_energy_mean_skew_kur(signal,windowing:bool=True):
    N = len(signal)
    #T = 1
    if windowing == True:
        w = blackman(N)
    else:
        w = 1
    #T = 1/ sampling_rate
    signal_f = fft(signal*w)
    #freqs =fftfreq(N,T)[:N//2]
    positive_real_fs = np.abs(signal_f[0:N//2])
    #freq_step = freqs[1]
    fft_var = np.var(positive_real_fs)
    fft_energy = energy_parseval(positive_real_fs)
    fft_mean = np.mean(positive_real_fs)
    fft_skew = skew(positive_real_fs)
    fft_kur = kurtosis(positive_real_fs)
    return fft_var,fft_energy,fft_mean,fft_skew,fft_kur


def peaks_after_zerocross_timeseries(signal,plot:bool=False):
    ### This function returns the maximum peaks and valleys between two zero cross sections
    
    peaks_position,peak_heights = find_peaks(abs(signal))
    #detect zero cross
    vec_zerocross = np.where(np.diff(np.sign(signal)))[0]
    
    vec_real_peaks = []
    
    for i in range(len(peaks_position)):        
        peak = peaks_position[i]
        
        if i == 0:
            vec_real_peaks.append(peak)
        elif i == len(peaks_position)-1:            
            nada = 0
        else:        
            # Find the two closer zero cross
            prev_z = 0
            next_z = 0        
            j = 0                        
            while 1:                                                  
                if vec_zerocross[j] > peak:    
                    if j == 0:
                        prev_z = 0                        
                    else:
                        prev_z = vec_zerocross[j-1]                            
                    next_z = vec_zerocross[j]                
                    break  
                elif j == len(vec_zerocross)-1:
                    prev_z = vec_zerocross[-1]                
                    break
                
                j += 1            
            
            if vec_real_peaks[-1] > prev_z:
                if abs(signal[vec_real_peaks[-1]]) < abs(signal[peak]):
                    vec_real_peaks[-1] = peak
            else:
                vec_real_peaks.append(peak)

    if plot==True:
        plt.plot(abs(signal))
    
    return vec_real_peaks



def avg_time_to_peak(signal,sampling_rate):    
    peak_vec = peaks_after_zerocross_timeseries(signal)
    vec_zerocross = np.where(np.diff(np.sign(signal)))[0]
    time_vec = []
    for i in range(len(peak_vec)):        
        peak = peak_vec[i]
        j = 0
        if i > 0:
            while 1:
                if vec_zerocross[j] > peak:
                    prev_z = vec_zerocross[j-1]                            
                    next_z = vec_zerocross[j]  
                    break
                elif j == len(vec_zerocross)-1:
                    prev_z = vec_zerocross[-1]
                    
                    break
                
                j +=1
            
            time_v = (peak - prev_z) * (1/sampling_rate)
            time_vec.append(time_v)
            
    return np.average(time_vec)
    
    
def time_between_zerocrossing(signal,sampling_rate):
    
    vec_zerocross = np.where(np.diff(np.sign(signal)))[0]
    vec_times = []
    for i in range(len(vec_zerocross)):
        if i > 0:
             time = (vec_zerocross[i] - vec_zerocross[i-1]) * (1/sampling_rate)
             vec_times.append(time)
    avg_zerocrossing = np.average(vec_times)
    return avg_zerocrossing
    

def extract_features_from_points(signal,sampling_rate):    
    features = np.zeros(12)
    cont = 0  #0                
    features[cont] = np.mean(signal)
    cont += 1 #1 
    features[cont] = np.var(signal)
    cont += 1 #2    
    features[cont] = np.std(signal)
    cont += 1 #3    
    features[cont] = np.max(abs(signal))
    cont += 1 #4
    features[cont] = time_between_zerocrossing(signal,sampling_rate)
    cont += 1 #5   
    features[cont] = avg_time_to_peak(signal,sampling_rate)
    cont += 1 #6
    v,e,m,s,k = fft_var_energy_mean_skew_kur(signal)
    features[cont] = v
    cont += 1 #7 
    features[cont] = e
    cont += 1 #8 
    features[cont] = m
    cont += 1 #9
    features[cont] = s
    cont += 1 #10
    features[cont] = k
    cont += 1 #11       
    features[cont] = dominant_frequency(signal,sampling_rate=sampling_rate,no_freq=1,show_plot=False)
    cont += 1 #12    

    return features


def find_correlation_matrix(array,plot:bool=False):
    X = pd.DataFrame(array)
    X = X.astype(float)
    correlation = X.corr(method ='pearson',numeric_only = True)  
    
    if plot:
        heatmap = seaborn.heatmap(correlation, annot = False)  
        heatmap.set(xlabel='IRIS values on x axis',ylabel='IRIS values on y axis\t',title ="Correlation matrix of IRIS dataset\n")  
        plt.show()  
    
    return correlation


def find_most_correlatedsignals(patient):
    
    patient_arr = np.array(patient[:,1:])
    hbo = []
    hbr = []
    hbt = []
    cont = 0 
    for i in range(np.shape(patient_arr)[1]):        
        if cont == 0:
            hbo.append(patient_arr[:,i])
        elif cont == 1:
            hbr.append(patient_arr[:,i])
        else:
            hbt.append(patient_arr[:,i])
        
        if cont < 2:
            cont += 1
        else:
            cont = 0 
        
    hbo_arr = np.zeros((np.shape(patient_arr)[0],len(hbo)))
    hbr_arr = np.zeros((np.shape(patient_arr)[0],len(hbo)))
    hbt_arr = np.zeros((np.shape(patient_arr)[0],len(hbo)))
    
    for i in range(len(hbo)):
        hbo_arr[:,i] = hbo[i]
        hbr_arr[:,i] = hbr[i]
        hbt_arr[:,i] = hbt[i]
        
    corr_mat_hbo = find_correlation_matrix(hbo_arr)
    corr_mat_hbr = find_correlation_matrix(hbr_arr)
    corr_mat_hbt = find_correlation_matrix(hbt_arr)
    corr_mat_hbo = pd.DataFrame(corr_mat_hbo)
    corr_mat_hbr = pd.DataFrame(corr_mat_hbr)
    corr_mat_hbt = pd.DataFrame(corr_mat_hbt)
        
    corr_value = np.zeros(3)
    pair_index = []
    for i in range(3)        :
        if i == 0 :
            corr_mat = corr_mat_hbo
        if i == 1:
            corr_mat = corr_mat_hbr
        if i == 2:
            corr_mat = corr_mat_hbt
        #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)    
        sort_index_value = (corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
                   .stack()
                   .sort_values(ascending=False))
                
        indexes_list = []
        highest_corr_list = []
        
        for index, value in sort_index_value.items():
            indexes_list.append(index)
            highest_corr_list.append(value)
        
        corr_value[i] = highest_corr_list[0]
        pair_index.append(indexes_list[0])
        
    
    return corr_value,pair_index
    
def most_active_band(patient):
    patient_arr = np.array(patient[:,1:])
    
    hbo = []
    hbr = []
    hbt = []
    cont = 0 
    for i in range(np.shape(patient_arr)[1]):        
        if cont == 0:
            hbo.append(patient_arr[:,i])
        elif cont == 1:
            hbr.append(patient_arr[:,i])
        else:
            hbt.append(patient_arr[:,i])
        
        if cont < 2:
            cont += 1
        else:
            cont = 0 
        
    hbo_arr = np.zeros((np.shape(patient_arr)[0],len(hbo)))
    hbr_arr = np.zeros((np.shape(patient_arr)[0],len(hbo)))
    hbt_arr = np.zeros((np.shape(patient_arr)[0],len(hbo)))
    
    global_max_hbo = 0
    band_hbo = 0
    for i in range(np.shape(hbo_arr)[1]):
        current_max = np.max(hbo_arr)
        if global_max_hbo <= current_max:
            global_max_hbo = current_max
            band_hbo = i 
    
    global_max_hbr = 0
    band_hbr = 0
    for i in range(np.shape(hbr_arr)[1]):
        current_max = np.max(hbo_arr)
        if global_max_hbr <= current_max:
            global_max_hbr = current_max
            band_hbr = i 
    
    global_max_hbt = 0
    band_hbt = 0
    for i in range(np.shape(hbt_arr)[1]):
        current_max = np.max(hbt_arr)
        if global_max_hbt <= current_max:
            global_max_hbt = current_max
            band_hbt = i 
    
    return band_hbo,band_hbr,band_hbt

def extract_features_connectivity(patient,sampling_rate):
    
    ## Get the most correlated indices from the three bands
    corr_value,pair_index = find_most_correlatedsignals(patient)
    hbo_corr = corr_value[0]    
    hbr_corr = corr_value[1]    
    hbt_corr = corr_value[2]    
    index_hbo_1 = pair_index[0][0]
    index_hbo_2 = pair_index[0][1]
    index_hbr_1 = pair_index[1][0]
    index_hbr_2 = pair_index[1][1]
    index_hbt_1 = pair_index[2][0]
    index_hbt_2 = pair_index[2][1]
        
    features = np.zeros(12)
    cont = 0  #0                
    features[cont] = hbo_corr
    cont += 1 #1 
    features[cont] = hbr_corr
    cont += 1 #2    
    features[cont] = hbt_corr
    cont += 1 #3
    features[cont] = index_hbo_1
    cont += 1 #4 
    features[cont] = index_hbo_2
    cont += 1 #5    
    features[cont] = index_hbr_1
    cont += 1 #6
    features[cont] = index_hbr_2
    cont += 1 #7 
    features[cont] = index_hbt_1
    cont += 1 #8    
    features[cont] = index_hbt_2
    cont += 1 #9
    
    band_hbo,band_hbr,band_hbt = most_active_band(patient)
    
    features[cont] = band_hbo
    cont += 1 #10 
    features[cont] = band_hbr
    cont += 1 #11   
    features[cont] = band_hbt
    cont += 1 #12
    
    return features


def extract_features_time(signal,sampling_rate):    
    features = np.zeros(6)
    cont = 0  #0                
    features[cont] = np.mean(signal)
    cont += 1 #1 
    features[cont] = np.var(signal)
    cont += 1 #2    
    features[cont] = np.std(signal)
    cont += 1 #3    
    features[cont] = np.max(abs(signal))
    cont += 1 #4
    features[cont] = time_between_zerocrossing(signal,sampling_rate)
    cont += 1 #5   
    features[cont] = avg_time_to_peak(signal,sampling_rate)
    cont += 1 #6
    
    return features


def extract_features_frequency(signal,sampling_rate):    
    features = np.zeros(6)
    cont = 0  #0                
    v,e,m,s,k = fft_var_energy_mean_skew_kur(signal)
    features[cont] = v
    cont += 1 #7 
    features[cont] = e
    cont += 1 #8 
    features[cont] = m
    cont += 1 #9
    features[cont] = s
    cont += 1 #10
    features[cont] = k
    cont += 1 #11       
    features[cont] = dominant_frequency(signal,sampling_rate=sampling_rate,no_freq=1,show_plot=False)
    cont += 1 #12    

    return features


def list_name_features(columns):
    list_features = []
    signal_features = ["mean","var","std","max_value","avg_time_between_zero_crossing","avg_time_to_peak","var_fft","energy_fft","mean_fft","skew_fft","kur_fft","dominant_freq"]
    connectivity_features = ["hbo_max_corr","hbr_max_corr","hbt_max_corr","index_hbo_1","index_hbo_2","index_hbr_1","index_hbr_2","index_hbt_1","index_hbt_2","most_active_band_hbo","most_active_band_hbr","most_active_band_hbt"]
    pearson_features =["hbo_1_0","hbo_2_0","hbo_2_1","hbo_3_0","hbo_3_1","hbo_3_2","hbo_4_0","hbo_4_1","hbo_4_2","hbo_4_3","hbo_5_0","hbo_5_1","hbo_5_2","hbo_5_3","hbo_5_4","hbo_6_0","hbo_6_1","hbo_6_2","hbo_6_3","hbo_6_4","hbo_6_5","hbo_7_0","hbo_7_1","hbo_7_2","hbo_7_3","hbo_7_4","hbo_7_5","hbo_7_6","hbo_8_0","hbo_8_1","hbo_8_2","hbo_8_3","hbo_8_4","hbo_8_5","hbo_8_6","hbo_8_7","hbo_9_0","hbo_9_1","hbo_9_2","hbo_9_3","hbo_9_4","hbo_9_5","hbo_9_6","hbo_9_7","hbo_9_8","hbo_10_0","hbo_10_1","hbo_10_2","hbo_10_3","hbo_10_4","hbo_10_5","hbo_10_6","hbo_10_7","hbo_10_8","hbo_10_9","hbo_11_0","hbo_11_1","hbo_11_2","hbo_11_3","hbo_11_4","hbo_11_5","hbo_11_6","hbo_11_7","hbo_11_8","hbo_11_9","hbo_11_10","hbo_12_0","hbo_12_1","hbo_12_2","hbo_12_3","hbo_12_4","hbo_12_5","hbo_12_6","hbo_12_7","hbo_12_8","hbo_12_9","hbo_12_10","hbo_12_11","hbo_13_0","hbo_13_1","hbo_13_2","hbo_13_3","hbo_13_4","hbo_13_5","hbo_13_6","hbo_13_7","hbo_13_8","hbo_13_9","hbo_13_10","hbo_13_11","hbo_13_12","hbo_14_0","hbo_14_1","hbo_14_2","hbo_14_3","hbo_14_4","hbo_14_5","hbo_14_6","hbo_14_7","hbo_14_8","hbo_14_9","hbo_14_10","hbo_14_11","hbo_14_12","hbo_14_13","hbo_15_0","hbo_15_1","hbo_15_2","hbo_15_3","hbo_15_4","hbo_15_5","hbo_15_6","hbo_15_7","hbo_15_8","hbo_15_9","hbo_15_10","hbo_15_11","hbo_15_12","hbo_15_13","hbo_15_14","hbo_16_0","hbo_16_1","hbo_16_2","hbo_16_3","hbo_16_4","hbo_16_5","hbo_16_6","hbo_16_7","hbo_16_8","hbo_16_9","hbo_16_10","hbo_16_11","hbo_16_12","hbo_16_13","hbo_16_14","hbo_16_15","hbo_17_0","hbo_17_1","hbo_17_2","hbo_17_3","hbo_17_4","hbo_17_5","hbo_17_6","hbo_17_7","hbo_17_8","hbo_17_9","hbo_17_10","hbo_17_11","hbo_17_12","hbo_17_13","hbo_17_14","hbo_17_15","hbo_17_16","hbo_18_0","hbo_18_1","hbo_18_2","hbo_18_3","hbo_18_4","hbo_18_5","hbo_18_6","hbo_18_7","hbo_18_8","hbo_18_9","hbo_18_10","hbo_18_11","hbo_18_12","hbo_18_13","hbo_18_14","hbo_18_15","hbo_18_16","hbo_18_17","hbo_19_0","hbo_19_1","hbo_19_2","hbo_19_3","hbo_19_4","hbo_19_5","hbo_19_6","hbo_19_7","hbo_19_8","hbo_19_9","hbo_19_10","hbo_19_11","hbo_19_12","hbo_19_13","hbo_19_14","hbo_19_15","hbo_19_16","hbo_19_17","hbo_19_18","hbo_20_0","hbo_20_1","hbo_20_2","hbo_20_3","hbo_20_4","hbo_20_5","hbo_20_6","hbo_20_7","hbo_20_8","hbo_20_9","hbo_20_10","hbo_20_11","hbo_20_12","hbo_20_13","hbo_20_14","hbo_20_15","hbo_20_16","hbo_20_17","hbo_20_18","hbo_20_19","hbo_21_0","hbo_21_1","hbo_21_2","hbo_21_3","hbo_21_4","hbo_21_5","hbo_21_6","hbo_21_7","hbo_21_8","hbo_21_9","hbo_21_10","hbo_21_11","hbo_21_12","hbo_21_13","hbo_21_14","hbo_21_15","hbo_21_16","hbo_21_17","hbo_21_18","hbo_21_19","hbo_21_20","hbr_1_0","hbr_2_0","hbr_2_1","hbr_3_0","hbr_3_1","hbr_3_2","hbr_4_0","hbr_4_1","hbr_4_2","hbr_4_3","hbr_5_0","hbr_5_1","hbr_5_2","hbr_5_3","hbr_5_4","hbr_6_0","hbr_6_1","hbr_6_2","hbr_6_3","hbr_6_4","hbr_6_5","hbr_7_0","hbr_7_1","hbr_7_2","hbr_7_3","hbr_7_4","hbr_7_5","hbr_7_6","hbr_8_0","hbr_8_1","hbr_8_2","hbr_8_3","hbr_8_4","hbr_8_5","hbr_8_6","hbr_8_7","hbr_9_0","hbr_9_1","hbr_9_2","hbr_9_3","hbr_9_4","hbr_9_5","hbr_9_6","hbr_9_7","hbr_9_8","hbr_10_0","hbr_10_1","hbr_10_2","hbr_10_3","hbr_10_4","hbr_10_5","hbr_10_6","hbr_10_7","hbr_10_8","hbr_10_9","hbr_11_0","hbr_11_1","hbr_11_2","hbr_11_3","hbr_11_4","hbr_11_5","hbr_11_6","hbr_11_7","hbr_11_8","hbr_11_9","hbr_11_10","hbr_12_0","hbr_12_1","hbr_12_2","hbr_12_3","hbr_12_4","hbr_12_5","hbr_12_6","hbr_12_7","hbr_12_8","hbr_12_9","hbr_12_10","hbr_12_11","hbr_13_0","hbr_13_1","hbr_13_2","hbr_13_3","hbr_13_4","hbr_13_5","hbr_13_6","hbr_13_7","hbr_13_8","hbr_13_9","hbr_13_10","hbr_13_11","hbr_13_12","hbr_14_0","hbr_14_1","hbr_14_2","hbr_14_3","hbr_14_4","hbr_14_5","hbr_14_6","hbr_14_7","hbr_14_8","hbr_14_9","hbr_14_10","hbr_14_11","hbr_14_12","hbr_14_13","hbr_15_0","hbr_15_1","hbr_15_2","hbr_15_3","hbr_15_4","hbr_15_5","hbr_15_6","hbr_15_7","hbr_15_8","hbr_15_9","hbr_15_10","hbr_15_11","hbr_15_12","hbr_15_13","hbr_15_14","hbr_16_0","hbr_16_1","hbr_16_2","hbr_16_3","hbr_16_4","hbr_16_5","hbr_16_6","hbr_16_7","hbr_16_8","hbr_16_9","hbr_16_10","hbr_16_11","hbr_16_12","hbr_16_13","hbr_16_14","hbr_16_15","hbr_17_0","hbr_17_1","hbr_17_2","hbr_17_3","hbr_17_4","hbr_17_5","hbr_17_6","hbr_17_7","hbr_17_8","hbr_17_9","hbr_17_10","hbr_17_11","hbr_17_12","hbr_17_13","hbr_17_14","hbr_17_15","hbr_17_16","hbr_18_0","hbr_18_1","hbr_18_2","hbr_18_3","hbr_18_4","hbr_18_5","hbr_18_6","hbr_18_7","hbr_18_8","hbr_18_9","hbr_18_10","hbr_18_11","hbr_18_12","hbr_18_13","hbr_18_14","hbr_18_15","hbr_18_16","hbr_18_17","hbr_19_0","hbr_19_1","hbr_19_2","hbr_19_3","hbr_19_4","hbr_19_5","hbr_19_6","hbr_19_7","hbr_19_8","hbr_19_9","hbr_19_10","hbr_19_11","hbr_19_12","hbr_19_13","hbr_19_14","hbr_19_15","hbr_19_16","hbr_19_17","hbr_19_18","hbr_20_0","hbr_20_1","hbr_20_2","hbr_20_3","hbr_20_4","hbr_20_5","hbr_20_6","hbr_20_7","hbr_20_8","hbr_20_9","hbr_20_10","hbr_20_11","hbr_20_12","hbr_20_13","hbr_20_14","hbr_20_15","hbr_20_16","hbr_20_17","hbr_20_18","hbr_20_19","hbr_21_0","hbr_21_1","hbr_21_2","hbr_21_3","hbr_21_4","hbr_21_5","hbr_21_6","hbr_21_7","hbr_21_8","hbr_21_9","hbr_21_10","hbr_21_11","hbr_21_12","hbr_21_13","hbr_21_14","hbr_21_15","hbr_21_16","hbr_21_17","hbr_21_18","hbr_21_19","hbr_21_20","hbt_1_0","hbt_2_0","hbt_2_1","hbt_3_0","hbt_3_1","hbt_3_2","hbt_4_0","hbt_4_1","hbt_4_2","hbt_4_3","hbt_5_0","hbt_5_1","hbt_5_2","hbt_5_3","hbt_5_4","hbt_6_0","hbt_6_1","hbt_6_2","hbt_6_3","hbt_6_4","hbt_6_5","hbt_7_0","hbt_7_1","hbt_7_2","hbt_7_3","hbt_7_4","hbt_7_5","hbt_7_6","hbt_8_0","hbt_8_1","hbt_8_2","hbt_8_3","hbt_8_4","hbt_8_5","hbt_8_6","hbt_8_7","hbt_9_0","hbt_9_1","hbt_9_2","hbt_9_3","hbt_9_4","hbt_9_5","hbt_9_6","hbt_9_7","hbt_9_8","hbt_10_0","hbt_10_1","hbt_10_2","hbt_10_3","hbt_10_4","hbt_10_5","hbt_10_6","hbt_10_7","hbt_10_8","hbt_10_9","hbt_11_0","hbt_11_1","hbt_11_2","hbt_11_3","hbt_11_4","hbt_11_5","hbt_11_6","hbt_11_7","hbt_11_8","hbt_11_9","hbt_11_10","hbt_12_0","hbt_12_1","hbt_12_2","hbt_12_3","hbt_12_4","hbt_12_5","hbt_12_6","hbt_12_7","hbt_12_8","hbt_12_9","hbt_12_10","hbt_12_11","hbt_13_0","hbt_13_1","hbt_13_2","hbt_13_3","hbt_13_4","hbt_13_5","hbt_13_6","hbt_13_7","hbt_13_8","hbt_13_9","hbt_13_10","hbt_13_11","hbt_13_12","hbt_14_0","hbt_14_1","hbt_14_2","hbt_14_3","hbt_14_4","hbt_14_5","hbt_14_6","hbt_14_7","hbt_14_8","hbt_14_9","hbt_14_10","hbt_14_11","hbt_14_12","hbt_14_13","hbt_15_0","hbt_15_1","hbt_15_2","hbt_15_3","hbt_15_4","hbt_15_5","hbt_15_6","hbt_15_7","hbt_15_8","hbt_15_9","hbt_15_10","hbt_15_11","hbt_15_12","hbt_15_13","hbt_15_14","hbt_16_0","hbt_16_1","hbt_16_2","hbt_16_3","hbt_16_4","hbt_16_5","hbt_16_6","hbt_16_7","hbt_16_8","hbt_16_9","hbt_16_10","hbt_16_11","hbt_16_12","hbt_16_13","hbt_16_14","hbt_16_15","hbt_17_0","hbt_17_1","hbt_17_2","hbt_17_3","hbt_17_4","hbt_17_5","hbt_17_6","hbt_17_7","hbt_17_8","hbt_17_9","hbt_17_10","hbt_17_11","hbt_17_12","hbt_17_13","hbt_17_14","hbt_17_15","hbt_17_16","hbt_18_0","hbt_18_1","hbt_18_2","hbt_18_3","hbt_18_4","hbt_18_5","hbt_18_6","hbt_18_7","hbt_18_8","hbt_18_9","hbt_18_10","hbt_18_11","hbt_18_12","hbt_18_13","hbt_18_14","hbt_18_15","hbt_18_16","hbt_18_17","hbt_19_0","hbt_19_1","hbt_19_2","hbt_19_3","hbt_19_4","hbt_19_5","hbt_19_6","hbt_19_7","hbt_19_8","hbt_19_9","hbt_19_10","hbt_19_11","hbt_19_12","hbt_19_13","hbt_19_14","hbt_19_15","hbt_19_16","hbt_19_17","hbt_19_18","hbt_20_0","hbt_20_1","hbt_20_2","hbt_20_3","hbt_20_4","hbt_20_5","hbt_20_6","hbt_20_7","hbt_20_8","hbt_20_9","hbt_20_10","hbt_20_11","hbt_20_12","hbt_20_13","hbt_20_14","hbt_20_15","hbt_20_16","hbt_20_17","hbt_20_18","hbt_20_19","hbt_21_0","hbt_21_1","hbt_21_2","hbt_21_3","hbt_21_4","hbt_21_5","hbt_21_6","hbt_21_7","hbt_21_8","hbt_21_9","hbt_21_10","hbt_21_11","hbt_21_12","hbt_21_13","hbt_21_14","hbt_21_15","hbt_21_16","hbt_21_17","hbt_21_18","hbt_21_19","hbt_21_20"]
    
    for i in range(columns):
       for j in range(len(signal_features)):
           f = signal_features[j] + "_col_" + str(i)
           list_features.append(f)
        
    for i in range(len(connectivity_features)):
        list_features.append(connectivity_features[i])
       
    for i in range(len(pearson_features)):
        list_features.append(pearson_features[i])
        
    return list_features
    
    


def list_name_features_2(columns):
    list_features = []
    list_features_freq = []
    
    signal_features = ["mean","var","std","max_value","avg_time_between_zero_crossing","avg_time_to_peak"]
    freq_features = ["var_fft","energy_fft","mean_fft","skew_fft","kur_fft","dominant_freq"]
    connectivity_features = ["hbo_max_corr","hbr_max_corr","hbt_max_corr","index_hbo_1","index_hbo_2","index_hbr_1","index_hbr_2","index_hbt_1","index_hbt_2","most_active_band_hbo","most_active_band_hbr","most_active_band_hbt"]
    for i in range(columns):
       for j in range(len(signal_features)):
           f = signal_features[j] + "_col_" + str(i)
           list_features.append(f)
           f2 = freq_features[j] + "_col_" + str(i)
           list_features_freq.append(f2)
       
    return list_features,list_features_freq,connectivity_features
    

def energy_parseval(signal):
    x = len(signal)
    suma = 0
    for i in range(0,x):        
        suma = suma + (signal[i])**2
        
    return suma
