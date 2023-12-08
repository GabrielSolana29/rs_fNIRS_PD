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
#from scipy.signal import decimate
import functions_ceemdan as fcmdn
import functions_paths as fpt
import functions_signal_processing_analysis as fspa
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


def automatic_no_points_f(signal,threshold_perc:float=.71):
    #Lag: the number of previous observations measured during autocorrelation    
    no_points = 0
    autocorr = 1
    autocorr_vec = sm.tsa.acf(signal,nlags=len(signal),fft=False) 
    
    while np.abs(autocorr) >= threshold_perc:
        autocorr = autocorr_vec[no_points]                  
        no_points += 1
        
    return no_points-1


def ceemdan_feature_label_extraction(time_series,limit_no_points_feature_extraction:int=10,percentage:int=99.9999,show_plots:bool=True,automatic_no_points:bool=False,smooth:bool=False,features:bool=False):
    t = np.linspace(0, 1, len(time_series))
    #% Obtain the IMFs from the time series
    ceemdan_f = fcmdn.initialize_CEEMDAN(trials_f=100,parallel_f=False)
    
    # Execute CEEMDAN on signal
    imf_mat = ceemdan_f.ceemdan(time_series,t)
    print("\nFinished Ceemdan")    
    
    if automatic_no_points == True:
        no_points_feature_extraction = []
        threshold_percentage = .71
        for i in range(len(imf_mat)):
            signal = imf_mat[i,:]            
            no_points = automatic_no_points_f(signal,threshold_perc=threshold_percentage)
            if no_points > limit_no_points_feature_extraction:
                no_points = limit_no_points_feature_extraction
                
            no_points_feature_extraction.append(no_points)
            
        no_points_feature_extraction_vec = np.array(no_points_feature_extraction)
    else:
        no_points_feature_extraction_vec = np.ones(len(imf_mat)) * limit_no_points_feature_extraction
        
    
    #% Obtain the vectors with the sampling distances and dominant frequencies
    sampling_distances,dominant_frequencies = sampled_distances_dominant_frequencies_imfs(imf_mat,percentage,show_plots,smooth=smooth)
    # With f10 you can visualize the figures with great quality
    print("\nFinished sampling distances")
    #% Get the training labels and features  
  
    #max_no_points_feature_extraction = np.max(no_points_feature_extraction_vec)
    
    # STANDARD METHOD
    #no_points_feature_extraction= int(no_points_feature_extraction_vec[0])
    #xtrain_list = training_points_imfs(imf_mat,sampling_distances,no_points_feature_extraction = no_points_feature_extraction)                
    #ytrain_list = training_label_imfs(imf_mat,sampling_distances,no_points_feature_extraction = no_points_feature_extraction)
    
    # Automatic method
    ytrain_list = training_label_imfs_no_points(imf_mat,sampling_distances,no_points_feature_extraction_vec)
    xtrain_list = training_points_imfs_no_points(imf_mat,no_points_feature_extraction_vec)    
        
    ## Experimental methods
    #ytrain_list = training_label_imfs_sampling_distance(imf_mat,sampling_distances,no_points_feature_extraction = no_points_feature_extraction)
    #ytrain_list = training_label_imfs(imf_mat,sampling_distances,no_points_feature_extraction = max_no_points_feature_extraction)
    #xtrain_list = training_points_imfs_sampling_distance(imf_mat,sampling_distances,no_points_feature_extraction)        
    #xtrain_list = training_features_imfs(imf_mat,sampling_distances,no_points_feature_extraction = no_points_feature_extraction)            
    
    if features == True:
        features_mat = training_features_imfs(imf_mat,sampling_distances,no_points_feature_extraction=no_points_feature_extraction_vec[0])
        return xtrain_list,ytrain_list,sampling_distances,dominant_frequencies,imf_mat,no_points_feature_extraction_vec,features_mat

    
    if features == False:
        return xtrain_list,ytrain_list,sampling_distances,dominant_frequencies,imf_mat,no_points_feature_extraction_vec


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


def last_n_points(vec,n):    
    new_vec = np.zeros(n)    
    x = len(vec)-1
    cont = n - 1
    
    for i in range(0,n):
        new_vec[cont] = vec[x-i]
        cont = cont - 1
        
    return new_vec


def dominant_frequency(signal,sampling_rate:float=1.0,no_freq:int=1,show_plot:bool=False,windowing:bool=True):    
    #https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html#d-discrete-fourier-transforms
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


def energy_parseval(signal):
    x = len(signal)
    suma = 0
    for i in range(0,x):        
        suma = suma + (signal[i])**2
        
    return suma

        
def sampled_signal(signal,sampling_distance):    
    x = len(signal)
    s_signal = np.zeros(x)
    cont = 1
    for i in range(0,x):
        if cont < sampling_distance:
            s_signal[i] = 0
            cont = cont + 1
        else:
            s_signal[i] = signal[i]
            cont = 1
                       
    return s_signal


def peaks_fft_deprecated(signal,sampling_rate:float=1.0,percentage:float=99.99,show_plot:bool=False,windowing:bool=True):    
    # Number of sample points
    N = len(signal)
    # sample spacing
    T = 1.0 / sampling_rate  # (sampling rate)     
    if windowing==True:        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
    else:
        w = 1
        
    signal_f = fft(signal*w) # Multiplie the original signal with the window and get the fourier transform
    freqs = fftfreq(N, T)[:N//2] 
    positive_real_fs = np.abs(signal_f[0:N//2]) # only positive real values
    freq_step = freqs[1]
    
    # Get the peaks from the signal 
    peaks_position,peak_heights = find_peaks(positive_real_fs)
    peaks_values = positive_real_fs[peaks_position]
    position,percentage_achieved = energy_parseval_acumulated(peaks_values,percentage)
    
    highest_relevant_frequency = peaks_position[position] * freq_step
    # Plot signal with peaks
    n = 50
    if show_plot==True:
        plt.plot(positive_real_fs[0:peaks_position[n]])
        plt.plot(peaks_position[0:n], positive_real_fs[peaks_position[0:n]], "x")        
        plt.grid()
        plt.title(label= "Local Maxima of spectrum")
        plt.show()
        
    return highest_relevant_frequency,position,percentage_achieved


def energy_parseval_acumulated(signal,percentage):
    x = len(signal)
    total_energy = energy_parseval(signal)
    suma = 0
    for i in range(0,x):
        suma = suma + (signal[i])**2
        perc = (suma*100)/total_energy
        if perc >= percentage:
            return i, perc    
    

def energy_parseval_sorted_acumulated(signal,percentage):
    x = len(signal)
    total_energy = energy_parseval(signal)
    suma = 0
    positions_signal = np.argsort(signal)
    #reverse the array so the first position is the highest local maximum
    positions_signal = positions_signal[::-1]
    for i in range(0,x):
        suma = suma + (signal[positions_signal[i]])**2
        perc = (suma*100)/total_energy
        if perc >= percentage:
            if i > 0:
                max_freq = np.max(positions_signal[0:i])
            else:
                max_freq = positions_signal[0]
            return max_freq, perc 
    

def peaks_fft(signal,sampling_rate:float=1.0,percentage:float=99.99,show_plot:bool=False,windowing:bool=True):    
    # Number of sample points
    N = len(signal)
    # sample spacing
    T = 1.0 / sampling_rate  # (sampling rate)     
    if windowing==True:        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
    else:
        w = 1
        
    signal_f = fft(signal*w) # Multiplie the original signal with the window and get the fourier transform
    freqs = fftfreq(N, T)[:N//2] 
    positive_real_fs = np.abs(signal_f[0:N//2]) # only positive real values
    freq_step = freqs[1]
    
    # Get the peaks from the signal 
    peaks_position,peak_heights = find_peaks(positive_real_fs)
    peaks_values = positive_real_fs[peaks_position]
    position,percentage_achieved = energy_parseval_sorted_acumulated(peaks_values,percentage)            
    highest_relevant_frequency = peaks_position[position] * freq_step
    
    # Plot signal with peaks
    n = 50    
    if len(peaks_position) <= n:
        n = len(peaks_position)-1
        
    if show_plot==True and len(peaks_position) >= n:
        plt.plot(positive_real_fs[0:peaks_position[n]])
        plt.plot(peaks_position[0:n], positive_real_fs[peaks_position[0:n]], "x")        
        plt.grid()
        plt.title(label= "Local Maxima of spectrum")
        plt.show()
        
    return highest_relevant_frequency,percentage_achieved
    

def find_no_decimals(number,no_int,verbosity:bool=False):
    str_perc = str(number)
    no_dec = len(str_perc)
    # The point and the two whole numbers oucupy 3 char. Those need to be removed to find the number of decimals
    no_dec -= no_int + 1
    if verbosity ==True:
        print("\nNo. of decimals: ",no_dec)
    
    return no_dec

def add_decimal_last_position(original_no,decimal_to_add):        
    lenstr = len(str(original_no).split(".")[1])
    no_int = len(str(original_no)) - lenstr - 1 #+1 because of the dot
    
    no_dec = find_no_decimals(original_no,no_int,False)
    add_dec = str(0)+ "."
    
    if no_dec > 0:
        for i in range(no_dec):
            add_dec = add_dec + "0"
    else:
        print("Error no decimals")
        return 0
    add_dec = add_dec + str(decimal_to_add)
    original_no += float(add_dec)
    original_no = round(original_no,no_dec+1)
    return original_no

    
def sampling_distance_nyq(signal,percentage:float=99.99,show_plot:bool=False,windowing:bool=True,smooth:bool=False):                
    signal_norm = normalize_vec(signal)
    df = dominant_frequency(signal_norm)
    highest_relevant_fr,percentage_achieved = peaks_fft(signal_norm,percentage,show_plot,windowing)
        
    # Condition 1: Highest relevant frequency must be equal or higher than the dominant frequency of the signal
    if df[0] > highest_relevant_fr:        
        percentage_w = percentage
        while df[0] > highest_relevant_fr:
            show_plot_in_while= False
            percentage_w = add_decimal_last_position(percentage_w,9)
            if percentage_w <= 100.000:                
                highest_relevant_fr,percentage_achieved = peaks_fft(signal,percentage_w,show_plot_in_while,windowing)
            else:
                highest_relevant_fr = add_decimal_last_position(percentage,9)
    
    # Condition 2: The RMSE between the signal and the reconstructed signal must be less than .001 (of normalized vec of imf)
    # Else: The nyquist theorem is incremtented .1 each iteration 
    nyq_f = highest_relevant_fr * 2
    sampling_distance = int(np.ceil(1/(nyq_f))) 
    #print("\nSampling Distance ",sampling_distance)
    rmse_error = fspa.rmse_reconstructed_original(signal_norm, sampling_distance)
    #print("\nRMSE ",rmse_error)
    if rmse_error >= .01:
        cont = 2
        while rmse_error >= .01:
            cont = cont + .1           
            nyq_f = highest_relevant_fr * cont
            sampling_distance = int(np.ceil(1/(nyq_f)))        
            rmse_error = fspa.rmse_reconstructed_original(signal_norm, sampling_distance,smooth=smooth)
    #print("\nRMSE2 ",rmse_error)
       
    if show_plot == True :        
        sampled_s = sampled_signal(signal,sampling_distance)
        t = np.linspace(0, 1, len(sampled_s))
        plt.title(label = "Signal sampled every " + str(sampling_distance) + " points")
        plt.plot(t,sampled_s,t,signal)
        plt.show()
    
    return sampling_distance


def sampled_distances_dominant_frequencies_imfs(imf_mat,percentage:float=99.99,show_plot:bool=False,windowing:bool=True,smooth:bool=False):
    total_imfs = len(imf_mat)
    #dominant_frequencies_vec = np.zeros(total_imfs-1)
    #sampling_distances_vec = np.zeros(total_imfs-1)
    
    dominant_frequencies_vec = np.zeros(total_imfs)
    sampling_distances_vec = np.zeros(total_imfs)    
    
    #for i in range(0,total_imfs-1):
    for i in range(0,total_imfs-1):
        current_imf = imf_mat[i,:]     
        no_frequencies = 1
        dominant_frequencies_vec[i] = dominant_frequency(current_imf,no_frequencies,show_plot,windowing)        
        sampling_distances_vec[i] = sampling_distance_nyq(current_imf,percentage,show_plot,windowing,smooth=smooth)
        
    # The sampling distance of the last imf is the same as the last-1 imf      
    sampling_distances_vec[-1] = sampling_distances_vec[-2]
    
    if show_plot== True:
        signal = imf_mat[-1,:]
        sampled_s = sampled_signal(signal,sampling_distances_vec[-1])
        t = np.linspace(0, 1, len(sampled_s))      
        plt.title(label="Signal sampled every " + str(sampling_distances_vec[-1]) +" points")        
        plt.plot(t,sampled_s,t,signal)
        plt.show()
    
    return sampling_distances_vec,dominant_frequencies_vec
   

def get_training_label_nyq(signal,sampling_distance):    
    no_training_s = int(len(signal) - sampling_distance)
    ytrain = np.zeros(no_training_s)
    for i in range(no_training_s):
        position = int(i + sampling_distance)
        ytrain[i] = signal[position]
        
    return ytrain


def training_label_imfs(imf_mat,sampling_distances,no_points_feature_extraction:int=1):
    start = no_points_feature_extraction-1 # The number of points that are needed for feature extraction
    x = len(imf_mat)
    labels = []
    #for i in range(0,x-1):                    
    for i in range(0,x): 
        signal = imf_mat[i,start:]
        sampling_d = int(sampling_distances[i])        
        y_signal = get_training_label_nyq(signal,sampling_d)
        labels.append(y_signal)
    
    return labels    


def training_label_imfs_no_points(imf_mat,sampling_distances,no_points_feature_extraction_vec):    
    x = len(imf_mat)
    labels = []
    
    for i in range(0,x): 
        start = int(no_points_feature_extraction_vec[i]-1) # The number of points that are needed for feature extraction
        signal = imf_mat[i,start:]
        sampling_d = int(sampling_distances[i])        
        y_signal = get_training_label_nyq(signal,sampling_d)
        labels.append(y_signal)
    
    return labels    



def get_training_label_nyq_sampling_distance(signal,sampling_distance,no_points_feature_extraction):    
    no_training_s = int(len(signal)-sampling_distance)
    ytrain = np.zeros(no_training_s)
    for i in range(no_training_s):
        position = int(i + sampling_distance)
        ytrain[i] = signal[position]
        
    return ytrain


def training_label_imfs_sampling_distance(imf_mat,sampling_distances,no_points_feature_extraction:int=1):    
    x = len(imf_mat)
    labels = []
                        
    for i in range(x): 
        sampling_d = int(sampling_distances[i])        
        start = (no_points_feature_extraction*sampling_d)-sampling_d # The number of points that are needed for feature extraction
        signal = imf_mat[i,start:]                
        y_signal = get_training_label_nyq_sampling_distance(signal,sampling_d,no_points_feature_extraction)
        labels.append(y_signal)
    
    return labels    


### returns the consecutive points
def training_points_imfs(imf_mat,sampling_distances,no_points_feature_extraction:int=1):    
    x = len(imf_mat)
    points = []
               
    for i in range(x):                    
        signal = imf_mat[i,0:]
        #sampling_d = int(sampling_distances[i])
        x_signal = get_n_samples_for_feature_extraction(signal,no_points_feature_extraction)
        points.append(x_signal)
    
    return points


### Returns the points according to the sampling distance
def training_points_imfs_sampling_distance(imf_mat,sampling_distances,no_points_feature_extraction):    
    x = len(imf_mat)
    points = []                   
    for i in range(x):                        
        signal = imf_mat[i,0:]
        sampling_d = int(sampling_distances[i])
        x_signal = get_n_samples_for_feature_extraction_sampling_distance(signal,no_points_feature_extraction,sampling_d)
        points.append(x_signal)
    
    return points


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


def get_n_samples_for_feature_extraction_sampling_distance(signal,n,sampling_distance):
    x_list = []
    end_for = len(signal)-(n*sampling_distance)+sampling_distance #from the current point i can predict n days in the future
    #end_for = len(signal)-(n) #from the current point i can predict n days in the future
    for i in range(end_for):          
        start = i
        end = i + (n*sampling_distance)
        x_list.append(signal[range(start,end,sampling_distance)])
        
    return x_list


def training_features_imfs_deprecated(imf_mat,sampling_distances,no_points_feature_extraction:int=1):    
    x = len(imf_mat)
    features = []
    
    for i in range(x):                    
        signal = imf_mat[i,0:]
        sampling_d = int(sampling_distances[i])
        x_signal = get_n_samples_for_feature_extraction(signal,no_points_feature_extraction,sampling_d)
        x_vec = []
        x_signal = np.array(x_signal)        
        for j in range(np.shape(x_signal)[0]):        
            f = extract_features_from_points(x_signal[j,:])
            x_vec.append(f)            
        x_vec = np.array(x_vec)
        features.append(x_vec)    
    
    return  features

def training_features_imfs(imf_mat,sampling_distances,no_points_feature_extraction:int=1):    
    x = len(imf_mat)
    features = []
    
    for i in range(x):                    
        signal = imf_mat[i,:]
        sampling_d = int(sampling_distances[i])
        x_signal = get_n_samples_for_feature_extraction(signal,int(no_points_feature_extraction))
        x_vec = []
        x_signal = np.array(x_signal)        
        for j in range(np.shape(x_signal)[0]):        
            f = extract_features_from_points(x_signal[j,:])
            x_vec.append(f)            
        x_vec = np.array(x_vec)
        features.append(x_vec)    
    
    return  features
    


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


def get_xtrain_deprecated(xtrain_list,current_imf,no_points_feature_extraction):
    x1 = xtrain_list[current_imf]
    xtrain = np.zeros((len(x1),no_points_feature_extraction))
    for i in range(len(x1)):
        xtrain[i,:] = x1[i]
    
    return xtrain
    
    
def get_xtrain_ytrain_deprecated(xtrain_list,ytrain_list,current_imf,no_points_feature_extraction):
    xtrain = get_xtrain(xtrain_list,current_imf,no_points_feature_extraction)       
    ytrain = ytrain_list[current_imf]
    
    return xtrain,ytrain

def get_xtrain(xtrain_list,current_imf):
    x1 = xtrain_list[current_imf]
    xtrain = np.zeros((len(x1),len(x1[0])))
    for i in range(len(x1)):
        xtrain[i,:] = x1[i]
    
    return xtrain
    
    
def get_xtrain_ytrain(xtrain_list,ytrain_list,current_imf):
    xtrain = get_xtrain(xtrain_list,current_imf)       
    ytrain = ytrain_list[current_imf]
    
    return xtrain,ytrain


def xtrain_ytrain_1_sample(imf_mat,no_points_feat,features:bool=False):
    xtrain_list = []
    ytrain_list = []
    for i in range(len(imf_mat)):
        xtrain = get_n_samples_for_feature_extraction(imf_mat[i,:],no_points_feat)
        if features== True:
            for z in range(len(xtrain)):
                f_vec = extract_features_from_points(xtrain[z])
                xtrain[z] = np.concatenate((f_vec,xtrain[z]),axis=0)
        
        xtrain_list.append(xtrain[:-1])        
        ytrain = []
        for j in range(len(imf_mat[i,:-no_points_feat])): 
            ytrain.append(imf_mat[i,j+no_points_feat])
        ytrain = np.array(ytrain)
        ytrain_list.append(ytrain)    
        
    return xtrain_list,ytrain_list    

def xtrain_for_training(xtrain_list,sampling_distances):
    ### For training we need to remove the sampling_distance[i] last features of the xtrain_list
    for i in range(len(xtrain_list)):
        xtrain_list[i] = xtrain_list[i][:-int(sampling_distances[i])]
        
    return xtrain_list  

def xtrain_concat_features_list(xtrain_list,features_list):
    new_list = []
    for i in range(len(xtrain_list)):
        x_mat = np.array(xtrain_list[i])
        f_mat = np.array(features_list[i])
        no_cols = len(x_mat[i,:]) + len(f_mat[i,:])
        new_mat = np.zeros((np.shape(x_mat)[0],no_cols))
        for j in range(np.shape(f_mat)[0]):                         
            new_mat[j:] = np.concatenate((f_mat[j,:],x_mat[j,:]),axis=0)
            
        new_list.append(new_mat)
    return new_list

def simple_moving_average(signal,n):       
    sma = np.zeros(len(signal))    
    for i in range(len(signal)):
        if i >= n:
            sma[i] = (sum(signal[i-n:i])-n) / n
    
    return sma

def exponential_moving_average(signal,n,smoothing:int=2):               
    k = smoothing / (n+1)
    ema = np.zeros(len(signal))
    for i in range(len(signal)):    
        if i == 0:                
            ema[i] = signal[i]
        else:
            ema[i] = (signal[i]*k )+ (ema[i-1]*(1-k))
    
    return ema

def average_gain_loss(signal,n):
    if len(signal)<n:
        print("Signal too small for AGL")
        return 0
    
    avg = [] # average gain
    avl = [] # average loss
    gain = 0
    loss = 0
    for k in range(n,len(signal)):        
        if k == n:            
            for i in range(1,n):                
                change = signal[i] - signal[i-1]                
                if change > 0:
                    gain += abs(change)
                if change < 0:
                    loss += abs(change)
            avg.append(gain / n)
            avl.append(loss / n)            
        else:
            change = signal[k] - signal[k-1]            
            if change > 0:
                avg.append(((avg[-1] * (n-1)) + abs(change))/n)
                avl.append(((avl[-1] * (n-1)) + 0)/n)                                
            elif change < 0:
                avg.append(((avg[-1] * (n-1)) + 0)/n)
                avl.append(((avl[-1] * (n-1)) + abs(change))/n)                                                                                             
            elif change == 0:
                avg.append(((avg[-1] * (n-1)) + 0)/n)
                avl.append(((avl[-1] * (n-1)) + 0)/n)                                                             
                
    avg = np.array(avg)
    avl = np.array(avl)
    return avg,avl    

def relative_strenght(signal,n):
    # We receive the avg and avl of the day n+1 to the last day included in the signal
    avg,avl = average_gain_loss(signal,n)
    rs = np.zeros(len(avg))
    
    for i in range(len(avg)):
        if avg[i] == 0:
            avg[i] = .0001
        if avl[i] == 0:
            avl[i] = .0001
            
        rs[i] = avg[i]/avl[i]
        
    return rs

def relative_strenght_index(signal,n):
    rs = relative_strenght(signal, n)
    rsi = np.zeros(len(rs))
    
    for i in range(len(rs)):
        rsi[i] = 100 - (100 / (1 + rs[i]))
    return rsi


def intersection_signals(s1,s2):
    list_intersections = []
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            list_intersections.append([i,s1[i]])
            
    return list_intersections
    

def cross_ema(signal,ns,smoothing:int=2):
    # emas stands for the n of the different emas
    emas = []
    for i in range(len(ns)):        
        ema = exponential_moving_average(signal, ns[i],smoothing=smoothing)
        emas.append(ema)        
             
    points = intersection_signals(emas[1],emas[0])
                
    return points

#%% Functions specific of fnirs feature extraction 

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
    
    
#%% Load the CSV
"""
import pandas as pd
path_name = "C:\\Users\\gabri\\Dropbox\\Financial series prediction\\3-EEMD_xgboost_causal_forest\\python_scripts\\CSV"
#path_name = "/home/gscode29/Dropbox/Financial series prediction/3-EEMD_xgboost_causal_forest/python_scripts/CSV" #linux
name_csv = "IBM"

complete_name = path_name + "\\" + name_csv + ".csv"
#complete_name = path_name + "/" + name_csv + ".csv" #linux

df = pd.read_csv(
    complete_name, 
    na_values=['NA', '?'])

time_series = df.to_numpy()
time_series = time_series[0:len(time_series),0]

#%% Test 

signal = time_series
show_plot = True
windowing = True
no_freq = 5
ans = dominant_frequency(signal,no_freq,show_plot,windowing)

percentage = 99.99
highest_relevant_frequency,percentage_achieved = peaks_fft(signal,percentage,show_plot)

"""

