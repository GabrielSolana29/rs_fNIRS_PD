import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.fft import fft#, fftfreq
from scipy.signal import blackman
from scipy.signal import find_peaks
import functions_feature_extraction as ffe
import statsmodels.api as sm
import functions_LSTM as flstm
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import functions_paths as fp


def mae(signal,target_signal):
    score_m = mean_absolute_error(target_signal,signal)    
    return score_m


def r_square(signal,target_signal):
    score_r = r2_score(target_signal,signal)
    return score_r


def rmse(signal,r_signal):
    # Root mean squared error
    r_error = mean_squared_error(signal,r_signal, squared = False)
    return r_error


def correlation_between_signals(signal,target_signal,verbosity:bool=False):
    stt = np.std(target_signal)
    sts = np.std(signal)
    if stt == 0.0 or sts == 0.0:
        print("\n Standar deviation in the signal is 0")
        return np.array([[0,0],[0,0]])
    corr = np.corrcoef(signal,target_signal)
    if verbosity == True:        
        print("\nCorrelation: \n",corr)
    return corr


def autocorrelation_imfs(imf_mat,verbosity:bool=False):
    no_imfs = np.shape(imf_mat)[0]
    ac_vec = []
    ac_mean_vec = []
    for i in range(no_imfs):
        #calculate autocorrelation
        ac_imf = sm.tsa.acf(imf_mat[i,:],nlags=int(np.ceil(len(imf_mat[i,:])/10)),fft=False)
        ac_vec.append(ac_imf)
        ac_mean = np.mean(ac_imf)
        ac_mean_vec.append(ac_mean)
        #if verbosity == True:
        #    print("mean auctocorrelation with nlags=len(signal)/10, imf ",i," ",ac_mean)
        
    if verbosity == True:     
        print("\n")
        plt.bar(range(0,no_imfs),ac_mean_vec)
        #plt.grid()
        plt.title("mean auctocorrelation with nlags=len(signal)/10")
        plt.show()
        
    return ac_vec,ac_mean_vec


def concentration_energy_spectrum(imf_mat,no_imfs,verbosity:bool=True,title_plot:str="Energy concentration"):
    perc_max_vec = []
    for z in range(no_imfs-1):    
        signal = imf_mat[z,:]
        # Number of sample points
        N = len(signal)
        # sample spacing        
        w = blackman(N) # Windowing the signal with a dedicated window function helps mitigate spectral leakage.       
        signal_f = fft(signal*w) # Multiplie the original signal with the window and get the fourier transform      
        positive_real_fs = np.abs(signal_f[0:N//2]) # only positive real values        
        # Get the peaks from the signal 
        peaks_position,peak_heights = find_peaks(positive_real_fs)
        peaks_values = positive_real_fs[peaks_position]
        suma = 0
        maxi = np.max(peaks_values)
        for i in range(0,len(peaks_values)):
            if peaks_values[i] == maxi:
                suma = suma
            else:
                suma = suma + peaks_values[i]    
        one_hundred = suma+maxi
        perc_maximum = (maxi*100)/one_hundred
        perc_max_vec.append(perc_maximum)
        if verbosity == True:
            print("percentage of the maximumn:",perc_maximum)
      
    if verbosity == True:
        plt.bar(np.arange(0,len(perc_max_vec),1),perc_max_vec)
        plt.title(title_plot)
        #plt.grid()
        plt.xlabel('IMF')
        plt.ylabel('Percentage in dm freq')
        plt.show()        
        
    return perc_max_vec


def variance_imfs(imf_mat,verbosity:bool=False):
    variance_vec = []
    no_imfs = np.shape(imf_mat)[0]
    
    for i in range(no_imfs):
        variance_vec.append(np.var(imf_mat[i,:]))
        if verbosity == True:
            print("Variance imf ", i ,": ",variance_vec[i])
        
    
    if verbosity == True:
        print("\n")   
        plt.bar(range(0,no_imfs),variance_vec)
        #plt.grid()
        plt.title("Variance in IMFS")
        plt.show()
        
    return variance_vec


def complete_analysis(imf_mat,no_imfs,sampling_distances,return_values:bool=False,smooth:bool=False,verbosity:bool=True):
    # Show the number of previous points that will be used as features
    plt.plot()
    # Show the conentration of energy in the dominant frequency in each imf
    ac_vec,ac_mean_vec = autocorrelation_imfs(imf_mat,verbosity=verbosity)
    # Show the autocorrelation of each imf
    concentration_energy_vec = concentration_energy_spectrum(imf_mat,no_imfs,verbosity=verbosity)
    # Show the variance of each imf
    var_vec = variance_imfs(imf_mat,verbosity=verbosity)        
    
    for i in range(np.shape(imf_mat)[0]):
        plt.show()
        plt.plot(imf_mat[i,:])
        plt.title("IMF: " + str(i))
    
    
    if return_values==True:
        return ac_vec,ac_mean_vec,concentration_energy_vec,var_vec
    
    
    
    
    
    
    