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
        

def reconstruct_sampled_signal(signal,sampling_distance,spline:bool=False,smooth:bool=False,verbosity:bool=False):
    rec_signal = np.zeros(len(signal))    
    cont = 0
    len_sig = len(signal)-1
    #print("\nsampling_distance: ",sampling_distance)
    #Polyorder must be less than window_length
    if 3 >= int(sampling_distance):
        smooth = False                
        #print("\nChange smooth to false",smooth)        
    
    if spline == False:
        while cont <= len_sig:
            start_l = signal[cont]
            stop_l = signal[int(cont+sampling_distance)]
            linear_union = np.linspace(start_l, stop_l, int(sampling_distance), endpoint=True)        
            rec_signal[cont:(cont+int(sampling_distance))] = linear_union
            cont = cont + int(sampling_distance)    
            
            if cont > (len_sig-sampling_distance) and cont <= len_sig:                 
                start_l = signal[cont]
                stop_l = signal[-1]
                siz_e = int(len(signal)-cont)            
                linear_union = np.linspace(start_l, stop_l, siz_e, endpoint=True)        
                rec_signal[cont:(cont + siz_e)] = linear_union
               
                if smooth== True:                      
                    if sampling_distance%2 == 0:
                        size_r = int(sampling_distance+1)                          
                        
                    else:
                        size_r = int(sampling_distance)
                                            
                    win = size_r
                    poly = 3
                    rec_signal = savgol_filter(rec_signal,win,poly)
                        
                
                if verbosity == True:
                    plt.plot(rec_signal)
                    #plt.grid()
                    plt.title("Reconstructed signal")
                    plt.show()    
                        
                return rec_signal
            
    if spline == True:
        while cont <= len_sig:
            xr = np.arange(3)
            start_1 = signal[cont]
            half_sampling_dist = int(cont+(sampling_distance/2))
            start_l = signal[half_sampling_dist]
            stop_l = signal[int(cont+sampling_distance)]
            yr = []
            yr.append(start_1)
            yr.append(start_l)
            yr.append(stop_l)
            yr = np.array(yr)
            cs =  CubicSpline(xr,yr)
            step = 1/3
            xs = np.arange(0,int(sampling_distance),step)
            spline_union = cs(xs)           
            rec_signal[cont:(cont+int(sampling_distance))] = spline_union[0:int(sampling_distance)]
            cont = cont + int(sampling_distance)
            
            if cont > (len_sig-sampling_distance) and cont <= len_sig:                 
                    xr = np.arange(3)
                    start_1 = signal[cont]
                    siz_e = int(len(signal)-cont)            
                    half_sampling_dist = int(np.floor(siz_e/2))
                    if half_sampling_dist == 0:
                        half_sampling_dist = 1                                            
                    start_l = signal[half_sampling_dist]
                    stop_l = signal[-1]
                    yr = []
                    yr.append(start_1)
                    yr.append(start_l)
                    yr.append(stop_l)
                    yr = np.array(yr)
                    cs =  CubicSpline(xr,yr)
                    step = 1/3
                    xs = np.arange(0,siz_e,step)
                    spline_union = cs(xs)
                    
                    rec_signal[cont:(cont + siz_e)] = spline_union[0:siz_e]
                    if verbosity == True:
                        plt.plot(rec_signal)
                        #plt.grid()
                        plt.title("Reconstructed signal")
                        plt.show()    
                            
                    return rec_signal
                
    if smooth== True:            
        if sampling_distance%2 == 0:
            size_r = int(sampling_distance+1)                
        else:
            size_r = int(sampling_distance)        
        win = size_r
        poly = 3
        rec_signal = savgol_filter(rec_signal,win,poly)
        
    if verbosity == True:
            plt.plot(rec_signal)
            #plt.grid()
            plt.title("Reconstructed signal")
            plt.show()                      

                
    return rec_signal


def rmse_reconstructed_original(signal,sampling_distance,smooth:bool=False,verbosity:bool=False):
    r_signal = reconstruct_sampled_signal(signal,sampling_distance,smooth=smooth,verbosity=verbosity)    
    error_r = rmse(signal,r_signal)
    if verbosity == True:
        print("\n Error between signal and reconstruction: " ,error_r)    
    
    return error_r


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
        plt.grid()
        plt.title("mean auctocorrelation with nlags=len(signal)/10")
        plt.show()
        
    return ac_vec,ac_mean_vec


def variance_imfs(imf_mat,verbosity:bool=False):
    variance_vec = []
    no_imfs = np.shape(imf_mat)[0]
    
    for i in range(no_imfs):
        variance_vec.append(np.var(imf_mat[i,:]))
        if verbosity == True:
            print("\nVariance imf ", i ,": ",variance_vec[i])
        
    if verbosity == True:
        print("\n")
        plt.bar(range(0,no_imfs),variance_vec)
        plt.grid()
        plt.title("Variance in IMFS")
        plt.show()
        
    return variance_vec


def error_imf_sampling_distance(imf_mat,sampling_distances,smooth:bool=False,verbosity:bool=True):
    vec_error = []
    for i in range(len(imf_mat)):    
        signal=imf_mat[i,:]
        signal = ffe.normalize_vec(signal)
        sampling_distance = sampling_distances[i]    
        rmse_error = rmse_reconstructed_original(signal, sampling_distance,smooth=smooth,verbosity=verbosity)
        vec_error.append(rmse_error)
    if verbosity == True:
        plt.bar(range(len(vec_error)),vec_error)
        plt.title("Error between signal and reconstruction")
        plt.xlabel("IMF")
        plt.ylabel("RMSE")
        plt.show()
                
    return vec_error


def predict_imf(xtrain,model,sampling_distance,no_points_feat,points_f,algorithm:str='linear_regression',show_plot:bool=False,smooth:bool=False,features:bool=False,no_features:int=11):
    pred_vec = np.zeros(points_f)    
    
    # Variables for the iterations
    pos = 0
    end = 0
    iterations = int(np.ceil(points_f/sampling_distance))
    cont = 0
    cont2 = True
    for j in range(iterations):        
                
        # Predict next sample
        last_sample = xtrain[-1:,:]                        
        last_sample = np.nan_to_num(last_sample,nan=0, posinf=0, neginf=0)                                      
        
        if algorithm=='LSTM':
            #print("last_sample: ", np.shape(last_sample))
            #print("flstm: ", np.shape(flstm.predict_function(model,last_sample)))
            prediction = flstm.predict_function(model,last_sample)[0] 
            
        if algorithm=='svm':            
            prediction = model.predict(last_sample)
        if algorithm=='linear_regression':
            prediction = model.predict(last_sample)
        if algorithm=='xgboost':
            prediction = model.predict(last_sample)
        
        if np.isnan(prediction):
            prediction = [0]
        
        # If the xtrain vector contains featuresand samples from the time series, we need to separate them to conduct linear interpolation between them 
        if features==True:
            last_sample = last_sample[0,no_features:]
            l_last_sample = last_sample[-1]
        else:                
            l_last_sample = last_sample[0][-1]
        step = (prediction - l_last_sample)/sampling_distance
        l_last_sample += step
                
        # Create linear points between the last point and the prediction
        if sampling_distance > 1:    
            linear_union = np.linspace(l_last_sample, prediction, int(sampling_distance), endpoint=True)[:,-1]
            
            ### Create the new feature
            l_lu = len(linear_union)
            if l_lu >= no_points_feat:
                new_feature = linear_union[(l_lu-no_points_feat):l_lu]
            else:        
                if features == True:
                    last_sample = xtrain[-1:,no_features:][-1]
                else:
                    last_sample = xtrain[-1:,:][-1]                            
                new_feature = np.hstack((last_sample[l_lu:],linear_union))        
                
            if features == True:
                ### Concatenate the new_feature with the xtrain vector and the extracted features
                feat_vec = ffe.extract_features_from_points(new_feature)    
                #print(np.shape(feat_vec),np.shape(new_feature))
                new_feature = np.concatenate((feat_vec[:,],new_feature[:,]),axis=0)                    
                xtrain = np.vstack((xtrain,new_feature))                            
            else:
                ### Concatenate the new_feature with the xtrain vector     
                xtrain = np.vstack((xtrain,new_feature))
                
        else:
            linear_union = prediction
            
            ### Create the new feature
            if features == True:
                last_feature = xtrain[-1:,no_features:][0]
            else:
                last_feature = xtrain[-1:,:][0]
            
            new_feature = np.hstack((last_feature,prediction))        
            new_feature = new_feature[1:]
            
            if features == True:
                feat_vec = ffe.extract_features_from_points(new_feature)                                        
                new_feature = np.concatenate((feat_vec[:,],new_feature[:,]),axis=0)    
                xtrain = np.vstack((xtrain,new_feature))
            else:            
                xtrain = np.vstack((xtrain,new_feature))
                
            
        # Fill the prediction matrix with the new predictions
        if j == iterations-1:
            end = int(points_f - pos)
            pred_vec[pos:] = linear_union[0:end]
        else:        
            end = pos + int(sampling_distance)
            pred_vec[pos:end] = linear_union
            pos = end      
            
            
        if smooth==True and int(len(pred_vec))>3: 
            win = 0            
            if 3< sampling_distance:# and cont2 == True:
                if sampling_distance%2 == 0:
                    size_w = int(sampling_distance+1)
                else:
                    size_w = int(sampling_distance)
                                                                  
                win = size_w          
                poly = 3        
                #tst = xtrain[-1,-int(win):]
                #print("TEST: ", np.size(tst), "  win: ",win)
                
                if no_points_feat > win:
                    #print("Igot in: ", np.size(tst), "  win: ",win, " ")
                    xtrain[-1,-int(win):] = savgol_filter(xtrain[-1,-int(win):],win,poly)                
               
                    
    #print("\nsampling_distance: ", sampling_distance," pred_Vec: ",int(len(pred_vec)))
    if smooth==True:
        win = 0
        if 3<sampling_distance:
            if sampling_distance%2 == 0:
                size_w = int(sampling_distance+1)   
            else:
                size_w = int(sampling_distance)                          
            
            if sampling_distance >= len(pred_vec):
                if int(len(pred_vec))%2 == 0:
                    size_w = int(len(pred_vec) - 1)
                else:
                    size_w = int(len(pred_vec) - 2)
            
            win = size_w                                                                                                    
            #tst = xtrain[-1,-int(win):]                        
            poly = 3
            if no_points_feat > win:
                #print("THE SECOND : ", np.size(tst), "  win: ",win, " ")
                pred_vec = savgol_filter(pred_vec,win,poly)
                       
    if show_plot==True: 
        plt.show()
        plt.title("Predictions")        
        #plt.grid()
        plt.plot(pred_vec)
        
    pred_vec = np.nan_to_num(pred_vec,nan=0, posinf=0, neginf=0)  
    return pred_vec


def complete_analysis(imf_mat,no_imfs,sampling_distances,return_values:bool=False,smooth:bool=False,verbosity:bool=True):
    # Show the number of previous points that will be used as features
    plt.plot()
    # Show the conentration of energy in the dominant frequency in each imf
    ac_vec,ac_mean_vec = autocorrelation_imfs(imf_mat,verbosity=verbosity)
    # Show the autocorrelation of each imf
    concentration_energy_vec = concentration_energy_spectrum(imf_mat,no_imfs,verbosity=verbosity)
    # Show the variance of each imf
    var_vec = variance_imfs(imf_mat,verbosity=verbosity)
    # Show the RMSE between signal and reconstruction
    vec_error = error_imf_sampling_distance(imf_mat,sampling_distances,smooth=smooth,verbosity=verbosity)
    if return_values==True:
        return ac_vec,ac_mean_vec,concentration_energy_vec,var_vec,vec_error
    


def plot_future(points_f,reconstructed_signal,time_series,init,end):
    
    plot_figure = plt.figure()
    plt.title("Real vs prediction " + str(points_f) +  " points in the future",figure=plot_figure)
    #plt.grid(figure=plot_figure)
    plt.plot(reconstructed_signal,label='Prediction',figure=plot_figure)
    plt.xlabel("points",figure=plot_figure)
    plt.ylabel("magnitude",figure=plot_figure)
    plt.plot(time_series[init:end],label='Real',figure=plot_figure)
    plt.legend(loc='upper right')
    plt.grid(figure=plot_figure)
        
    return plot_figure     


def save_plot_future(figure,name_signal,algorithm,scenario,points_f,no_points_feat,smooth,verbosity:bool=False,linux:bool=False):
    path_fig = fp.path_figures(linux=linux)
    name = path_fig + name_signal +"_"+ str(algorithm) + "_scenario_" + str(scenario) + "_" + str(points_f) + "_points_f_" + str(no_points_feat) + "_no_points_feat_" + "smooth=" + str(smooth)
    figure.savefig(name)
    if verbosity==True:
        print("\nFigure correctly saved")
        
        

def plot_fig(x_values,y_values,title:str='',grid:bool=False,x_label:str="",y_label:str=""):
    # create a new figure
    plot_fig = plt.figure()                
    plt.title(title,figure=plot_fig)
    plt.plot(x_values,y_values,figure=plot_fig)
    if x_label != "":
        plt.xlabel(x_label)
    if y_label != "":
        plt.ylabel(y_label)
        
    if grid==True:
        plt.grid(re=plot_fig)
    # return it
    return plot_fig



