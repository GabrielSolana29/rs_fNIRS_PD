#Functions of Complete ensemble EMD with adaptive noise (CEEMDAN)
#%% import libraries
# $ pip install EMD-signal
import pandas as pd
import numpy as np
from PyEMD import EMD
from PyEMD import EEMD
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#%% Functions 

def initialize_CEEMDAN(trials_f:int=100,epsilon_f:float=0.005,parallel_f:bool=False,processes_f:int=1,noise_scale_f:float=1.0):
    ## We are adding a noise seed for reproducibility 
    # epsilon:float (default: 0.005) Scale for added noise (ϵ) which multiply std σ: β=ϵ⋅σ    
    ceemdan = CEEMDAN(trials=trials_f, epsilon=epsilon_f, parallel=parallel_f, processes=processes_f)
    ceemdan.noise_seed(0)    

    return ceemdan


#%% Example


"""
ceemdan_f = initialize_CEEMDAN(trials_f=200,processes_f=24)
# Execute CEEMDAN on signal
imf_mat = ceemdan_f.ceemdan(s,t)
imf_1 = imf_mat[1,:]




print("Functions from CEEMDAN loaded")
"""