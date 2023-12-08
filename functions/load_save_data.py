#pip install pyarrow
#pip install fastparquet
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import copy as cp
import pickle as pk
import functions_paths as fpt

def normalize_vec(vec):
    x = len(vec)
    new_vec = np.zeros(x)
    mi = min(vec)
    ma = max(vec)    
    
    for i in range(0,x):
        new_vec[i] = (vec[i]-mi)/(ma-mi)
        
    return new_vec


def load_csv(complete_name,path,linux:bool=False):
    if linux == False:
        complete_name = path + "\\" + complete_name + ".csv"    
    else:
        complete_name = path + "/" + complete_name + ".csv"
     
    df = pd.read_csv(
       complete_name, 
       na_values=['NA', '?'])
                                                                                                                        
    return df


def save_csv_file(name_file,path,data,linux:bool=False,verbosity:bool=True,header:bool=True):
    header_csv = []
    if header ==  True:
        for i in range(np.shape(data)[1]):
            h_s = "x" + str(i)
            header_csv.append(h_s)
        header_csv = np.array(header_csv)  
        data = np.vstack((header_csv,data))    
        
    data = pd.DataFrame(data)

    if linux == False:
        complete_name = path + "\\" + name_file + ".csv"
    else:
        complete_name = path + "/" + name_file + ".csv"
    # Save csv with pandas
    data.to_csv(complete_name, header=False, index=False) 
       
    if verbosity== True:
        print("\nFile: '",name_file,"' correctly saved")


def save_file_as_txt(file,name_file,linux:bool=False):
    
    path = fpt.path_saved_models_params(linux=linux)
    if linux==False:
        complete_name = path + "\\" + name_file + ".txt"
    else:
        complete_name = path + "/" + name_file + ".txt"
    
    
    with open(complete_name, "wb") as fp:   #Pickling
        pk.dump(file, fp)

    
def load_file_from_txt(name_file,linux:bool=False,verbosity:bool=False):
    path = fpt.path_saved_models_params(linux=linux)
    if linux==False:
        complete_name = path + "\\" + name_file + ".txt"
    else:
        complete_name = path + "/" + name_file + ".txt"
        
    with open(complete_name, "rb") as fp:   # Unpickling
        b = pk.load(fp)
    
    if verbosity==True:
        print("\nTxt file loaded")
    return b


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    