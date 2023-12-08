#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alpha_vantage.timeseries import TimeSeries

#%% Load the time series from alphavantage
#def download_data(config):
def download_data(name_stock,output_size,key_av, close_f):
    #ts = TimeSeries(key=config["alpha_vantage"]["key"])
    ts = TimeSeries(key = key_av)
        
    #data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    data, meta_data = ts.get_daily_adjusted(name_stock, outputsize=output_size)

    data_date = [date for date in data.keys()]
    data_date.reverse()

    #data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price = [float(data[date][close_f]) for date in data.keys()]
    
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    #print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

#%% My functions
def get_time_series(name_stock):
    
    key_av = "AH0KC5MDR9NVKPLD"
    outputsize = "full"
    close_f = "5. adjusted close"
    
    #config["alpha_vantage"]["symbol"] = name_stock    
    #data_date, data_close_price, num_data_points, display_date_range = download_data(config)
    data_date, data_close_price, num_data_points, display_date_range = download_data(name_stock,outputsize,key_av, close_f)

    return data_close_price,num_data_points

def plot_time_series(time_series,name_stock):
    fig = figure(figsize=(45, 20), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    
    plt.plot(np.arange(len(time_series)), time_series, color="#001f3f")
    #xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(time_series))
    plt.xticks(x, x, rotation='vertical')
    plt.title("Daily close price for " + name_stock + ", " )#+ display_date_range)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.show()

def save_timeseries_csv(data,name,path):    
    comp_name = path + "\\"+ name + ".csv"
    df = pd.DataFrame(data)
    df.columns = [name]
    df.to_csv(comp_name,index=False,header=True)
    
        
#%% Test    
"""
name_stock = "Tsla"
time_series,num_data_c = get_time_series(name_stock)
plot_time_series(time_series,name_stock)  
name = name_stock 
path = 'C:\\Users\\gabri\\Dropbox\\Financial series prediction\\3-EEMD_xgboost_causal_forest\\python_scripts'
save_timeseries_csv(time_series,name,path)
"""

print("Functions from time series loaded")