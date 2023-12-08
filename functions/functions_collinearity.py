#https://towardsdatascience.com/targeting-multicollinearity-with-python-3bd3b4088d0b

## In this script functions to identify and remove the features that have high collinearity are provided
## this is based on the code from the link above
import numpyt as np
import pandas as pd


def correlation_between_features(data_frame,drop_features:bool=False,threshold:float=0.95):
    
    corr_matrix = data_frame.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    
    if drop_features == True:
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Drop features 
        data_frame.drop(to_drop, axis=1, inplace=True)
        
        return data_frame 

    else:
        return upper