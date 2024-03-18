import numpy as np

#%% Functions 
def get_patient(df,number,control:bool=False):
    
    if control == True:
        number = number + .1
    
    for i in range(np.shape(df)[0]):    
        key_ini = 0
        if df.id[i] == number:            
            key_ini = 1
             
        if key_ini == 1:
            cont = 0
            while(1):
                if i+cont < np.shape(df)[0]:
                    if df.id[i+cont] == number:
                        cont += 1
                    else:
                        break
                else:
                    break
                
            df = np.array(df)
            patient = df[i:i+cont,:]

            return patient[:,:-2]
            