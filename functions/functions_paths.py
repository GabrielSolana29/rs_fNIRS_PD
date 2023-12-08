


def path_functions(linux:bool=False):
    import os
    directory = os.getcwd()
    
    if linux == True:          
        directory_functions = str(directory +"/functions/")
        path = directory_functions
    else:
        path = str(directory +"\\functions\\")      
        #path = "C:\\Users\\gabri\\Box\\Doctorado Sistemas Inteligentes Udlap\\Publicaciones\\fMRI_PD\\Classification time series\\python_scripts\\functions"
    
    return path

def path_CSV(linux:bool=False):
    import os
    directory = os.getcwd()        
    
    if linux == True:     
        directory_functions = str(directory +"/CSV/")
        path = directory_functions        
    else:
        path = str(directory +"\\CSV\\")      
        #path = "C:\\Users\\gabri\\Box\\Doctorado Sistemas Inteligentes Udlap\\Publicaciones\\fMRI_PD\\Classification time series\\python_scripts\\CSV"
    
    return path


def path_saved_models_params(linux:bool=False):
    import os
    directory = os.getcwd()
    if linux == True:        
        directory_functions = str(directory +"/saved_model_params/")
        path = directory_functions        
    else:
        path = str(directory +"\\saved_model_params\\")      
        #path = "C:\\Users\\gabri\\Box\\Doctorado Sistemas Inteligentes Udlap\\Publicaciones\\fMRI_PD\\Classification time series\\python_scripts\\saved_models_params" # windows    
    return path
    

def path_figures(linux:bool=False):
    import os
    directory = os.getcwd()
    if linux == True:        
        directory_functions = str(directory +"/Figures/")
        path = directory_functions        
    else:
        path = str(directory +"\\Figures\\")         
        #path = "C:\\Users\\gabri\\Box\\Doctorado Sistemas Inteligentes Udlap\\Publicaciones\\fMRI_PD\\Classification time series\\python_scripts\\Figures\\" # windows    
    return path
    