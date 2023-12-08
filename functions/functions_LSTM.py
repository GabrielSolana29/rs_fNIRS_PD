#Recurrent neural networks lSTM
import numpy as np
import os
import pandas as pd
# PyTorch library
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 
from torch.utils.data import TensorDataset, DataLoader
import functions_feature_extraction as ffe
import functions_paths as fpt
import load_save_data as fld

def initialize_lstm():
    ## Recomended by pytorch official website to set the enviromental variable to avoid errors in cuda 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"    
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        
    else:
        device = torch.device("cpu")        
                                

def array_to_tensor(arr):
    tensor = Variable(torch.Tensor(arr))
    return tensor

#reshape tensor to have row,timestamp,feature
def reshape_tensor(tensor):
    if np.shape(tensor.shape)[-1] <=1:       
       new_tensor = torch.reshape(tensor,(tensor.shape[0],1,1))
       return new_tensor
    else:   
        new_tensor = torch.reshape(tensor,(tensor.shape[0],1,tensor.shape[1]))
        return new_tensor
    
def reshape_tensor_label(tensor):
      new_tensor = torch.reshape(tensor,(tensor.shape[0],1))
      return new_tensor
    

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True) #lstm
            self.fc_1 =  nn.Linear(hidden_size, 5) #fully connected 1 #used to be 50           
            self.fc = nn.Linear(5, num_classes) #fully connected last layer # used to be 50
            
        if num_layers == 2:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                   num_layers=num_layers, batch_first=True) #lstm
            self.fc_1 =  nn.Linear(hidden_size,5) #fully connected 1
            self.fc_2 =  nn.Linear(5, 5) #fully connected 2
            self.fc = nn.Linear(5, num_classes) #fully connected last layer 
            
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        if self.num_layers == 1:
            out = self.fc_1(out) #first Dense            
            out = self.relu(out) #relu
            out = self.fc(out) #Final Output
            
        if self.num_layers == 2:                        
            out = self.fc_1(out) #first Dense
            out = self.relu(out) #relu
            out = self.fc_2(out) #second Dense
            out = self.relu(out) #relu
            out = self.fc(out) #Final Output
            
        return out
        
  
def train_test_tensor(xtrain,ytrain):
    size = np.shape(xtrain)[0]
    size_test = int(np.floor(size/5))
    
    X_train_tensor = array_to_tensor(xtrain[:size_test])
    X_test_tensor = array_to_tensor(xtrain[size_test:])
    
    y_train_tensor = array_to_tensor(ytrain[:size_test]) 
    y_test_tensor = array_to_tensor(ytrain[size_test:]) 
    
    X_train_tensor = reshape_tensor(X_train_tensor)
    X_test_tensor = reshape_tensor(X_test_tensor)
    
    y_train_tensor = reshape_tensor_label(y_train_tensor)
    y_test_tensor = reshape_tensor_label(y_test_tensor)
    
    return X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor


def train_LSTM(x_train,x_test,y_train,y_test,n_layers:int=1,n_epochs:int=1000,lr:int=.1):    
    input_size = np.shape(x_train)[-1]
    hidden_size = input_size*2
    num_layers = n_layers
    num_classes = 1
      
    lstm = LSTM1(num_classes,input_size,hidden_size,num_layers,x_train.shape[-1])
        
    #cost_function = torch.nn.CrossEntropyLoss()
    #cost_function1 = torch.nn.L1Loss()
    #cost_function = torch.nn.MSELoss()
    #cost_function1 = torch.nn.MSELoss()
    cost_function = torch.nn.HuberLoss() 
    #cost_function = cost_function1 + cost_function2
    #cost_function = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(lstm.parameters(),lr=lr)
    
    for epoch in range(n_epochs):
        outputs = lstm.forward(x_train)
        optimizer.zero_grad()        
        #loss = cost_function1(outputs, y_train)+ cost_function2(outputs,y_train)
        loss = cost_function(outputs, y_train)
        loss.backward()        
        optimizer.step()
        if loss.item() <= .000005:
            return lstm
            
        if epoch % 100 ==0:
            print("Epoch: %d, loss: %1.5f" % (epoch,loss.item()))
            
    return lstm
            

def train_save_LSTM(xtrain_list,ytrain_list,name_csv,no_imf,no_points_feat,n_layers:int=1,n_epochs:int=1000,lr:float=.1,gpu:bool=False):        

    # Get the data to train the model in tensor form 
    xtrain,ytrain = ffe.get_xtrain_ytrain(xtrain_list,ytrain_list,no_imf)        
    x_train,x_test,y_train,y_test = train_test_tensor(xtrain,ytrain)  
    #xtrain = array_to_tensor(xtrain)
    #xtrain = reshape_tensor(xtrain)
    #ytrain = array_to_tensor(ytrain)
    #ytrain = reshape_tensor(ytrain)
    model = train_LSTM(x_train,x_test,y_train,y_test,n_layers=n_layers,n_epochs=n_epochs,lr=lr)
    #model = train_LSTM(xtrain,x_test,ytrain,y_test,n_layers=n_layers,n_epochs=n_epochs,lr=lr)
    print("size_xtrain", np.shape(xtrain))
    name_model = "lstm_imf"+ str(no_imf) +"_" + str(name_csv)
    save_model_lstm(model, name_model)    
        
    #return model

 
def load_all_lstm_models(no_imfs,name_csv,linux:bool=False,verbosity:bool=True):    
    path_name = fpt.path_saved_models_params(linux=linux) 
    os.chdir(path_name)    
    model_vec = []
           
    for i in range(no_imfs):         
        name_model = "lstm_imf"+ str(i) +"_" + str(name_csv)
        model_vec.append(load_model_lstm(name_model,linux=linux))
                
    return model_vec
    

def load_params_construct_lstm(name_model,linux:bool=False):      
    if linux==True:
        name_file = name_model + "_construct"        
    else:
        name_file = name_model + "_construct" 
    
    file = fld.load_file_from_txt(name_file,linux=linux)
    return file


def load_model_lstm(name_model,linux:bool=False):
    path_name = fpt.path_saved_models_params(linux=linux)   
    #os.chdir(path_name) 
    if linux==True:
        name_file = path_name + "/" + name_model + ".pt"        
    else:
        name_file = path_name + "\\" + name_model + ".pt"            
        
    params = load_params_construct_lstm(name_model,linux=linux)    
    model = LSTM1(params[0],params[1],params[2],params[3],params[4])
    model.load_state_dict(torch.load(name_file))
    model.eval()
    
    return model
    

def save_params_construct_lstm(model,name_model,linux:bool=False):
            
    if linux==True:
        name_model = name_model + "_construct"        
    else:
        name_model = name_model + "_construct" 
    
    lst = []
    lst.append(model.num_classes)
    lst.append(model.input_size)
    lst.append(model.hidden_size)
    lst.append(model.num_layers)
    lst.append(model.seq_length)
    lst = np.array(lst)
    
    fld.save_file_as_txt(lst,name_model,linux=linux)
    

def save_model_lstm(model,name_model,linux:bool=False,verbosity:bool=True):    
    path_name = fpt.path_saved_models_params(linux=linux)   
    #os.chdir(path_name) 
    if linux==True:
        name_file = path_name + "/" + name_model + ".pt"        
    else:
        name_file = path_name + "\\" + name_model + ".pt"            
        
    save_params_construct_lstm(model,name_model,linux=linux)        
    # Save
    torch.save(model.state_dict(), name_file)

    if verbosity == True:
        print("\nmodel saved")


def predict_lstm(model,xtrain):         
    xtrain_tensor = array_to_tensor(xtrain)    
    xtrain_tensor = reshape_tensor(xtrain_tensor)
        
    prediction = model(xtrain_tensor)
    prediction = prediction.data.numpy()
    
    return prediction 
    

def predict_function(model,xtrain):    
    prediction = predict_lstm(model,xtrain)
    return prediction

    

"""
#%% DATA
import matplotlib.pyplot as plt
inicio = 0
fin = 3000
inicio2 = int(fin+(fin/4))
xtrain = time_series[:fin]
#xtrain = np.vstack((xtrain,xtrain))
xtrain = np.transpose(xtrain)
ytrain = time_series[inicio + 1:fin + 1]
xtest = time_series[inicio+1:(inicio2+1)]
#xtest = np.vstack((xtest,xtest))
xtest = np.transpose(xtest)
ytest = time_series[inicio+2:inicio2+2]

X_train_tensor = array_to_tensor(xtrain)
X_test_tensor = array_to_tensor(xtest)

y_train_tensor = array_to_tensor(ytrain)
y_test_tensor = array_to_tensor(ytest)


X_train_tensor = reshape_tensor(X_train_tensor)
X_test_tensor = reshape_tensor(X_test_tensor)

y_train_tensor = reshape_tensor_label(y_train_tensor)
y_test_tensor = reshape_tensor_label(y_test_tensor)
#X_train_tensor.shape
#y_train_tensor.shape
    
#%% NN
num_epochs= 30000
learning_rate = .00005
input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1
  
lstm = LSTM1(num_classes,input_size,hidden_size,num_layers,X_train_tensor.shape[-1])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm.forward(X_train_tensor)
    optimizer.zero_grad()
    
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 ==0:
        print("Epoch: %d, loss: %1.5f" % (epoch,loss.item()))


#%% Predict 

xtrain = time_series[inicio2:-2]
xtrain = np.transpose(xtrain)
ytrain = time_series[inicio2+1:-1]

xtrain = array_to_tensor(xtrain)
xtrain = reshape_tensor(xtrain)

prediction = lstm(xtrain)
prediction = prediction.data.numpy()
prediction = prediction[:,0]

plt.plot(prediction)
plt.plot(ytrain)

fspa.correlation_between_signals(ytrain,prediction,verbosity=True)
"""