modtype="GRU" #GRU, LSTM,BILSTM
seq_length=48
initm=10
test_size=0.2
GWL_type= "inertial" #annual, inertial, mixed
L=8
wavelet='la'+str(L)
import numpy as np
import matplotlib.pyplot as plt

import math
import geopandas as gpd
import pandas as pd
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
from tensorflow import device
from tensorflow.random import set_seed
from numpy.random import seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
import optuna
from optuna.samplers import TPESampler
from uncertainties import unumpy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
from typing import Tuple, List
import pickle
from wmtsa.modwt import modwt,cir_shift
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*1.5)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
  
ds = xr.open_dataset('/media/chidesiv/DATA2/Phase2/3_stations/ERA5_inputdata/1940_2022/ERA5_monthly_1940_2023_new.nc')

def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse



def calculate_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    data_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / data_range
    return nrmse


def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse
# function for wavelet transform (MODWT) which results in a dataframe containing both wavelet and scaling coefficients
def Wavelet_transform(X_train_s:np.ndarray)->pd.DataFrame:
    """
    Performs a wavelet decomposition using the wmtsa_cima library
    """
    mywav={}
    mysca={}
    mywavcir={}
    myscacir={}
    
    for i in range(X_train_s.shape[1]):
        mywav[i],mysca[i]=modwt(X_train_s[:,i],wtf=wavelet,nlevels=4,boundary='periodic',RetainVJ=False)
        mywavcir[i],myscacir[i]=cir_shift(mywav[i],mysca[i])
    wavlist=[np.transpose(mywavcir[i]) for i in range(X_train_s.shape[1])]  
    scalist=[myscacir[i] for i in range(X_train_s.shape[1])]
    li= wavlist+scalist
       
    X_train_w=pd.DataFrame(data=np.column_stack(li))
    return X_train_w


def Scaling_data(X_train,X_valid,y_train):
    """
    Scaling function to fit on training data and transform on test data

    Parameters
    ----------
    X_train : TYPE
        Original training data of input variables.
    X_valid : TYPE
        Original testing data.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train_s : TYPE
        scaled training input variables
    X_valid_s : TYPE
        scaled test input variables
    y_train_s : TYPE
        scaled training target variable.
    X_C_scaled : TYPE
        combined scaled training and test input variables

    """
    Scaler=MinMaxScaler(feature_range=(0,1))
    X_train_s=Scaler.fit_transform(X_train)
    X_valid_s=Scaler.transform(X_valid)
    target_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler.fit(np.array(y_train).reshape(-1,1))
    y_train_s=target_scaler.transform(np.array(y_train).reshape(-1,1))
    
    X_C_scaled=np.concatenate((X_train_s,X_valid_s),axis=0)
   
    return X_train_s,X_valid_s,y_train_s,X_C_scaled
 
def Scaling_extend(X_train,X_extend):
    """
    Scaling function to fit on training data and transform on test data

    Parameters
    ----------
    X_train : TYPE
        Original training data of input variables.
    X_valid : TYPE
        Original testing data.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train_s : TYPE
        scaled training input variables
    X_valid_s : TYPE
        scaled test input variables
    y_train_s : TYPE
        scaled training target variable.
    X_C_scaled : TYPE
        combined scaled training and test input variables

    """
    Scaler=MinMaxScaler(feature_range=(0,1))
    X_train_s=Scaler.fit_transform(X_train)
    X_extend_s=Scaler.transform(X_extend)
    
    
    
   
    return X_extend_s
 

def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

def reshape_onlyinputdata(x: np.ndarray,  seq_length: int) -> Tuple[np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        

    return x_new


#Hyperparameter tuning with Optuna for LSTM


def func_dl(trial):
   with device('/gpu:0'):    
    #     tf.config.experimental.set_memory_growth('/gpu:0', True)    
        set_seed(2)
        seed(1)
        
        
        
        optimizer_candidates={
            "adam":Adam(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "SGD":SGD(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "RMSprop":RMSprop(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True))
        }
        optimizer_name=trial.suggest_categorical("optimizer",list(optimizer_candidates))
        optimizer1=optimizer_candidates[optimizer_name]

    
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50,restore_best_weights = True),TFKerasPruningCallback(trial, monitor='val_loss')]
        
        epochs=trial.suggest_int('epochs', 50, 500,step=50)
        batch_size=trial.suggest_int('batch_size', 16,256,step=16)
        #weight=trial.suggest_float("weight", 1, 5)
        n_layers = trial.suggest_int('n_layers', 1, 1)
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 10, 100,step=10)
            return_sequences = True
            if i == n_layers-1:
                return_sequences = False
            # Activation function for the hidden layer
            if modtype == "GRU":
                model.add(GRU(num_hidden,input_shape=(X_train_l_all.shape[1],X_train_l_all.shape[2]),return_sequences=return_sequences))
            elif modtype == "LSTM":
                model.add(LSTM(num_hidden,input_shape=(X_train_l_all.shape[1],X_train_l_all.shape[2]),return_sequences=return_sequences))
            elif modtype == "BILSTM":
                model.add(Bidirectional(LSTM(num_hidden,input_shape=(X_train_l_all.shape[1],X_train_l_all.shape[2]),return_sequences=return_sequences)))
            model.add(Dropout(trial.suggest_float("dropout_l{}".format(i), 0.2, 0.2), name = "dropout_l{}".format(i)))
            #model.add(Dense(units = 1, name="dense_2", kernel_initializer=trial.suggest_categorical("kernel_initializer",['uniform', 'lecun_uniform']),  activation = 'Relu'))
        #model.add(BatchNormalization())study_blstm_
        model.add(Dense(1))
        model.compile(optimizer=optimizer1,loss="mse",metrics=['mse'])
        ##model.summary()


        model.fit(X_train_l_all,y_train_l_all,validation_data = (X_valid_ls,y_valid_ls ),shuffle = False,batch_size =batch_size,epochs=epochs,callbacks=callbacks
                  ,verbose = False)
  
        score=model.evaluate(X_valid_ls,y_valid_ls)

       

        return score[1]



def optimizer_1(learning_rate,optimizer):
        tf.random.set_seed(init+11111)
        if optimizer==Adam:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer==SGD:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        return opt


def gwlmodel(init, params,X_train_l, y_train_l, modtype, GWL_type):
        with tf.device('/gpu:0'):
            seed(init+99999)
            tf.random.set_seed(init+11111)
            par= params.best_params
            # seq_length = par.get(par_names[0])
            learning_rate=par.get(par_names[0])
            optimizer = par.get(par_names[1])
            epochs = par.get(par_names[2])
            batch_size = par.get(par_names[3])
            n_layers = par.get(par_names[4])
            # X_train, X_valid, y_train, y_valid=Scaling_data( X_tr'+ str(file_c['code_bss'][k])], X_va'+ str(file_c['code_bss'][k])], y_t'+ str(file_c['code_bss'][k])], y_v'+ str(file_c['code_bss'][k])])
            # X_train_l, y_train_l = reshape_data(X_train, y_train,seq_length=seq_length)
            model = Sequential()
            # i = 1
            for i in range(n_layers):
                return_sequences = True
                if i == n_layers-1:
                    return_sequences = False
                if modtype == "GRU":
                    model.add(GRU(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
                elif modtype == "LSTM":
                    model.add(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
                elif modtype == "BILSTM":
                    model.add(Bidirectional(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences)))
                model.add(Dropout(par["dropout_l{}".format(i)]))
            model.add(Dense(1))
            opt = optimizer_1(learning_rate,optimizer)
            model.compile(optimizer = opt, loss="mse",metrics = ['mse'])
            callbacks = [EarlyStopping(monitor = 'val_loss', mode ='min', verbose = 1, patience=50,restore_best_weights = True), tf.keras.callbacks.ModelCheckpoint(filepath='best_model'+str(modtype)+str(wavelet)+str(GWL_type)+str(init)+'.h5', monitor='val_loss', save_best_only = True, mode = 'min')]
            model.fit(X_train_l, y_train_l,validation_split = 0.2, batch_size=batch_size, epochs=epochs,callbacks=callbacks)
        return model
    


# read the shapefile



shape = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

# extract the coordinates
coords = shape[shape['Class'] == GWL_type].geometry.apply(lambda x: x.coords[:]).apply(pd.Series)


coords['code_bss'] = shape['code_bss']
coords['Class'] = shape['Class']
# rename the columns
coords.columns = ['x','code_bss','Class']


# convert the DataFrame to two columns
coords[['long','lat']] = pd.DataFrame(coords.x.tolist(), index=coords.index)

# drop the original column
coords.drop(columns=['x'], inplace=True)




from pyproj import Transformer
transformer = Transformer.from_crs( "EPSG:2154","EPSG:4326")
x, y = coords['lat'], coords['long']
#coords['latdeg'], coords['longdeg']= transformer.transform(x, y)

# Create an empty dictionary to store grid values for each code_bss
grid_values_dict = {}

# Iterate over each code_bss in the dataframe
for code_bss in coords['code_bss'].unique():
    # Filter the dataframe to get rows with the current code_bss
    df_code_bss = coords[coords['code_bss'] == code_bss]
    # Extract the grid values for each latdeg and longdeg
    grid_values = ds.sel(latitude=df_code_bss['lat'].values, longitude=df_code_bss['long'].values, method='nearest')
    # Add the grid values to the dictionary
    grid_values_dict[code_bss] = grid_values

# You can now access the grid values for each code_bss in the grid_values_dict
# print(grid_values_dict)

result = {}
for code in grid_values_dict.keys():
    df = pd.DataFrame()
    for var in grid_values_dict[code].data_vars:
        # print(var)
        # df_var = pd.DataFrame(grid_values_dict[code][var].values)
        df_var = grid_values_dict[code][var].to_dataframe()
        df_var.columns = [var]
        df = pd.concat([df, df_var], axis=1)
    df["code_bss"] = code
    result[code] = df

df_result1 = pd.concat(result.values())
# df_result = df_result.set_index(['time','expver'])
# df_result = df_result.query('expver == 1')
# df_result= df_result.drop(columns=['sst','mror','mssror','msror','d2m','ro','ssro'])
df_result= df_result1.drop(columns=['si10', 'ssr'])
df_result = df_result.dropna()



file_GWL =pd.read_csv(r"/media/chidesiv/DATA2/National_scale_models/newdata/new_data_all.csv")


file_GWL = file_GWL.rename(columns={'code': 'code_bss', 'hydraulic_head': 'val_calc_ngf','date':'date_mesure'})
# file_GWL =pd.read_csv(r"/media/chidesiv/DATA2/PhD/data/BRGM_DATA/Pour_Siva/Pour_Siva/GWL_time_series/chroniques_pretraitees.txt", sep='$')
file_GWL['code_bss'].unique()
file_GWL['date_mesure'] = pd.to_datetime(file_GWL['date_mesure'])

df_list = []
for code in grid_values_dict.keys():
    df = file_GWL[file_GWL['code_bss'] == code]
    df = df.assign(date=df['date_mesure']).drop(columns=['date_mesure'])
    df_list.append(df)

gwl_combine = pd.concat(df_list)


#Combined available GWL data with input data

gwl_combine['date'] = pd.to_datetime(gwl_combine['date'])
gwl_combine = gwl_combine.set_index('date')
gwl_combine = gwl_combine.groupby('code_bss').resample('MS').mean()

gwl_combine = gwl_combine.interpolate(method='linear')
gwl_combine = gwl_combine.reset_index()

df_result = df_result.rename(columns={'code_bss': 'code_bss'})
df_result['date'] = pd.to_datetime(df_result.index.get_level_values(0))
df_result = df_result.set_index('date')
# df_result = df_result.groupby('code_bss').resample('MS').mean(numeric_only=True)
df_result = df_result.reset_index()

final_df = pd.merge(df_result, gwl_combine, on=['code_bss','date'], how='inner')



# Extract the period of input data available for reconstruction based on first date of GWL available 


# Create dictionary to store extended dataframes
X_train_l_all = np.empty((0, seq_length, 2*5))
y_train_l_all = np.empty((0,  1))

X_valid_ls= np.empty((0, seq_length, 2*5))
y_valid_ls= np.empty((0,  1))

# Iterate over each code_bss
for code in final_df['code_bss'].unique():
    # Extract data for the current code_bss from final_df
    #Extract Class value for each code_bss from coords dataframe
    Class = coords[coords['code_bss'] == code]['Class'].values[0]

    df = final_df[final_df['code_bss'] == code]
    df_x1=df_result[df_result['code_bss'] == code]
    df_x=df_x1.drop(columns=['code_bss'])
    # Split into X and y
    X = df.drop(columns=['val_calc_ngf','code_bss'])

    y = df[['val_calc_ngf']]
    minimum_date= pd.to_datetime('1970-01-01')


    min_train_date = max(minimum_date, X['date'].min())
    lev=4 # number of levels for decomposition, here we choose 4


    f=((int(pow(2, lev))-1)*(L-1))+1 # number of boundary affected coefficients to be removed as per quilty and adamowski (2018)
    
    # Add 48 months to the minimum date
    first_date = min_train_date  - pd.DateOffset(months=48+f)
    # Filter X and y for train and test sets
    start_date = '2015-01-01'
    end_date = '2023-08-01'
    df_x.reset_index(drop=True, inplace=True)
    # X.reset_index(drop=True, inplace=True)
    # y.reset_index(drop=True, inplace=True)
    
    X_train = df_x[(df_x['date'] > first_date) & (df_x['date'] <= df['date'].iloc[-1])]
    y_train = y[(X['date'] >= min_train_date) & (X['date'] <= df['date'].iloc[-1])]
    X_test = X[(X['date'] >= start_date) & (X['date'] <= end_date)]
    y_test = y[(X['date'] >= start_date) & (X['date'] <= end_date)]
    
    # Extract rows from df_result with dates before the first date in X_train
    # Calculate the minimum date in the 'date' column
    #min_date = X['date'].min()
   
    #extended_df = df_result[(df_result['code_bss'] == code) & (df_result['date'] < min_date)]
    
    # Store the extended dataframe in the dictionary
    #extended_dfs[code] = extended_df
    
    # Create X_extend and y_extend by filtering the extended_df
    #X_extend = extended_df[extended_df['date'] < min_date].drop(columns=['code_bss'])
    # print(f'code_bss: {code}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
    X_train_s, X_valid_s, y_train_s,X_C_scaled=Scaling_data(X_train.drop(columns=['date']), X_test.drop(columns=['date']), y_train)
    
    #X_extend_s=Scaling_extend(X_train.drop(columns=['date']), X_extend.drop(columns=['date']))
    
    X_train_w1=Wavelet_transform(X_train_s)
    X_C_w1=Wavelet_transform(X_C_scaled)
    
    #Removing boundary affected coefficients in the beginning

   
    
    
    X_c= reshape_onlyinputdata(np.array(X_C_w1),seq_length=seq_length)
    
    X_train_l1=reshape_onlyinputdata(np.array(X_train_w1),seq_length=seq_length)
    
    X_train_l=X_train_l1[f:]
    
    y_train_l=y_train_s
    
    X_train_, X_valid_, y_train_, y_valid_  = train_test_split(X_train_l, y_train_l , test_size=0.2,random_state=1,shuffle=False)
        
    
    
    X_valid_ls = np.concatenate([X_valid_ls, X_valid_], axis=0)
    y_valid_ls = np.concatenate([y_valid_ls, y_valid_], axis=0)
    
    
    # Concatenate X_train_l and y_train_l with the previous code_bss
    X_train_l_all = np.concatenate([X_train_l_all, X_train_l], axis=0)
    y_train_l_all = np.concatenate([y_train_l_all, y_train_l], axis=0)
    
    X_C_w=X_c[f:]
    
    
    # X_ex_W1=Wavelet_transform(X_extend_s)
    
    X_valid_l=X_C_w[int((len(X_train_l))):]

    # # X_train_l,y_train_l= reshape_data(X_train_s,y_train_s,seq_length=seq_length)
    # # X_c= reshape_onlyinputdata(X_C_scaled,seq_length=seq_length)
    # X_extend_l1= reshape_onlyinputdata(np.array(X_ex_W1),seq_length=seq_length)
    # X_extend_l=X_extend_l1[f:]
    
    # X_valid_l=X_C_w[int((len(X_train_l))):]
    

    #X_train_ls, X_valid_ls, y_train_ls, y_valid_ls  = train_test_split(X_train_l_all, y_train_l_all , test_size=0.2,random_state=1,shuffle=False)
        
        
        
# Study_DL= optuna.create_study(direction='minimize',sampler=TPESampler(seed=10),study_name='study_'+str(modtype)+str(wavelet)+str(GWL_type))
# Study_DL.optimize(func_dl,n_trials=30) 
# #         # file_name = f"{code.replace('/','_')}.pkl"
# #         # with open(file_name, 'wb') as file:
# #         #     pickle.dump(Study_DL, file)
# pickle.dump(Study_DL,open('./'+'study_'+str(modtype)+str(GWL_type)+str(wavelet)+'.pkl', 'wb'))
    
# par = Study_DL.best_params
# par_names = list(par.keys())
   
DLmodels = {} 
for init in range(initm):
        # DLmodels[init] = gwlmodel(init, Study_DL,X_train_l_all, y_train_l_all, modtype, GWL_type)
          DLmodels[init] =tf.keras.models.load_model('best_model'+str(modtype)+str(wavelet)+str(GWL_type)+str(init)+'.h5')
# for code in final_df['code_bss'].unique():
target_scalers = {}
# scores = pd.DataFrame(columns=['code_bss','Class',
#                         'NSE_tr',
#                         'KGE_tr',
#                         'MAE_tr',
#                         'RMSE_tr',
#                         'NRMSE_tr',
#                         'NSE_te',
#                         'KGE_te',
#                         'MAE_te',
#                         'RMSE_te',
#                         'NRMSE_te'])

# for code in final_df['code_bss'].unique():
#     # Extract data for the current code_bss from final_df
#     #Extract Class value for each code_bss from coords dataframe
#     Class = coords[coords['code_bss'] == code]['Class'].values[0]

#     df = final_df[final_df['code_bss'] == code]
#     df_x1=df_result[df_result['code_bss'] == code]
#     df_x=df_x1.drop(columns=['code_bss'])
#     # Split into X and y
#     X = df.drop(columns=['val_calc_ngf','code_bss'])

#     y = df[['val_calc_ngf']]
#     minimum_date= pd.to_datetime('1970-01-01')


#     min_train_date = max(minimum_date, X['date'].min())
#     lev=4 # number of levels for decomposition, here we choose 4


#     f=((int(pow(2, lev))-1)*(L-1))+1 # number of boundary affected coefficients to be removed as per quilty and adamowski (2018)
    
#     # Add 48 months to the minimum date
#     first_date = min_train_date  - pd.DateOffset(months=48+f)
#     # Filter X and y for train and test sets
#     start_date = '2015-01-01'
#     end_date = '2023-08-01'
#     df_x.reset_index(drop=True, inplace=True)
#     # X.reset_index(drop=True, inplace=True)
#     # y.reset_index(drop=True, inplace=True)
    
#     X_train = df_x[(df_x['date'] > first_date) & (df_x['date'] <= df['date'].iloc[-1])]
#     y_train = y[(X['date'] >= min_train_date) & (X['date'] <= df['date'].iloc[-1])]
#     X_test = X[(X['date'] >= start_date) & (X['date'] <= end_date)]
#     y_test = y[(X['date'] >= start_date) & (X['date'] <= end_date)]
    
#     # Extract rows from df_result with dates before the first date in X_train
#     # Calculate the minimum date in the 'date' column
#     #min_date = X['date'].min()
   
#     #extended_df = df_result[(df_result['code_bss'] == code) & (df_result['date'] < min_date)]
    
#     # Store the extended dataframe in the dictionary
#     #extended_dfs[code] = extended_df
    
#     # Create X_extend and y_extend by filtering the extended_df
#     #X_extend = extended_df[extended_df['date'] < min_date].drop(columns=['code_bss'])
#     # print(f'code_bss: {code}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
#     X_train_s, X_valid_s, y_train_s,X_C_scaled=Scaling_data(X_train.drop(columns=['date']), X_test.drop(columns=['date']), y_train)
    
#     #X_extend_s=Scaling_extend(X_train.drop(columns=['date']), X_extend.drop(columns=['date']))
    
#     X_train_w1=Wavelet_transform(X_train_s)
#     X_C_w1=Wavelet_transform(X_C_scaled)
    
#     #Removing boundary affected coefficients in the beginning

   
    
    
#     X_c= reshape_onlyinputdata(np.array(X_C_w1),seq_length=seq_length)
    
#     X_train_l1=reshape_onlyinputdata(np.array(X_train_w1),seq_length=seq_length)
    
#     X_train_l=X_train_l1[f:]
    
#     y_train_l=y_train_s
    
#     target_scalers[code] = MinMaxScaler(feature_range=(0,1))
#     target_scalers[code].fit(np.array(y_train).reshape(-1,1))

#     sim_init = np.zeros((len(X_valid_l), initm))
#     sim_init[:] = np.nan
#     sim_tr = np.zeros((len(X_train_l), initm))
#     sim_tr[:] = np.nan
#     # sim_ex = np.zeros((len(X_extend_l), initm))
#     # sim_ex[:] = np.nan
#     # scores_tr = pd.DataFrame(columns=['R2_tr','RMSE_tr','MAE_tr'])
#     # scores_te = pd.DataFrame(columns=['R2_te','RMSE_te','MAE_te'])
    
#     # for code in final_df['code_bss'].unique():    
#     for init in range(initm):
#         # model = gwlmodel(init, Study_DL,X_train_l_all, y_train_l_all, modtype, GWL_type)
#         # model=tf.keras.models.load_model('best_model'+str(modtype)+str(init)+'{}.h5'.format(code.replace('/','_')))
#         # y_pred_valid = models[init].predict(X_valid_l)
#         # sim_test = target_scalers[code].inverse_transform(y_pred_valid)
#         y_pred_train = DLmodels[init].predict(X_train_l)
#         sim_train = target_scalers[code].inverse_transform(y_pred_train)
        
#         # sim_init[:, init] = sim_test[:, 0]
#         sim_tr[:, init] = sim_train[:, 0]
#         # sim_ex[:, init] = sim_extend[:, 0]
        
    
#     # sim_f = pd.DataFrame(sim_init)
#     sim_t = pd.DataFrame(sim_tr)
#     # sim_e = pd.DataFrame(sim_ex)
#     lower_percentile = 2.5  # 2.5th percentile
#     upper_percentile = 97.5  # 97.5th percentile
    
#     # Calculate the lower and upper bounds of the prediction interval
#     lower_bound_train = np.percentile(sim_tr, lower_percentile, axis=1)
#     upper_bound_train = np.percentile(sim_tr, upper_percentile, axis=1)
    
#     # lower_bound_test = np.percentile(sim_init, lower_percentile, axis=1)
#     # upper_bound_test = np.percentile(sim_init, upper_percentile, axis=1)
    
#     # lower_bound_extend = np.percentile(sim_ex, lower_percentile, axis=1)
#     # upper_bound_extend = np.percentile(sim_ex, upper_percentile, axis=1)
    
#     # sim_median = sim_f.median(axis=1)
#     sim_tr_median = sim_t.median(axis=1)
#     # sim_ex_median=sim_e.median(axis=1)
    
#     # sim_init_uncertainty = unumpy.uarray(sim_f.mean(axis=1), 1.96 * sim_f.std(axis=1))
#     # sim_tr_uncertainty = unumpy.uarray(sim_t.mean(axis=1), 1.96 * sim_t.std(axis=1))
#     # sim_ex_uncertainty = unumpy.uarray(sim_ex.mean(axis=1), 1.96 * sim_ex.std(axis=1))
#     # Calculate the desired percentiles for the prediction interval
    

#     # sim = np.asarray(sim_median).reshape(-1, 1)
#     sim_train = np.asarray(sim_tr_median).reshape(-1, 1)
#     # sim_extend=np.asarray(sim_ex_median).reshape(-1, 1)
    
#     obs_tr = np.asarray(target_scalers[code].inverse_transform(y_train_l).reshape(-1, 1))
    



#     from matplotlib import pyplot
#     pyplot.figure(figsize=(20,6))
#     Train_index=X_train.date
#     Test_index=X_test.date
#     # Extend_index=X_extend.date
#     from matplotlib import rc
#     # rc('font',**{'family':'serif'})
#     # rc('text', usetex = True)

  
#     pyplot.plot(df['date'], df['val_calc_ngf'],linestyle=':', marker='*', label=code)     
#     # pyplot.plot( Train_index,y_train, 'k', label ="observed train", linewidth=1.5,alpha=0.9)

#     pyplot.fill_between(Train_index[seq_length+f-1:],lower_bound_train,
#                     upper_bound_train, facecolor = (1.0, 0.8549, 0.72549),
#                     label ='95% confidence training',linewidth = 1,
#                     edgecolor = (1.0, 0.62745, 0.47843))    
#     pyplot.plot( Train_index[seq_length+f-1:],sim_train, 'r', label ="simulated  median train", linewidth = 0.9)
    
    
#     pyplot.title((str(modtype)+' '+str(wavelet)+ "ERA5 cluster NO ohe no static"+str(code)), size=24,fontweight = 'bold')
#     pyplot.xticks(fontsize=14)
#     pyplot.yticks(fontsize=14)
#     pyplot.ylabel('GWL [m asl]', size=24)
#     pyplot.xlabel('Date',size=24)
#     pyplot.legend()
#     pyplot.tight_layout()


def load_data_for_code(base_directory, models, scenarios, code, test_period_start, test_period_end, projection_period_start, projection_period_end):
    test_dataframes = {}
    projection_dataframes = {}
    
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize the scaler

    for model in models:
        for scenario in scenarios:
            file_path = os.path.join(base_directory, str(model), str(scenario), "{}.csv".format(str(code).replace('/', '_')))
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index, errors='coerce')  # Ensure index is DatetimeIndex
                
                test_df = df.loc[test_period_start:test_period_end]
                projection_df = df.loc[projection_period_start:projection_period_end]
                
                columns_to_drop = ['lat', 'long', 'code']
                test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], errors='ignore')
                projection_df = projection_df.drop(columns=[col for col in columns_to_drop if col in projection_df.columns], errors='ignore')
                
                if not test_df.empty:
                    # Resample and calculate the mean for monthly data
                    test_df_monthly = test_df.resample('MS').mean()
                    # Fit and transform the scaler on the test data
                    scaler.fit(test_df_monthly)
                    test_df_scaled = pd.DataFrame(scaler.transform(test_df_monthly), columns=test_df_monthly.columns, index=test_df_monthly.index)
                    test_dataframes[(model, scenario)] = test_df_scaled.reset_index()
                
                if not projection_df.empty:
                    # Resample and calculate the mean for monthly data
                    projection_df_monthly = projection_df.resample('MS').mean()
                    # Transform the projection data using the scaler fitted on the test data
                    projection_df_scaled = pd.DataFrame(scaler.transform(projection_df_monthly), columns=projection_df_monthly.columns, index=projection_df_monthly.index)
                    projection_dataframes[(model, scenario)] = projection_df_scaled.reset_index()
                    
                # Continue from your current function

                # Initialize dictionaries to store the NumPy arrays
                test_arrays = {}
                projection_arrays = {}
            
                for key, df in test_dataframes.items():
                    # Convert the DataFrame to a NumPy array and store
                    test_arrays[key] = df.drop(columns=['Date']).to_numpy()
            
                for key, df in projection_dataframes.items():
                    # Convert the DataFrame to a NumPy array and store
                    projection_arrays[key] = df.drop(columns=['Date']).to_numpy()
            
                return test_arrays, projection_arrays

    
    # return test_dataframes, projection_dataframes


# codes = coords['code_bss'].unique()  # Assuming 'coords' contains 'code_bss'


    # Further processing here, like saving these DataFrames to new files or performing analysis

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd

def load_and_scale_data_for_single_code_model_scenario(base_directory, model, scenario, code, test_period_start, test_period_end, projection_period_start, projection_period_end):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize the scaler
    
    file_path = os.path.join(base_directory, str(model), str(scenario), "{}.csv".format(code.replace('/','_')))
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index, errors='coerce')  # Ensure index is DatetimeIndex

        # Prepare test and projection datasets
        test_df = df.loc[test_period_start:test_period_end].drop(columns=['lat', 'long', 'code'], errors='ignore')
        projection_df = df.loc[projection_period_start:projection_period_end].drop(columns=['lat', 'long', 'code'], errors='ignore')
        
        test_df_dates = None
        projection_df_dates = None
        
        if not test_df.empty:
            test_df_monthly = test_df.resample('MS').mean()
            test_df_dates = test_df_monthly.index  # Save the dates for test dataset
            
            scaler.fit(test_df_monthly)  # Fit scaler on the test data
            test_df_scaled = scaler.transform(test_df_monthly)
            
        if not projection_df.empty:
            projection_df_monthly = projection_df.resample('MS').mean()
            projection_df_dates = projection_df_monthly.index  # Save the dates for projection dataset
            projection_df_scaled = scaler.transform(projection_df_monthly)
            
        # Return both scaled data and their corresponding dates
        return (test_df_scaled, test_df_dates), (projection_df_scaled, projection_df_dates)
    else:
        return None, None  # Return None if file does not exist


import os
import xarray as xr
import pandas as pd

# Parameters
base_directory = "/media/chidesiv/DATA2/Climate_projections/Combined/"
selected_models = ['ACCESS-ESM1-5', 'CanESM5', 'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'GISS-E2-1-G', 'FGOALS-g3', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL', 'TaiESM1']
variables = ['pr', 'tas']
scenarios = ['ssp245', 'ssp370', 'ssp585']

lev=4 # number of levels for decomposition, here we choose 4


f=((int(pow(2, lev))-1)*(L-1))+1 # number of boundary affected coefficients to be removed as per quilty and adamowski (2018)
 


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os



# Modify the plotting part as follows:
for code in final_df['code_bss'].unique():
    print(f"Code: {code}")
    Class = coords[coords['code_bss'] == code]['Class'].values[0]
    
    fig, axs = plt.subplots(1, len(scenarios), figsize=(20, 6), sharey=True)
    
    if len(scenarios) == 1:  # If there's only one scenario, ensure axs is iterable
        axs = [axs]
    
    for scenario_idx, scenario in enumerate(scenarios):
        ax = axs[scenario_idx]
        
        for model in selected_models:
            print(os.path.join(base_directory, str(model), str(scenario), "{}.csv".format(code.replace('/','_'))))
            X_test, X_projections = load_and_scale_data_for_single_code_model_scenario(
                base_directory=base_directory, 
                model=model, 
                scenario=scenario, 
                code=code, 
                test_period_start='2015-01-01', 
                test_period_end='2015-12-31', 
                projection_period_start='2015-01-01', 
                projection_period_end='2100-12-30'
            )

            # Assuming Wavelet_transform and reshape_onlyinputdata functions are defined elsewhere
            X_proj_wt = Wavelet_transform(X_projections)
            X_proj_l1 = reshape_onlyinputdata(np.array(X_proj_wt), seq_length=seq_length)
            X_proj_l = X_proj_l1[f:]

            sim_pr_median = np.zeros(len(X_proj_l))  # Placeholder for median projections
            
            for init in range(initm):  # Assuming 'initm' is the number of models
                y_project = DLmodels[init].predict(X_proj_l)
                sim_proj = target_scalers[code].inverse_transform(y_project)
                sim_pr_median += sim_proj[:, 0]
            
            sim_pr_median /= initm  # Calculate the median projection
            
            ax.plot(sim_pr_median, label=f"Model {model}", linewidth=0.9)  # Plot median projection for each model
        
        ax.set_title(f"{scenario} {Class} for {code}", size=12, fontweight='bold')
        ax.set_ylabel('GWL [m asl]' if scenario_idx == 0 else "")
        ax.set_xlabel('Time Step', size=12)
        ax.legend()

    plt.suptitle(f"CMIP6 Projections for {code}", size=24, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join("./", f"CMIP6_Projections_{Class}_{code.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()



for code in final_df['code_bss'].unique():
    print(f"Code: {code}")
    Class = coords[coords['code_bss'] == code]['Class'].values[0]
    df = final_df[final_df['code_bss'] == code]
    X = df.drop(columns=['val_calc_ngf','code_bss'])

    y = df[['val_calc_ngf']]
    minimum_date = pd.to_datetime('1970-01-01')
    min_train_date = max(minimum_date, X['date'].min())
    y_train = y[(X['date'] >= min_train_date) & (X['date'] <= '2014-12-01')]
    
    target_scalers[code] = MinMaxScaler(feature_range=(0,1))
    target_scalers[code].fit(np.array(y_train).reshape(-1,1))
    
    for model in selected_models:
        for scenario in scenarios:
            print(os.path.join(base_directory, str(model), str(scenario), "{}.csv".format(code.replace('/','_'))))
            (X_test_scaled, X_test_dates), (X_projections_scaled, X_projections_dates) = load_and_scale_data_for_single_code_model_scenario(
                base_directory=base_directory, 
                model=model, 
                scenario=scenario, 
                code=code, 
                test_period_start='2015-01-01', 
                test_period_end='2015-12-31', 
                projection_period_start='2015-01-01', 
                projection_period_end='2100-12-30'
            )

            # Handling X_projections_scaled for processing
            # Ensure X_projections_scaled is not None before processing
            if X_projections_scaled is not None:
                X_proj_wt = Wavelet_transform(X_projections_scaled)
                X_proj_l1 = reshape_onlyinputdata(np.array(X_proj_wt), seq_length=seq_length)
                X_proj_l = X_proj_l1[f:]

                sim_pr = np.zeros((len(X_proj_l), 10))  # Adjust based on your model list
                sim_pr[:] = np.nan
                    
                for init in range(initm):  # Assuming 'models' is a list of trained ML models
                    y_project = DLmodels[init].predict(X_proj_l)
                    sim_proj = target_scalers[code].inverse_transform(y_project)
                    sim_pr[:, init] = sim_proj[:, 0]
                
                sim_p = pd.DataFrame(sim_pr, index=X_projections_dates[f+seq_length-1:])  # Use the dates starting from 'f' offset
                
                lower_bound_Project = np.percentile(sim_p, 2.5, axis=1)
                upper_bound_Project = np.percentile(sim_p, 97.5, axis=1)
                sim_pr_median = sim_p.median(axis=1)
                
                # Plotting
                pyplot.figure(figsize=(20, 6))
                pyplot.plot(sim_pr_median.index, sim_pr_median, 'g', label="Projected median", linewidth=0.9)
                pyplot.fill_between(sim_pr_median.index, lower_bound_Project, upper_bound_Project, facecolor=(1, 0.1, 0, 0.2), label='95% confidence interval', linewidth=1, edgecolor=(1, 0.2, 0, 0.2))
                pyplot.title(f"{model} CMIP6 {scenario}_{Class} for {code}", size=24, fontweight='bold')
                pyplot.ylabel('GWL [m asl]', size=24)
                pyplot.xlabel('Date', size=24)
                pyplot.legend()
                pyplot.tight_layout()
                
                save_path = os.path.join("./", f"Well_ID_project_{Class}_{model}_{scenario}_{code.replace('/', '_')}.png")
                # pyplot.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
                pyplot.close()



# # Now X_projections_np_global contains all the numpy arrays for each code, model, and scenario combination

# from matplotlib import pyplot
               
# for code in final_df['code_bss'].unique():
#     print(f"Code: {code}")
#     Class = coords[coords['code_bss'] == code]['Class'].values[0]
#     df = final_df[final_df['code_bss'] == code]
#     X = df.drop(columns=['val_calc_ngf','code_bss'])

#     y = df[['val_calc_ngf']]
#     minimum_date= pd.to_datetime('1970-01-01')
#     min_train_date = max(minimum_date, X['date'].min())
#     y_train = y[(X['date'] >= min_train_date) & (X['date'] <= '2014-12-01')]
    
#     target_scalers[code] = MinMaxScaler(feature_range=(0,1))
#     target_scalers[code].fit(np.array(y_train).reshape(-1,1))
#     for model in selected_models:
#         for scenario in scenarios:
#             print(os.path.join(base_directory, str(model), str(scenario), "{}.csv".format(code.replace('/','_'))))
#             X_test, X_projections= load_and_scale_data_for_single_code_model_scenario(
#                 base_directory=base_directory, 
#                 model=model, 
#                 scenario=scenario, 
#                 code=code, 
#                 test_period_start='2015-01-01', 
#                 test_period_end='2015-12-31', 
#                 projection_period_start='2015-01-01', 
#                 projection_period_end='2100-12-30'
#             )
            
#             # print(X_test)
#             # Process or save the test and projection dataframes as needed
#             # Example: print the keys to see which model and scenario combinations have data
            
#             # print("Test DataFrames:", X_test.keys())
#             # print("Projection DataFrames:", X_projections.keys())
            

#             # for key, X_proj_array in X_projections.items():
#             print(code)
#             print(X_projections)
#                 # Assuming Wavelet_transform and reshape_onlyinputdata functions are defined elsewhere
#             X_proj_wt = Wavelet_transform(X_projections)
            
#             X_proj_l1 = reshape_onlyinputdata(np.array(X_proj_wt), seq_length=seq_length)
            
#             X_proj_l = X_proj_l1[f:]

#             sim_pr = np.zeros((len(X_proj_l), 10))  # Adjust based on your model list
#             sim_pr[:] = np.nan
                
#             for init in range(initm):  # Assuming 'models' is a list of trained ML models
#                     y_project = DLmodels[init].predict(X_proj_l)
#                     sim_proj = target_scalers[code].inverse_transform(y_project)
#                     sim_pr[:, init] = sim_proj[:, 0]
                
#             sim_p = pd.DataFrame(sim_pr)
#             lower_bound_Project = np.percentile(sim_p, 2.5, axis=1)
#             upper_bound_Project = np.percentile(sim_p, 97.5, axis=1)
#             sim_pr_median = sim_p.median(axis=1)
                
#                 # Plotting
#             pyplot.figure(figsize=(20, 6))
#             pyplot.plot(sim_pr_median, 'g', label="Projected median", linewidth=0.9)
#             pyplot.fill_between(range(len(sim_pr_median)), lower_bound_Project, upper_bound_Project, facecolor=(1, 0.1, 0, 0.2), label='95% confidence interval', linewidth=1, edgecolor=(1, 0.2, 0, 0.2))
#             pyplot.title(f"{model} CMIP6 {scenario}_{Class} for {code}", size=24, fontweight='bold')
#             pyplot.ylabel('GWL [m asl]', size=24)
#             pyplot.xlabel('Date', size=24)
#             pyplot.legend()
#             pyplot.tight_layout()
#             save_path = os.path.join("./", f"Well_ID_project_{Class}_{model}_{scenario}_{code.replace('/', '_')}.png")
#             pyplot.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
#             pyplot.close()

            # for key, X_proj_array in X_projections.items():
            #     # Apply wavelet transform
            #     X_proj_wt = Wavelet_transform(X_proj_array)
            
            
            #     X_Project_l1=reshape_onlyinputdata(np.array(X_proj_wt),seq_length=seq_length)
                
            #     X_Project_l=X_Project_l1[f:]
                
                
            #     sim_pr = np.zeros((len(X_Project_l), initm))
            #     sim_pr[:] = np.nan
                    
            #     for init in range(initm):
                 
            #         y_project = models[init].predict(X_Project_l)
            #         sim_proj = target_scalers[code].inverse_transform(y_project)
                    
            #         sim_pr[:, init] = sim_proj[:, 0]
            
            #       sim_p = pd.DataFrame(sim_pr)   

                  
            #       lower_percentile = 2.5  # 2.5th percentile
            #       upper_percentile = 97.5  # 97.5th percentile
                 
            #       lower_bound_Project = np.percentile(sim_p, lower_percentile, axis=1)
            #       upper_bound_Project = np.percentile(sim_p, upper_percentile, axis=1)
                 
            #       sim_pr_median=sim_p.median(axis=1)
                 
            #       sim_project=np.asarray(sim_pr_median).reshape(-1, 1)
                 
            #       from matplotlib import pyplot
            #       pyplot.figure(figsize=(20,6))
            #       # Extend_index=X_extend.date
            #       from matplotlib import rc
                 
            #       pyplot.plot( sim_project, 'g', label ="projected median", linewidth = 0.9)
                 
            #       pyplot.fill_between(lower_bound_Project,
            #             upper_bound_Project, facecolor = (1,0.1,0,0.2),
            #             label ='95% confidence extend',linewidth = 1,
            #             edgecolor = (1,0.2,0,0.2))  
                 
            #       pyplot.title((str(modtype)+' '+ "CMIP6"+str(wavelet)+str(code)+str(model)+str(scenario)), size=24,fontweight = 'bold')
                 
            #       pyplot.ylabel('GWL [m asl]', size=24)
            #       pyplot.xlabel('Date',size=24)
            #       pyplot.legend()
            #       pyplot.tight_layout()
                 
            #       pyplot.savefig('./'+'Well_ID_project'+str(modtype)+str(wavelet)+str(initm)+str(model)+str(scenario)+'{}.png'.format(code.replace('/','_')), dpi=600, bbox_inches = 'tight', pad_inches = 0.1)
                 
              



# fig, axs = plt.subplots(nrows=1, ncols=len(final_df['code_bss'].unique()), figsize=(20,5))

# i = 0
# for code in final_df['code_bss'].unique():

#     df = final_df[final_df['code_bss'] == code]
#     Xcorr = df.drop(columns=['code_bss','qualification'])
#     # Create a correlation matrix between the variables in df2
#     corr = Xcorr.corr()

#     corr_specific_variable = corr['val_calc_ngf']
#     corr_specific_variable = corr_specific_variable.drop('val_calc_ngf')
#     corr_specific_variable.plot.bar(ax=axs[i])
#     axs[i].set_title(code, size=24, fontweight='bold')
#     i += 1

#     # Use seaborn to create a heatmap of the correlation matrix
#     # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')

# import matplotlib.pyplot as plt

# for code in final_df['code_bss'].unique():
#     df = final_df[final_df['code_bss'] == code]
#     # plt.plot(df['date'], df['val_calc_ngf'], label=code)
#     plt.plot(df['date'], df['sro'], label='sro'+str(code))
# plt.xlabel('Date')
# plt.ylabel('val_calc_ngf')
# plt.legend()
# plt.show()
