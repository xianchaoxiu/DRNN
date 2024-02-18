from sklearn.neighbors import KernelDensity
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
time=datetime.datetime.now().strftime('%m%d%H%M')
## Author Jin XU, ecust
# X1=np.array([-2.1,-1.3,-0.4,1.9,5.1])

####利用KDE计算累积概率为alpha的对应X值
def compute_threshold(data,bw=None,alpha=0.99):
    ##先用sklearn的KDE拟合
    ##首先将数据尺度缩放到近似无穷大，然后根据近似微分求解
    data=data.reshape(-1,1)
    Min=np.min(data)
    Max=np.max(data)
    Range=Max-Min
    ##起点和重点
    x_start=Min-Range
    x_end=Max+Range
    ###nums越大之后估计的累积概率越大
    nums=2**12
    dx=(x_end-x_start)/(nums-1)
    data_plot=np.linspace(x_start,x_end,nums)
    if bw is None:
        ##最佳带宽选择
        ##参考：Adrian W, Bowman Adelchi Azzalini
        # - Applied Smoothing Techniques for Data Analysis_
        # the Kernel Approach with S-Plus Illustrations (1997)
        ##章节2.4.2 Normal optimal smoothing,中位数估计方差效果更好，
        #与matlab的ksdensity一致
        data_median=np.median(data)
        new_median=np.median(np.abs(data-data_median))/0.6745
        ##np.std(data,ddof=1)当ddof=1时计算无偏标准差，即除以n-1，为0时除以n
        bw=new_median*((4/(3*data.shape[0]))**0.2)
    print(bw)
    # print(data.shape)
    ##
    kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(data.reshape(-1,1))
    ###得到的是log后的概率密度，之后要用exp恢复
    log_pdf = kde.score_samples(data_plot.reshape(-1, 1))
    pdf=np.exp(log_pdf)
    ##画概率密度图
    plt.plot(data_plot,pdf)
    plt.show()
    ##CDF：累积概率密度
    CDF=0
    index=0

    while CDF<=alpha:
        CDF+=pdf[index]*dx
        index+=1

    return index,data_plot[index]


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K
from statsmodels.graphics.gofplots import qqplot
import pylab
import scipy.stats as stats
from numpy.random import seed
from scipy.io import savemat
seed(1)
tf.random.set_seed(2)
T=1000
ss=30405#30605
tt=ss+T

# seed=1
data = loadmat("DATA_ALL.mat")
#type(data)
data.keys() 
# datap = data['X'][ss:tt,:] 
# datam = data['Y'][ss:tt,:]
datap = data['D'][ss:tt,64:102] 
datam = data['D'][ss:tt,233:236] 
datam[...,1] = data['D'][ss:tt,241] 
datam[...,2] = data['D'][ss:tt,245] 
# # print(datap.shape)

# 
# plt.figure()
# plt.plot(datap)
# plt.plot(datam)
# plt.show()

# data = loadmat("I:/levelD.mat")
# datap = data['DD1'][60499:61499,:4]
# datam = np.ones((1000,2))
# datam[:,0] = data['DD1'][60499:61499,4]
# datam[:,1] = data['DD1'][60499:61499,4]
# print(datap.shape)
# print(datam.shape)
data = np.append(datap,datam,axis=1)
df = pd.DataFrame(data)  # array-->dataframe

"""
this heat map shows the correlation between different variables.
"""
def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=5, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=5)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Variable Correlation Heatmap", fontsize=14)
    # plt.show()

# show_heatmap(df)

## Data Preprocessing
nx=datap.shape[1]
ny=datam.shape[1]
Ith=0.001
split_fraction = 0.98
train_split = int(split_fraction * int(df.shape[0]))

learning_rate1 = 0.1
learning_rate2 = 0.01
batch_size = 10
epochs1 = 100
epochs2 = 100
sequence_length = 10
sequence_length2 = 10
unit1 = 5
nf=2
f1=-0.016
f3=0
f2=7
## Data Preprocessing
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std , data_mean , data_std

data_normalized , data_mean , data_std= normalize(df.values, train_split)
data_normalized = pd.DataFrame(data_normalized)
data_normalized.head()

train_data = data_normalized.loc[0 : train_split - 1]
val_data = data_normalized.loc[train_split:]
print("train_data:", train_data.shape)  #
print("val_data:", val_data.shape)
# Training dataset
x_train = train_data[[i for i in range(nx)]].values
y_train = train_data[[i for i in range(nx,nx+ny)]].values



def timeseries_dataset_from_array_diy(
    data,
    targets,
    sequence_length,
    sequence_stride,
    sampling_rate,
    batch_size,
    shuffle,
    seed,
    start_index,
    end_index):


  if start_index is None:
    start_index = 0
  if end_index is None:
    end_index = len(data)

  # Determine the lowest dtype to store start positions (to lower memory usage).
  num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
  if num_seqs < 2147483647:
    index_dtype = 'int32'
  else:
    index_dtype = 'int64'

  # Generate start positions
  start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
  if shuffle:
    if seed is None:
      seed = np.random.randint(1e6)
    rng = np.random.RandomState(seed)
    rng.shuffle(start_positions)

  sequence_length = math_ops.cast(sequence_length, dtype=index_dtype)
  sampling_rate = math_ops.cast(sampling_rate, dtype=index_dtype)

  positions_ds = dataset_ops.Dataset.from_tensors(start_positions).repeat()

  # For each initial window position, generates indices of the window elements
  indices = dataset_ops.Dataset.zip(
      (dataset_ops.Dataset.range(len(start_positions)), positions_ds)).map(
          lambda i, positions: math_ops.range(  # pylint: disable=g-long-lambda
              positions[i],
              positions[i] + sequence_length * sampling_rate,
              sampling_rate),
          num_parallel_calls=dataset_ops.AUTOTUNE)

  dataset = sequences_from_indices(data, indices, start_index, end_index)
  target_ds = sequences_from_indices(targets, indices, start_index, end_index)
  dataset = dataset_ops.Dataset.zip((dataset, target_ds))
  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  return dataset

def sequences_from_indices(array, indices_ds, start_index, end_index):
  dataset = dataset_ops.Dataset.from_tensors(array[start_index : end_index])
  dataset = dataset_ops.Dataset.zip((dataset.repeat(), indices_ds)).map(
      lambda steps, inds: array_ops.gather(steps, inds),  # pylint: disable=unnecessary-lambda
      num_parallel_calls=dataset_ops.AUTOTUNE)
  return dataset

dataset_train = timeseries_dataset_from_array_diy(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=batch_size,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None
)
# Validation dataset
x_val = val_data[[i for i in range(nx)]].values
y_val = val_data[[i for i in range(nx,nx+ny)]].values
# print("y_val1_mean=", np.mean(y_val[:,0]))
dataset_val = timeseries_dataset_from_array_diy(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=batch_size,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None
)

# print("dataset_train:", dataset_train.shape)
for batch in dataset_train.take(1):
    inputs, targets = batch

## Training                                                            111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
# RNN1= keras.layers.SimpleRNN(200,return_sequences=True,dropout=0.0)(inputs)
# outputs= keras.layers.SimpleRNN(targets.shape[2],return_sequences=True,dropout=0.0)(RNN1)

# lstm1= keras.layers.LSTM(50,return_sequences=True,dropout=0.0)(inputs)
# lstm2= keras.layers.LSTM(50,return_sequences=True)(lstm1)
# outputs= keras.layers.LSTM(targets.shape[2],return_sequences=True,dropout=0.0)(lstm2)

# lstm1= keras.layers.LSTM(50,return_sequences=True,dropout=0.0)(inputs)
# outputs= keras.layers.LSTM(targets.shape[2],return_sequences=True,dropout=0.0)(lstm1)

# outputs= keras.layers.SimpleRNN(targets.shape[2],return_sequences=True,dropout=0.2)(inputs)
# outputs= keras.layers.LSTM(targets.shape[2],return_sequences=True,dropout=0.2)(inputs)
outputs= keras.layers.GRU(targets.shape[2],return_sequences=True,dropout=0.2)(inputs)
# model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# , merge_mode='concat'
# outputs= keras.layers.Bidirectional(keras.layers.LSTM(targets.shape[2],return_sequences=True))(inputs)

# print(whole_seq_output.shape)
# print(whole_seq_output.shape)
# outputs = keras.layers.Conv1D(
#             filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
#         )(lstm_out)
# outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate1, decay=0.1), loss="mae")
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0), loss="mae")
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0), loss=loss_diy)
model.summary()
#dataset_train = dataset_train.repeat(epochs1)
# dataset_val = dataset_val.repeat(epochs1)
history = model.fit(
    dataset_train,
    epochs=epochs1,
    validation_data=dataset_val,
)

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()


# visualize_loss(history, "Training and Validation Loss")
##################################### Sigma_r  ##################################################################
m=int(split_fraction*T/sequence_length)   #### 500*0.8/50
xx = np.ones((m,sequence_length,nx)) #新建三维数组，且初始值为1
yy = np.ones((m,sequence_length,ny))
for i in range(m): 
    xx[i,:,:] = x_train[i*sequence_length:(i+1)*sequence_length,:]
    yy[i,:,:] = y_train[i*sequence_length:(i+1)*sequence_length,:]
model.predict(xx)
residual=model.predict(xx)-yy
residual_train=np.ones((m*sequence_length,ny))
for i in range(m):
    residual_train[i*sequence_length:(i+1)*sequence_length,:]=residual[i,:,:]
# print("residual_train", residual_train.shape)

residual_train1=residual_train
r_mean1 = residual_train.mean(axis=0)
Sigma_r1=np.cov(residual_train.T)

# Save the model
model.save('models/'+time+'_shipall_model_pre.h5')
# # Load the model
# reconstructed_model = keras.models.load_model("ship_model1.h5") # 曾报错 解决见https://github.com/keras-team/keras/issues/3977
########################################################## d00 ###################################################################
data = loadmat("DATA_ALL.mat")
data.keys() 
# datap_te = data['X'][ss:tt,:] 
# datam_te = data['Y'][ss:tt,:] 
datap_te = data['D'][ss:tt,64:102] 
datam_te = data['D'][ss:tt,233:236] 
datam_te[...,1] = data['D'][ss:tt,241] 
datam_te[...,2] = data['D'][ss:tt,245] 
data_te = np.append(datap_te,datam_te,axis=1)
# data = loadmat("I:/levelD.mat")
# data.keys() 
# datap_te = data['DD1'][60499:61499,:4]
# datam_te = np.ones((1000,2))
# datam_te[:,0] = data['DD1'][60499:61499,4]
# datam_te[:,1] = data['DD1'][60499:61499,4]
# data_te = np.append(datap_te,datam_te,axis=1)
data_te=(data_te - data_mean) / data_std
# print("data_te", data_te.shape)
m=int(T/sequence_length)   
xx = np.ones((m,sequence_length,nx)) #新建三维数组，且初始值为1
yy = np.ones((m,sequence_length,ny))
for i in range(m):
    x_te = data_te[i*sequence_length:(i+1)*sequence_length,:nx] 
    y_te = data_te[i*sequence_length:(i+1)*sequence_length,nx:] 
    xx[i,:,:] = x_te
    yy[i,:,:] = y_te
# model.predict(xx)
# print("xx", xx.shape)
# print("predict", model.predict(xx).shape)
residual_te=model.predict(xx)-yy
# print("residual_te", residual_te.shape)
residual_00=np.ones((m*sequence_length,ny))
for i in range(m):
    residual_00[i*sequence_length:(i+1)*sequence_length,:]=residual_te[i,:,:]
################ GaussCheck #######################
a=1
r=np.random.randn(residual_00.shape[0],residual_00.shape[1])
Indicator=np.square(np.mean(1/a*np.log((np.exp(a*residual_00)+np.exp(-a*residual_00))/2),0)-np.mean(1/a*np.log((np.exp(a*r)+np.exp(-a*r))/2),0))
print("Indicator",Indicator)
savemat('shipIndicator1.mat', {'Indicator':Indicator})


def GaussCheck(x):
    # Calculate quantiles and least-square-fit curve
    (quantiles, values), (slope, intercept, r) = stats.probplot(x, dist='norm')

    #plot results
    plt.plot(values, quantiles,'+b')
    plt.plot(quantiles * slope + intercept, quantiles, 'r')

    # define ticks
    ticks_perc=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]

    #transfrom them from precentile to cumulative density
    ticks_quan=[stats.norm.ppf(i) for i in ticks_perc]

    #assign new ticks
    plt.yticks(ticks_quan,ticks_perc)
    
    plt.ylabel("Probability")
   

    #show plot
    plt.grid()
    # plt.show()
for i in range(ny):
    # plt.subplot(11,1,i+1)
    plt.figure()
    GaussCheck(residual_00[:,i])
# plt.figure()
# plt.subplot(3,1,1)
# GaussCheck(residual_00[:,0])
# plt.xlabel("$r_1$")
# plt.subplot(3,1,2)
# GaussCheck(residual_00[:,1])
# plt.xlabel("$r_2$")
# plt.subplot(3,1,3)
# GaussCheck(residual_00[:,3])
# plt.xlabel("$r_4$")
# sfig = plt.gcf() # 'get current figure'
# sfig.savefig('pp1.eps', format='eps', dpi=1000)
plt.show()
residual_pre=residual_00
############### Fault detection ######################

# dfn, dfd = 11, 400-11
# threshold=(11*(400*400-1)/(400*(400-11)))*stats.f.ppf(0.99,dfn,dfd)
FAR=0
FDR=0
T_square1=np.ones((residual_00.shape[0],1))
for i in range(residual_00.shape[0]):
    r=residual_00[i,:]-r_mean1
    T_square1[i,:]=np.dot(np.dot(r,np.linalg.inv(Sigma_r1)),r.T)
i,th1=compute_threshold(T_square1,bw=None,alpha=0.99)
print("threshold",th1)
# input()
FAR=np.sum(T_square1>th1)/T*100
print("FAR:", FAR, "%")
plt.figure()
plt.plot(T_square1)
threshold=th1*np.ones((residual_00.shape[0],1))
plt.plot(threshold)
plt.show()
######################################################### residual_fault1 ###############################################################
data = loadmat("DATA_ALL.mat")
data.keys() 
datap_f = data['D'][ss:tt,64:102] 
datam_f = data['D'][ss:tt,233:236] 
datam_f[...,1] = data['D'][ss:tt,241] 
datam_f[...,2] = data['D'][ss:tt,245] 

# datap_f = data['X'][ss:tt,:] 
# datam_f = data['Y'][ss:tt,:] 
data_f = np.append(datap_f,datam_f,axis=1)
# data = loadmat("I:/levelD.mat")
# data.keys() 
# datap_f = data['DD1'][60499:61499,:4]
# datam_f = np.ones((1000,2))
# datam_f[:,0] = data['DD1'][60499:61499,4]
# datam_f[:,1] = data['DD1'][60499:61499,4]
# data_f = np.append(datap_f,datam_f,axis=1)
data_fault=np.ones((nf,T,nx+ny))
residual_fault1=np.ones((nf,T,ny))
for i in range(nf):
    data_fault[i,:,:] = data_f
    if i==1:
        data_fault[i,200:,i] = data_f[200:,i]+f1
        data_fault[i,200:,2:7] = data_f[200:,2:7]+f3
        data_fault[i,200:,15:] = data_f[200:,15:]+f2
    plt.figure()
    plt.plot(data_fault[i,:,i])
plt.show()
    # data_fault[i-1,:,:] = np.append(data_[i-1,:,0:15],data_[i-1,:,15:],axis=1)
data_fault=(data_fault - data_mean) / data_std
# print("data_fault", data_fault.shape)
m=int(T/sequence_length)
# 960%sequence_length    
xx = np.ones((m+1,sequence_length,nx)) #新建三维数组，且初始值为1
yy = np.ones((m+1,sequence_length,ny))
for j in range(nf):
    for i in range(m): 
        xx[i,:,:] = data_fault[j,i*sequence_length:(i+1)*sequence_length,:nx]
        yy[i,:,:] = data_fault[j,i*sequence_length:(i+1)*sequence_length,nx:]
    xx[m,:,:] = data_fault[j,T-sequence_length:,:nx]
    yy[m,:,:] = data_fault[j,T-sequence_length:,nx:]
    # print("xx", xx.shape)
# print("predict", model.predict(xx).shape)
    residual=model.predict(xx)-yy
# print("residual_te", residual_te.shape)
    # residual_00=np.ones((m*sequence_length,11))
    for i in range(m):
        residual_fault1[j,i*sequence_length:(i+1)*sequence_length,:]=residual[i,:,:]
    residual_fault1[j,T-sequence_length:,:]=residual[m,:,:]
# print("residual_fault1", residual_fault1.shape)

######################################################## 2222222222222222222222 ###################################################################################
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
# RNN1= keras.layers.SimpleRNN(200,return_sequences=True,dropout=0.0)(inputs)
# outputs= keras.layers.SimpleRNN(targets.shape[2],return_sequences=True,dropout=0.0)(RNN1)

lstm1= keras.layers.GRU(50,return_sequences=True,dropout=0.5)(inputs)
lstm2= keras.layers.GRU(100,return_sequences=True,dropout=0.5)(lstm1)
lstm3= keras.layers.GRU(100,return_sequences=True,dropout=0.5)(lstm2)
lstm4= keras.layers.GRU(50,return_sequences=True,dropout=0.5)(lstm3)
outputs= keras.layers.GRU(targets.shape[2],return_sequences=True)(lstm4)

# lstm1= keras.layers.SimpleRNN(50,return_sequences=True,dropout=0.2)(inputs)
# outputs= keras.layers.SimpleRNN(targets.shape[2],return_sequences=True,dropout=0.2)(lstm1)

# outputs= keras.layers.SimpleRNN(targets.shape[2],return_sequences=True,dropout=0.0)(inputs)
# outputs= keras.layers.LSTM(targets.shape[2],return_sequences=True,dropout=0.0)(lstm1)
# outputs= keras.layers.GRU(targets.shape[2],return_sequences=True,dropout=0.2)(inputs)

# outputs= keras.layers.Bidirectional(keras.layers.LSTM(targets.shape[2],return_sequences=True), merge_mode='concat')(inputs)

# print(whole_seq_output.shape)
# print(whole_seq_output.shape)
# outputs = keras.layers.Conv1D(
#             filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
#         )(lstm_out)
# outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
def loss_diy(y_true, y_pred, a= 1):  #### a->[1,2]
    r = K.random_normal(shape=(tf.shape(y_true)[0] , tf.shape(y_true)[1]), mean=0.0, stddev=1.0, seed=1)
    # mse + gaussianity
    # return keras.losses.mean_squared_error(y_true, y_pred) + K.square(K.mean(1/a*tf.keras.losses.logcosh(a*y_true, a*y_pred)) - K.mean(1/a*K.log((K.exp(a*r)+K.exp(-a*r))/2)))
    # mae + gaussianity
    # return keras.losses.mean_absolute_error(y_true, y_pred) + K.square(K.mean(1/a*tf.keras.losses.logcosh(a*y_true, a*y_pred)) - K.mean(1/a*K.log((K.exp(a*r)+K.exp(-a*r))/2)))
    # K.mean(K.square(K.square(y_pred-y_true)),axis=-1)
    #  gaussianity
    return K.square(K.mean(1/a*tf.keras.losses.logcosh(a*y_true, a*y_pred)) - K.mean(1/a*K.log((K.exp(a*r)+K.exp(-a*r))/2)))
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0), loss="mse")
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0), loss="mae")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate2, decay=0.0), loss=loss_diy)
model.summary()
# K.set_value(mnt_net_model.optimizer.lr,0.1*K.get_value(mnt_net_model.optimizer.lr))

history = model.fit(
    dataset_train,
    epochs=epochs2,
    validation_data=dataset_val,
    #steps_per_epoch=5,
    #validation_steps=1
)
model.save('models/'+time+'_shipall_model_post.h5')
##################################### Sigma_r2  ##################################################################
m=int(split_fraction*T/sequence_length)   #### 500*0.8/50
xx = np.ones((m,sequence_length,nx)) #新建三维数组，且初始值为1
yy = np.ones((m,sequence_length,ny))
for i in range(m): 
    xx[i,:,:] = x_train[i*sequence_length:(i+1)*sequence_length,:]
    yy[i,:,:] = y_train[i*sequence_length:(i+1)*sequence_length,:]
model.predict(xx)
residual=model.predict(xx)-yy
residual_train=np.ones((m*sequence_length,ny))
for i in range(m):
    residual_train[i*sequence_length:(i+1)*sequence_length,:]=residual[i,:,:]
# print("residual_train", residual_train.shape)
residual_train2=residual_train
r_mean = residual_train.mean(axis=0)
Sigma_r2=np.cov(residual_train.T)
########################################################## d00 ###################################################################
data = loadmat("DATA_ALL.mat")
data.keys() 

# datap_te = data['X'][ss:tt,:] 
# datam_te = data['Y'][ss:tt,:] 
datap_te = data['D'][ss:tt,64:102] 
datam_te = data['D'][ss:tt,233:236] 
datam_te[...,1] = data['D'][ss:tt,241] 
datam_te[...,2] = data['D'][ss:tt,245] 
data_te = np.append(datap_te,datam_te,axis=1)

# data = loadmat("I:/levelD.mat")
# data.keys() 
# datap_te = data['DD1'][60499:61499,:4]
# datam_te = np.ones((1000,2))
# datam_te[:,0] = data['DD1'][60499:61499,4]
# datam_te[:,1] = data['DD1'][60499:61499,4]
# data_te = np.append(datap_te,datam_te,axis=1)
data_te=(data_te - data_mean) / data_std
# print("data_te", data_te.shape)
m=int(T/sequence_length2)   
xx = np.ones((m,sequence_length2,nx)) #新建三维数组，且初始值为1
yy = np.ones((m,sequence_length2,ny))
for i in range(m):
    x_te = data_te[i*sequence_length2:(i+1)*sequence_length2,:nx] 
    y_te = data_te[i*sequence_length2:(i+1)*sequence_length2,nx:] 
    xx[i,:,:] = x_te
    yy[i,:,:] = y_te
model.predict(xx)
# print("xx", xx.shape)
# print("predict", model.predict(xx).shape)
residual_te=model.predict(xx)-yy
# print("residual_te", residual_te.shape)
residual_00=np.ones((m*sequence_length2,ny))
for i in range(m):
    residual_00[i*sequence_length2:(i+1)*sequence_length2,:]=residual_te[i,:,:]
# print("residual_00", residual_00.shape)
# r_mean = residual_00.mean(axis=0)
# Sigma_r2=np.cov(residual_00.T)
# ############### qqplot #######################
# qqplot(residual_te[0,:,0], line='s')
# plt.show()
# ############### ppplot #######################
# stats.probplot(residual_te[0,:,0], dist="norm", plot=pylab)
# pylab.show()
# ############### CDF plot #######################
# res = stats.relfreq(residual_te[0,:,0], numbins=50)
# x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
# y=np.cumsum(res.frequency)
# plt.plot(x,y)
# plt.title('Figure6 CDF')
# plt.show()
################ GaussCheck #######################
a=1
r=np.random.randn(residual_00.shape[0],residual_00.shape[1])
Indicator2=np.square(np.mean(1/a*np.log((np.exp(a*residual_00)+np.exp(-a*residual_00))/2),0)-np.mean(1/a*np.log((np.exp(a*r)+np.exp(-a*r))/2),0))
print("Indicator2",Indicator2)
print("Ith",Ith)
savemat('shipIth.mat', {'Ith':Ith})
savemat('shipIndicator2.mat', {'Indicator2':Indicator2})
r1=Indicator<Ith
r2=Indicator>Ith
residual_train=np.dot(residual_train1,np.diag(r1))+np.dot(residual_train2,np.diag(r2))
r_mean0 = residual_train.mean(axis=0)
Sigma_r0=np.cov(residual_train.T)
def GaussCheck_com(x,y):
    # Calculate quantiles and least-square-fit curve
    (xquantiles, xvalues), (xslope, xintercept, xr) = stats.probplot(x, dist='norm')
    (yquantiles, yvalues), (yslope, yintercept, yr) = stats.probplot(y, dist='norm')

    #plot results
    plt.plot(yvalues, yquantiles,'+k',label='pre')
    plt.plot(xvalues, xquantiles,'+b',label='post')
    plt.plot(xquantiles * xslope + xintercept, xquantiles, 'r')

    # define ticks
    ticks_perc=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]

    #transfrom th2em from precentile to cumulative density
    ticks_quan=[stats.norm.ppf(i) for i in ticks_perc]

    #assign new ticks
    plt.yticks(ticks_quan,ticks_perc)
    plt.ylabel("Probability")
    #show plot
    plt.legend()
    plt.grid()
    # plt.show()

for i in range(ny):
    # plt.subplot(11,1,i+1)
    plt.figure()
    GaussCheck(residual_00[:,i])
# plt.figure()
# plt.subplot(4,2,1)
# GaussCheck_com(residual_00[:,2],residual_pre[:,2])
# plt.xlabel("$r_3$")
# rr=["$r_5$","$r_6$","$r_7$","$r_8$","$r_9$","$r_{10}$","$r_{11}$"]
# for i in range(4,11):
#     plt.subplot(4,2,i-2)
#     GaussCheck_com(residual_00[:,i],residual_pre[:,i])
#     plt.xlabel(rr[i-4])
plt.show()
############### Fault detection ######################
# dfn, dfd = 15, 400-15
# th2=(15*(400*400-1)/(400*(400-15)))*stats.f.ppf(0.99,dfn,dfd)
# FAR=0
# FDR=0
# T_square2=np.ones((residual_00.shape[0],1))
# for i in range(residual_00.shape[0]):
#     r=residual_00[i,:]-r_mean
#     T_square2[i,:]=np.dot(np.dot(r,np.linalg.inv(Sigma_r2)),r.T)
#     if T_square2[i,:]>th2:
#         FAR += 1
# FAR=FAR/(m*sequence_length2)*100
# print("FAR:", FAR, "%")
# plt.figure()
# plt.plot(T_square2)
# th2reshold=th2*np.ones((residual_00.shape[0],1))
# plt.plot(th2reshold)
# plt.show()
######################################################### residual_fault2 ###############################################################
# data_=np.ones((21,960,52))
# data_fault=np.ones((21,960,52))
# residual_fault2=np.ones((21,960,11))
# for i in range(1,22):
#     if i<10:
#         name='d0'+str(i)+'_te'  
#     else:
#         name='d'+str(i)+'_te'
#     mat='I:/'+name+'.mat'
#     data = loadmat(mat)
#     data.keys() 
#     data_[i-1,:,:] = data[name]
#     data_fault[i-1,:,:] = np.append(data_[i-1,:,0:41],data_[i-1,:,41:],axis=1)
# data_fault=(data_fault - data_mean) / data_std
# print("data_fault", data_fault.shape)
m=int(T/sequence_length2)
# 960%sequence_length2    
xx = np.ones((m+1,sequence_length2,nx)) #新建三维数组，且初始值为1
yy = np.ones((m+1,sequence_length2,ny))
residual_fault2=np.ones((nf,T,ny))
for j in range(nf):
    for i in range(m): 
        xx[i,:,:] = data_fault[j,i*sequence_length2:(i+1)*sequence_length2,:nx]
        yy[i,:,:] = data_fault[j,i*sequence_length2:(i+1)*sequence_length2,nx:]
    xx[m,:,:] = data_fault[j,T-sequence_length2:,:nx]
    yy[m,:,:] = data_fault[j,T-sequence_length2:,nx:]
    # print("xx", xx.shape)
# print("predict", model.predict(xx).shape)
    residual=model.predict(xx)-yy
# print("residual_te", residual_te.shape)
    # residual_00=np.ones((m*sequence_length2,11))
    for i in range(m):
        residual_fault2[j,i*sequence_length2:(i+1)*sequence_length2,:]=residual[i,:,:]
    residual_fault2[j,T-sequence_length2:,:]=residual[m,:,:]
# print("residual_fault2", residual_fault2.shape)

########################################################## joint fault detection result  ###################################################################
dfn, dfd = ny, T-ny
th=(ny*(T*T-1)/(T*(T-ny)))*stats.f.ppf(0.99,dfn,dfd)
threshold=th*np.ones((residual_fault1.shape[1],1))
FAR=np.ones((nf,1))
FDR=np.ones((nf,1))
residual_fault=np.dot(residual_fault1,np.diag(r1))+np.dot(residual_fault2,np.diag(r2))
plt.figure()
plt.plot(residual_fault[0,...])
plt.show()
plt.savefig('residual.png')
print(Sigma_r0)
T_square=np.ones((residual_fault.shape[1],nf))
plt.figure()
for j in range(nf):
    for i in range(residual_fault1.shape[1]):
        r=residual_fault[j,i,:]-r_mean0
        T_square[i,j]=np.dot(np.dot(r,np.linalg.inv(Sigma_r0)),r.T)
    a1=T_square[:200,j]>th
    b1=T_square[200:,j]>=th
    FAR[j]=np.sum(a1)/200*100
    FDR[j]=np.sum(b1)/(T-200)*100
    fig=plt.subplot(nf,1,j+1)
    fig.set_title(str(j+1))
    plt.plot(T_square[:,j])
    plt.plot(threshold)
plt.show()
plt.savefig('T_square.png')

# plt.figure()
# for j in range(5,10):
#     for i in range(residual_fault1.shape[1]):
#         r=residual_fault[j,i,:]-r_mean0
#         T_square[i,j]=np.dot(np.dot(r,np.linalg.inv(Sigma_r0)),r.T)
#     a1=T_square[:160,j]>th
#     b1=T_square[160:,j]>=th
#     FAR[j]=np.sum(a1)/160*100
#     FDR[j]=np.sum(b1)/800*100
#     fig=plt.subplot(5,1,j-4)
#     fig.set_title(str(j+1))
#     plt.plot(T_square[:,j])
#     plt.plot(threshold)
# plt.show()

# # plt.figure()
# for j in range(10,15):
#     for i in range(residual_fault1.shape[1]):
#         r=residual_fault[j,i,:]-r_mean0
#         T_square[i,j]=np.dot(np.dot(r,np.linalg.inv(Sigma_r0)),r.T)
#     a1=T_square[:160,j]>th
#     b1=T_square[160:,j]>=th
#     FAR[j]=np.sum(a1)/160*100
#     FDR[j]=np.sum(b1)/800*100
#     fig=plt.subplot(5,1,j-9)
#     fig.set_title(str(j+1))
#     plt.plot(T_square[:,j])
#     plt.plot(threshold)
# plt.show()

# # plt.figure()
# for j in range(15,21):
#     for i in range(residual_fault1.shape[1]):
#         r=residual_fault[j,i,:]-r_mean0
#         T_square[i,j]=np.dot(np.dot(r,np.linalg.inv(Sigma_r0)),r.T)
#     a1=T_square[:160,j]>th
#     b1=T_square[160:,j]>=th
#     FAR[j]=np.sum(a1)/160*100
#     FDR[j]=np.sum(b1)/800*100
#     fig=plt.subplot(6,1,j-14)
#     fig.set_title(str(j+1))
#     plt.plot(T_square[:,j])
#     plt.plot(threshold)
# plt.show()
# savemat('shipRNN_th.mat', {'th':threshold})
# savemat('shipRNN_fd.mat', {'T_square':T_square})
print("average FAR:", FAR.mean(axis=0))
print("average FDR:", FDR.mean(axis=0))
print("FAR:", FAR)
print("FDR:", FDR)
