import numpy as np
import pandas as pd
import sys
import math
from optimizer import Adagrad,Adam,SGDm,RMSProp
import matplotlib.pyplot as plt

#get the data
data=pd.read_csv("./train.csv",encoding='big5')
data=data.iloc[:,3:]
data[data=='NR']=0
raw_data=data.to_numpy()


#extract features and make data set
month_data={}
for month in range(12):
    sample=np.empty([18,480])
    for day in range(20):
        sample[:,day*24:(day+1)*24]=raw_data[18*(20*month+day):18*(20*month+day+1),:]
    month_data[month]=sample


x=np.empty([12*471,18*9],dtype=float)
y=np.empty([12*471,1],dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour>14:
                continue
            else:
                x[month*471+day*24+hour,:]=month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1)
                y[month*471+day*20+hour,0]=month_data[month][9,day*24+hour+9]

#Normalize:
mean_x=np.mean(x,axis=0)
std_x=np.std(x,axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j]!=0:
            x[i][j]=(x[i][j]-mean_x[j])/std_x[j]

#split data into two set:
x_train_set=x[:math.floor(len(x)*0.8),:]
y_train_set=y[:math.floor(len(y)*0.8),:]
x_validation=x[math.floor(len(x)*0.8):,:]
y_validation=y[math.floor(len(y)*0.8):,:]


#parameter:
dim=18*9+1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate=100
iter_time=100

#four different model:
adagrad_result,adagrad_loss=Adagrad(x,y,w,learning_rate,iter_time=1000,dim=dim)
rms_result,rms_loss=RMSProp(x,y,w,learning_rate,iter_time=1000,dim=dim)
sdg_result,sdg_loss=SGDm(x,y,w,learning_rate,iter_time=20,dim=dim)
adam_result,adam_loss=Adam(x,y,w,learning_rate,iter_time=1000,dim=dim)


#Visilization:

ax1=plt.subplot(221)
ax1.plot(range(0,1000,100),adagrad_loss,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 6)
ax1.set_xlabel("Adagrad")

ax2=plt.subplot(222)
ax2.plot(range(0,1000,100),rms_loss,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 6)
ax2.set_xlabel("RMSProp")

ax3=plt.subplot(223)
ax3.plot(range(0,20),sdg_loss,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 6)
ax3.set_xlabel("SDGm")

ax4=plt.subplot(224)
ax4.plot(range(0,1000,100),adam_loss,color='b',linestyle=':',marker = 'o',markerfacecolor='r',markersize = 6)
ax4.set_xlabel("Adam")

plt.show()
#np.save('weight.npy',w)


























