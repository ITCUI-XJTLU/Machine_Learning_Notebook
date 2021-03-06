import numpy as np
import pandas as pd
import csv


######test:
#fisrt,get data
testdata=pd.read_csv('./test.csv',header=None,encoding='big5')
test_data=testdata.iloc[:,2:]
test_data[test_data=='NR']=0
test_data=test_data.to_numpy()
test_x=np.empty([240,18*9],dtype=float)

for i in range(240): #填充数据
    test_x[i,:]=test_data[18*i:18*(i+1),:].reshape(1,-1)
std_test_x=np.std(test_x,axis=0)
mean_test_x=np.mean(test_x,axis=0)

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_test_x[j] != 0:
            test_x[i][j]=(test_x[i][j]-mean_test_x[j])/std_test_x[j]
test_x=np.concatenate((np.ones([240,1]),test_x),axis=1).astype(float)
#test_x=[240,18*9+1]





w=np.load('weight.npy')
#w=[18*9+1,1]
ans_y=np.dot(test_x,w)
print(len(ans_y))

##save the prediction:
with open('./prdiction.csv',mode='w',newline='') as f:
    csv_writer=csv.writer(f)
    header=['id','value']
    print(header)
    csv_writer.writerow(header)

    for i in range(len(ans_y)):
        row=['id:'+str(i),ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)



















