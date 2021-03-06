import numpy as np
import pandas
import math


def Adagrad(x, y, w, learning_rate, iter_time,dim):
    adagrad = np.zeros([dim, 1])
    eps = 0.00000001
    loss_list = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
        if (t % 100 == 0):
            if loss != float('inf') and loss != float('-inf'):
                loss_list.append(loss)
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    return w, loss_list


# RMSProp gradient decent:
def RMSProp(x, y, w, learning_rate, iter_time,dim):
    eps = 0.00000001
    alpa2 = 0.9  # 一般都取0.9
    prop = np.zeros([dim, 1])
    loss_list = []

    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
        if (t % 100 == 0):
            if(loss != float('inf') and loss != float('-inf')):
                loss_list.append(loss)
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
        if t == 0:
            prop = gradient ** 2
        else:
            prop = alpa2 * prop + (1 - alpa2) * (gradient ** 2)

        w = w - learning_rate * gradient / np.sqrt(prop + eps)

    return w, loss_list


##SGDm with momentum:
def SGDm(x, y, w, learning_rate, iter_time,dim):
    v = np.zeros([dim, 1])
    lamda = 0.9
    loss_list = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
        if loss != float('inf') and loss != float('-inf'):
            loss_list.append(loss)
        if (t > -1):
            print(str(t) + " times: " + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)
        v = lamda * v - learning_rate * gradient
        w = w + v
    return w, loss_list


##Adam optimiter:
def Adam(x, y, w, learning_rate, iter_time,dim):
    beta1 = 0.9
    beta2 = 0.99
    eps = 0.00000001
    monmentum = np.zeros([dim, 1])
    prop = np.zeros([dim, 1])
    loss_list = []

    for t in range(1, iter_time+100):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
        if (t%100==0):
            if loss != float('inf') and loss != float('-inf'):
                loss_list.append(loss)
            print(str(t) + " times: " + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)

        monmentum = beta1 * monmentum + (1 - beta1) * gradient
        if (t == 1):
            prop = gradient ** 2
        else:
            prop = beta2 * prop + (1 - beta2) * (gradient ** 2)

        monmentum_hat = monmentum / (1 - math.pow(beta1, t))
        prop_hat = prop / (1 - math.pow(beta2, t))

        w = w - learning_rate * monmentum_hat / (np.sqrt(prop) + eps)
    return w, loss_list



