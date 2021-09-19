# -*- coding: utf-8 -*-

import numpy as np
import os
import math
import scipy.io
import time
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

def make_theta(outputs):
    theta_list = []
    theta_0 = np.random.rand(25, 401) - 0.5
    theta_list.append(theta_0*0.24)
    theta_1 = np.random.rand(outputs, 26) - 0.5
    theta_list.append(theta_1*0.24)
    return theta_list

def make_y_array(y_label, y):
    for i in range(y_label.shape[0]):
        j = int(y_label[i])
        y[i][j] = 1
    return y

def sigmoid(array):
    (hight, width) = array.shape
    answer = np.empty((hight, width))
    for row in range(hight):
        for column in range(width):
            if array[row][column] >500: #expがバグるので大きめのところで切ってる
                answer[row][column] = 1
            elif array[row][column] < -500: #速度向上のため
                answer[row][column] = 0
            else:
                answer[row][column] =1.0/(1.0 + math.exp((-1) * array[row][column]))
    return answer

def addBias(vector):
    vector = np.insert(vector, [0], 1)
    vector = vector[:, np.newaxis] #次元数が0しかないので追加
    return vector

def Predict(data_list, theta):
    data_list = addBias(data_list) #バイアス追加
    a2 = sigmoid(addBias(np.dot(theta[0], data_list)))
    a3 = sigmoid(np.dot(theta[1], a2))
    return a2, a3

def CostFunction(x,y,theta,lam):
    num_data_list = x.shape[0]
    Cost = 0 
    for m in range(num_data_list):
        y_m = (y[m])[:,np.newaxis]
        log1 = np.log(Predict(x[m],theta)[1])
        log2 = np.log(1-(Predict(x[m],theta)[1]))
        Cost += np.sum((-1)*y_m*log1)+np.sum((-1)*(1-y_m)*log2)
    Cost += (1/2)*lam*(np.sum(theta[0][:, : 400]**2) + np.sum(theta[1][:, : 25]**2))
    return Cost/num_data_list

def Backpropagation(x,y,theta,lam, training_set_number):
    test_x = x[training_set_number:]
    test_y = y[training_set_number:]

    training_x = x[:(training_set_number-1)]
    training_y = y[:(training_set_number-1)]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax2.set_ylim([0, 100])
    ax3.set_ylim([0, 100])

    plt_J = []
    plt_acc = []
    plt_test_acc = []
    iter_num = []
    
    num_data_list = training_x.shape[0]
    m = num_data_list
    iter = 1
    while True:
        try:
            DELTA_1 = []
            DELTA_2 = [] #初期化
            for M in range(num_data_list):
                x_m = training_x[M]
                y_m = training_y[M][:, np.newaxis]
                a1 = addBias(x_m)
                (a2, a3) = Predict(x_m,theta)

                delta_2 = []
                delta_3 = []
                
                delta_3 = a3 - y_m
                delta_2 = (np.dot(theta[1].T, delta_3))*((a2)*(1-a2))
                delta_2 = np.delete(delta_2,0,0) #delta[0]を除去
                if M == 0:
                    DELTA_2 = np.dot(delta_3,a2.T)
                    DELTA_1 = np.dot(delta_2,a1.T)
                else:
                    DELTA_2 += np.dot(delta_3,a2.T)
                    DELTA_1 += np.dot(delta_2,a1.T)
            D_1 = calculate_D(theta[0],DELTA_1,lam,m)
            D_2 = calculate_D(theta[1],DELTA_2,lam,m)
            Refresh_theta(theta, D_1, D_2)
            
            J = CostFunction(training_x, training_y, theta, lam)
            acc = accuracy(training_x, training_y, theta)
            test_acc = accuracy(test_x, test_y, theta)
            
            plt_J.append(J)
            plt_acc.append(acc)
            plt_test_acc.append(test_acc)
            iter_num.append(iter)
            
            print(f"{iter} th Cost = ", J)
            if iter % 10 == 0:
                print("training_accuracy = ", acc, "%", "\ntest_accurancy = ", test_acc, "%")
                end = time.time()
                print("経過時間 = ", round((end-start), 2), "秒")
            iter += 1
        except KeyboardInterrupt:
            ax1.plot(iter_num,plt_J ,"b-")
            ax2.plot(iter_num,plt_acc  ,"r-")
            ax3.plot(iter_num,plt_test_acc  ,"c-")
            plt.show()
            break
    return(CostFunction(x, y, theta, lam))

def calculate_D(theta,DELTA,lam,m):
    D = np.zeros((theta.shape))
    for i in range(theta.shape[0]):
        if i == 0:
            D[0] = (1/m) * DELTA[0]
        else:
            D[i] = (1/m) * (DELTA[i] + (lam * theta[i]))
    return D

def Refresh_theta(theta,D1,D2):
    theta[0] = theta[0] - D1
    theta[1] = theta[1] - D2
    return

def accuracy(x, y, theta):
    correct_number = 0
    num_data_list = x.shape[0]
    m = num_data_list
    for i in range(m):
        pred = Predict(x[i], theta)[1]
        answer = y[i]
        if np.argmax(pred) == np.argmax(answer):
            correct_number += 1
    percent = round((correct_number/m)*100, 4)
    return percent


(outputs, lam, training_set_number) = (10, 0.1, 3000)

dir_path = os.path.dirname(__file__)
mat_path = os.path.join(dir_path, "ex4data1.mat")

X = (scipy.io.loadmat(mat_path)["X"])
y_label = (scipy.io.loadmat(mat_path)["y"])
y_label[np.where(y_label == 10)] = 0
Y = np.zeros((y_label.shape[0], outputs))
Y = make_y_array(y_label, Y)

p = np.random.permutation(X.shape[0])#シャッフル
X = X[p]
Y = Y[p]

theta_list = make_theta(outputs) #theta 初期化

#前準備完了

print("\nX_shape = ", X.shape)
print("y_shape = ", y_label.shape)
for i in range(2):
    print("theta_%s_shape = " % i, theta_list[i].shape)
print("Initial Cost = ", CostFunction(X, Y, theta_list, lam))
print("initial accuracy = ", accuracy(X, Y, theta_list), "%")
print("\nctrl + C を押した段階で学習を終了するよ!\n")

start = time.time()

Backpropagation(X, Y, theta_list, lam, training_set_number)

print("\nlast accuracy = ", accuracy(X, Y, theta_list), "%")