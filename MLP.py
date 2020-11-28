# -*- coding: utf-8 -*-
"""
Neural Network Multi-Layer Perseptron (XOR Problem)
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-11-28
See here for more information :
    https://deep-eye.tistory.com/16
    https://deep-i.net
"""

import numpy as np
from matplotlib import pyplot as plt

# train data (XOR Problem)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Intialization

# input - hidden layer
w1 = np.random.randn(2,2)
b1 = np.random.randn(1,2)

# hidden - output layer
w2 = np.random.randn(1,2)
b2 = np.random.randn(1)

# epoch
ep = 20000
# learning rate
lr =1
mse = []

# Neural Networks 2-2-1
for i in range(ep):
    
    E  = np.array([])
    result = np.array([])
    
    for j in range(len(x)):
        Ha = np.array([])
        
        # feedforward
        # input - hidden layer
        for k in range(len(w1)):
            Ha = np.append(Ha,1 / (1 + np.exp(-(np.sum(x[j] * w1[k]) + b1[0][k]))))
        
        # hideen - output layer
        Hb = 1 / (1 + np.exp(-(np.sum(Ha * w2) + b2)))
        
        # error
        E = np.append(E,y[j] - Hb)
        result = np.append(result,Hb)
        
        # back-propagation
        # output - hidden layer
        alpha_2 = E[j] * Hb * (1-Hb)
        
        # hidden - input layer
        alpha_1 = alpha_2 * Ha * (1-Ha) * w2
        
        # update
        w2 = w2 + (lr * alpha_2 * Ha)
        b2 = b2 + lr * alpha_2
        
        w1 = w1 + np.ones((2,2)) * lr * alpha_1 * x[j]
        b1 = b1 +  lr * alpha_1
        
    print('EPOCH : %05d MSE : %04f RESULTS : 0 0 => %04f 0 1 => %04f 1 0 => %04f 1 1 => %04f'
          %(i,np.mean(E**2),result[0],result[1],result[2],result[3]))
    
    mse.append(np.mean(E**2))

    # plot graph
    
    if i%100 == 0:
        plt.xlabel('EPOCH')
        plt.ylabel('MSE')
        plt.title('MLP TEST')
        plt.plot(mse)
        plt.show()

    
