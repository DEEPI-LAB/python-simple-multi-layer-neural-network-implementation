# -*- coding: utf-8 -*-
"""
Neural Networks Representation of
AND, OR and XOR Logic Gates - Perceptron Algorithm
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-11-29
See here for more information :
    https://deep-eye.tistory.com/16
    https://deep-i.net
"""
import numpy as np
from matplotlib import pyplot as plt

class MLP():
    
    # Generate Simple logic Gate dataset
    def genLogicData(dtype = 'XOR'):
        x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64)
        if dtype == 'xor' or dtype == 'XOR' or dtype == 0:
            y = np.array([0,1,1,0],dtype=np.float64)
        elif dtype == 'or' or dtype == 'OR' or dtype == 1:
            y = np.array([0,1,1,1],dtype=np.float64)
        elif dtype == 'and' or dtype == 'AND' or dtype == 2:
            y = np.array([0,0,0,1],dtype=np.float64)
        else: raise Exception('No matching logic gates found')
        return x,y

    # Weights
    def dense(input_dim,output_dim):
        weights = np.random.randn(input_dim, output_dim) * 0.1
        return weights
    # Bias
    def bias(output_dim):
        bias = np.random.randn(output_dim) * 0.1
        return bias

    # Make Layer
    def layers(inputX,outputY,*weight):

        xnum,xdim = inputX.shape

        Layer = dict()
        Layer['input'] = inputX
        for ii,i in enumerate(weight):
            Layer['layer_{}_w'.format(ii)] = i
            Layer['layer_{}_b'.format(ii)] = np.random.randn(i.shape[1])
        Layer['output'] = outputY
        return Layer
    
    # def activation(types):
    #     if types == 'sigmoid':

    #%% MLP Train
    def Train(Layer,Option):

        ep = Option['ep']
        lr = Option['lr']
        # af = acivation(Option['activation'])
        # loss = loss(Option['loss'])
        
        
        input_num,input_dim = Layer['input'].shape
        result = []
        mse = []
        print("Start Learning")
        for z in range(ep):
            H = [[] for i in range(int(len(Layer)/2))]
            H[0] = Layer['input']

            # FEEDFORWARD
            for mm,m in enumerate(H):
                if mm == int(len(Layer)/2)-1: break
                H[mm+1] = 1 / (1 + np.exp(-(m.dot(
                                         Layer['layer_{}_w'.format(mm)]) +
                                         np.vstack([Layer['layer_{}_b'.format(mm)]]*m.shape[0]))))

            E =  (Layer['output'] - H[-1].T).T
            Z = [[] for i in range(int(len(Layer)/2))]
            Z[0] =  E 

            # BACK-PROPAGATION
            for mm,m in enumerate(Z):
                if mm == int(len(Layer)/2)-1: break
                try:
                    Z[mm+1] = m*(H[-(mm+1)] * (1-H[-(mm+1)]))
                except:
                    Z[mm+1] = m.T.dot(H[-(mm+1)] * (1-H[-(mm+1)]))

            # UPDATE
            for i in range(int(len(Layer)/2)-1):
                   try:
                        Layer['layer_{}_w'.format(i)] = Layer['layer_{}_w'.format(i)] + \
                          (lr * Z[-(i+1)].T.dot(H[i])).T
                   except:
                        Layer['layer_{}_w'.format(i)] = Layer['layer_{}_w'.format(i)] + \
                          (lr * Z[-(i+1)]*H[i].T)

            result.append(np.mean(E**2))

            print('EPOCH : %05d  MSE : %.04f    RESULTS : 0 0 -> %.03f 0 1 -> %.03f 1 0 -> %.03f 1 1 -> %.03f'
                  %(z,result[-1],np.round(H[-1][0],3),np.round(H[-1][1],3),np.round(H[-1][2],3),np.round(H[-1][3],3)))

            if Option['visualization'] == True:
                    mse.append(np.mean(E**2))

                    # plot graph
                    if z%100 == 0:
                        plt.xlabel('EPOCH')
                        plt.ylabel('MSE')
                        plt.title('MLP TEST')
                        plt.plot(mse)
                        plt.show()

    def option(ep = 10000, lr = 1,  activation = 'sigmoid',loss = 'gd',flag = True):

        Option = dict()
        Option['lr'] = lr
        Option['ep'] = ep
        Option['acitivation'] = activation
        Option['loss'] = loss
        Option['visualization'] = flag
        return Option
