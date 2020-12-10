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
import matplotlib.pyplot as plt

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

    # def genMnistData(dtype = 'train'):
    #     pass

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

    def activation(self,types):
        # 0 : Sigmoid 1 : tanh 2  :ReLu
        if types == 'sigmoid': return 0
        elif types =='tanh' : return 1
        elif types == 'ReLu' : return 2
        else: raise Exception('No matching avtivataion function found')

    def loss(self,types):
        # 0 : gd 1 : momentum 2  :adam
        if types == 'gd': return 0
        elif types =='momentum' : return 1
        elif types == 'adam' : return 2
        else: raise Exception('No matching loss function found')
        
    def sigmoidFeed(self,iweight,oweight,bias):
        return 1 / (1 + np.exp(-(iweight.dot(oweight) + np.vstack([bias * iweight.shape[0]]))))
    def sigmoidBack(self,iweight,oweight):
        try: return iweight * (oweight * (1-oweight))
        except: return iweight.T.dot(oweight * (1-oweight))

    #%% MLP Train
    def Train(self,Layer,Option):

        ep = Option['ep']
        lr = Option['lr']
        af = self.activation(Option['activation'])
        loss = self.loss(Option['loss'])

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
                if af == 0 : H[mm+1] = self.sigmoidFeed(m,Layer['layer_{}_w'.format(mm)],
                                                        Layer['layer_{}_b'.format(mm)])

            E =  (Layer['output'] - H[-1].T).T
            Z = [[] for i in range(int(len(Layer)/2))]
            Z[0] =  E 

            # BACK-PROPAGATION
            for mm,m in enumerate(Z):
                if mm == int(len(Layer)/2)-1: break
                if af == 0 : Z[mm+1] = self.sigmoidBack(m,H[-(mm+1)])

            # UPDATE
            for i in range(int(len(Layer)/2)-1):
                   try:
                        Layer['layer_{}_w'.format(i)] = Layer['layer_{}_w'.format(i)] + \
                          (lr * Z[-(i+1)].T.dot(H[i])).T
                   except:
                        Layer['layer_{}_w'.format(i)] = Layer['layer_{}_w'.format(i)] + \
                          (lr * Z[-(i+1)]*H[i].T)

            result.append(np.mean(E**2))
            if z%50 == 0:

                print('EPOCH : %05d  MSE : %.04f    RESULTS : 0 0 -> %.03f 0 1 -> %.03f 1 0 -> %.03f 1 1 -> %.03f'
                      %(z,result[-1],np.round(H[-1][0],3),np.round(H[-1][1],3),np.round(H[-1][2],3),np.round(H[-1][3],3)))

                if Option['visualization'] == True:
                    mse = np.mean(E**2)
                    plt.xlabel('EPOCH')
                    plt.ylabel('MSE')
                    plt.title('MLP TEST')
                    plt.scatter(z, mse, s = 2,c='red')
                    plt.pause(0.001)
        plt.show()

    def option(ep = 10000, lr = 1,  activation = 'sigmoid',loss = 'gd',flag = True):

        Option = dict()
        Option['lr'] = lr
        Option['ep'] = ep
        Option['activation'] = activation
        Option['loss'] = loss
        Option['visualization'] = flag
        return Option
