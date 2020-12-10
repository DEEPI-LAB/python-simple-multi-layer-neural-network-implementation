# -*- coding: utf-8 -*-
"""
Neural Networks Representation of
AND, OR and XOR Logic Gates - Perceptron Algorithm
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-09
See here for more information :
    https://deep-eye.tistory.com/16
    https://deep-i.net
"""

from mlp import MLP

#%% Run Demo
# Generate Input Data
x,y = MLP.genLogicData(0)
# Initialize MLP Parameters
weight_1 = MLP.dense(2,10)
weight_2 = MLP.dense(10,1)
# Connect Layer
Layer = MLP.layers(x,y,weight_1,weight_2)
# Train Options
Option = MLP.option(10000,1,'sigmoid','gd',True)
MLP = MLP()
MLP.Train(Layer,Option)