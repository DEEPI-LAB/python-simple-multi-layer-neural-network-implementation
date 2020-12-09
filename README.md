# MLP
## General description
Python implementation of 'Multi-Layers Perceptron' Algorithm using only Numpy and Matplotlib.
## Requirements
You should install **numpy** and **matplotlib** for clustering and visualization.

    pip3 install numpy matplotlib

## Usage
### Generate Logic Gate Data
    x,y = MLP.genLogicData('XOR')
### Initialize Weights
    weight_1 = MLP.dense(2,10)
    weight_2 = MLP.dense(10,1)

    Layer = MLP.layers(x,y,weight_1,weight_2)
### Training Options
    Option = MLP.option(ep = 10000,lr = 1,'sigmoid','gd',visualization = True)
###  Training
    MLP.Train(Layer,Option)
## Test MLP
Open the **runDemo file**. You can just click **F5** in an **IDE environment** to see the sample data results.

![results](https://blog.kakaocdn.net/dn/bfBwoQ/btqPKuqgYMq/lpKcNXNJm6OK8BvNm7KyR0/img.png)
## Author
Jongwon Kim : [https://deep-eye.tistory.com/](https://deep-eye.tistory.com/)