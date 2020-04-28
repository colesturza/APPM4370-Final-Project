import numpy as np

def ELU(x, a=1):
    for i in range(len(x)):
        if x[i]<=0:
            x[i] = a*(np.exp(x[i])-1)

    return x

def sigmoid(x):
    return 1/(1+exp(-x))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def softplus(x):
    return np.log(1+np.exp(x))

def linear(x):
    return x

def ReLu(x):
    for i in range(len(x)):
        if x[i]<=0:
            x[i] = 0

    return x
