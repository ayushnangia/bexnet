"""

we use an optimizer to adjust our parameters during 
backpropagation.

"""

from bexnet.nn import NeuralNet

class Optimzer:
    def step(self,net:NeuralNet)-> None:
        raise NotImplementedError




class SGD(Optimzer):
    def __init__(self,lr:float=0.01)->None:
        self.lr=lr
    def step(self,net:NeuralNet)-> None:
        for param, grad in net.params_and_grads():
            param-=self.lr*grad
            