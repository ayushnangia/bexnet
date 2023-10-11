"""
Neural net is a collection of layers


"""

from typing import Sequence

from bexnet.layers import Layer
from bexnet.tensor import Tensor


class NeuralNet:
    def __init__(self,layers: Sequence[Layer])->None:
        self.layers=layers
    def forward(self,inputs: Tensor)->Tensor:
        for layer in self.layers:
            inputs=layer.forward(inputs)
        return inputs
    def backward(self,grad:Tensor)->Tensor:
        for layer in reversed(self.layers):
            grad=layer.backward(grad)
        return grad
    
