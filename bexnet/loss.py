""" loss function measurse the goodness of our model """

import numpy as np
from bexnet.tensor import Tensor


class Loss:
    def loss(self, predicted:Tensor, actual: Tensor)-> float:
        raise NotImplementedError
    def grad(self, predicted: Tensor, actual: Tensor)-> Tensor:
        raise NotImplementedError
    

class MSE(Loss):
    """
        Computes the Mean Squared Error (MSE) between predicted and actual values.
        
        Parameters:
        Loss (Tensor): A tensor containing the differences between predicted and actual values.
        
        Returns:
        float: The mean squared error computed as the mean of the squared elements in the Loss tensor.
    """
    def loss(self, predicted:Tensor, actual: Tensor)-> float:
        return np.sum((predicted-actual)**2)/ predicted.size
    def grad(self, predicted: Tensor, actual: Tensor)-> Tensor:
        return 2*(predicted-actual) / predicted.size
    