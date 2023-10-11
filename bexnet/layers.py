"""

Layers are an essential part of a Neural network.
There are many types of layers.
They need to have inputs do a forward pass and a backward gradient propagation.


"""
import numpy as np
from bexnet.tensor import Tensor
from typing import Dict,Callable
class Layer:
    def __init__(self)->None:
        self.params: Dict[str,Tensor]={}
        self.grads: Dict[str,Tensor]={}
    def forward(self, inputs:Tensor)->Tensor:
        """
        Producing the output to the given inputs

        """
        raise NotImplementedError
    def backward(self,grad:Tensor)-> Tensor:
        """
        Performing Backpropagation on this layer using gradient
        """
        raise NotImplementedError
    
class Linear(Layer):
    """
    Implements a linear layer, performing an affine transformation.
    
    This layer computes the operation `output = input * weight + bias`, 
    where `weight` and `bias` are learnable parameters.
    
    Inherits from:
    Layer: A base class for different types of layers in a neural network.
    
    Attributes:
    weight (Tensor): The weight matrix of the linear transformation.
    bias (Tensor): The bias vector of the linear transformation.
    
    Methods:
    forward(input: Tensor) -> Tensor:
        Performs the affine transformation and returns the result.
        
    backward(grad_output: Tensor) -> Tensor:
        Computes the gradient of the layer parameters and returns 
        the gradient with respect to the input.
    """
    def __init__(self,input_size:int, output_size:int)->None:
        #inputs will be (batch_size,input_size)
        #outputs will be (batch_size,input_size)
        super().__init__()
        self.params['w']=np.random.randn(input_size,output_size)
        self.params['b']=np.random.randn(output_size)
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computes the forward pass of the layer using the given inputs.
        
        The forward pass computes the affine transformation: 
        output = inputs @ self.params['w'] + self.params['b'], 
        where '@' denotes matrix multiplication, 'self.params['w']' 
        is the weight matrix, and 'self.params['b']' is the bias vector.
        
        Parameters:
        inputs (Tensor): The input tensor to the layer.
        
        Returns:
        Tensor: The output tensor from the layer.
        """
        self.inputs=inputs
        return inputs @ self.params['w']+self.params['b']
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Computes the backward pass of the layer using the given gradient.
        
        The backward pass computes the gradient of the layer parameters 
        and the gradient with respect to the input. In this method, 
        only the gradient with respect to the input is returned.
        
        Parameters:
        grad (Tensor): The gradient tensor of the loss with respect to the output of this layer.
        
        Returns:
        Tensor: The gradient tensor of the loss with respect to the input of this layer.
        """
        self.grads['b']=np.sum(grad,axis=0)
        self.grads['w']=self.inputs.T @ grad
        return grad @ self.params['w'].T        




F = Callable[[Tensor], Tensor]
class Activation(Layer):
    """
    Implements an activation layer to apply a specified activation function 
    to the input tensor.
    
    This layer applies an activation function element-wise to the input tensor, 
    which introduces non-linearity into the model, enabling it to learn from 
    the error backpropagated through the network.
    
    Inherits from:
    Layer: A base class for different types of layers in a neural network.
    
    Attributes:
    function (Callable): The activation function to be applied.
    """

    def __init__(self,f:F,f_prime:F)->None:
        super().__init__()
        self.f=f
        self.f_prime=f_prime
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs=inputs
        return self.f(inputs)
    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs)* grad





def tanh(x:Tensor)->Tensor:
    return np.tanh(x)

def tanh_prime(x:Tensor)-> Tensor:
    y=tanh(x)
    return 1-y**2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh,tanh_prime)
    
def sigmoid(x: Tensor) -> Tensor:
    """
    Computes the sigmoid activation function element-wise on the input tensor.
    
    Parameters:
    x (Tensor): The input tensor.
    
    Returns:
    Tensor: The result of applying the sigmoid function to the input tensor.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    """
    Computes the derivative of the sigmoid activation function 
    element-wise on the input tensor.
    
    Parameters:
    x (Tensor): The input tensor.
    
    Returns:
    Tensor: The result of applying the derivative of the sigmoid function 
            to the input tensor.
    """
    s = sigmoid(x)
    return s * (1 - s)

class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)
