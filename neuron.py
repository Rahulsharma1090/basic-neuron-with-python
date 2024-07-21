import numpy as np

np.random.seed(0)
x=np.array([[1,2,3,2.5],
                [2,5,-1,2],
                [-1.5,2.7,3.3,-0.8]])

class Layer_dense:
    def __init__(self,n_input,n_neuron):
        self.weights=0.10*np.random.randn(n_input,n_neuron)
        self.biases=np.zeros(n_neuron)
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

class Activation_ReLU:
    def forward(self,input):
        self.output= np.maximum(0,input)
        
    
    
    
layer1=Layer_dense(4,5)
layer2=Layer_dense(5,2)

layer1.forward(x)
print(layer1.output)
        
    
