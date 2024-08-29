from autograd import Value
import random
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, nin, activation):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation  
        
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu() if self.activation == 'relu' else act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, activation):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
            
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts, activation, step):
        sz_io = [nin] + nouts # len(sz_io) = len(nouts) + 1
        self.layers = [Layer(sz_io[i], sz_io[i+1], activation) for i in range(len(nouts))]
        self.step = step
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.
    
    def update_parameters(self):
        for p in self.parameters():
            p.data += self.step * -p.grad
