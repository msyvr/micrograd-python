import math as m
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline # jno

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0 # nb: base case for nn_output: nn_output.grad = 1.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label
  
  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance (other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward

    return out
  
  def __mul__(self, other):
    other = other if isinstance (other, Value) else Value(other)        
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __rmul__(self, other):
    return self * other # would not be valid for matrix multiplication

  def __exp__(self):
    x = self.data
    out = Value(m.exp(x), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad # because d(e^x)/dx = e^x
    out._backward = _backward

    return out  

  def __pow__(self, constant):
    power = Value(constant)
    out = Value(self.data ** power.data, (self, ), 'pow')

    def _backward():
        self.grad += power.data * self.data ** (power.data - 1)
    

     
  
  def __sub__(self, other):
    out = Value(self.data - other.data, (self, other))

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __div__(self, other):
    out = Value(self.data / other.data, (self, other))

    def _backward():
      self.grad += (1/other.data) * out.grad
      other.grad += -self.data * (other.data**(-2)) * out.grad
    out._backward = _backward

    return out
  
  def tanh(self):
    x = self.data
    t = (m.exp(2*x) - 1)/(m.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out
  
  def backward(self): 
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append[v]      
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
        