## An implementation of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
Reimplementing this as a warm up to implementing in Rust! Two blog posts accompany this implementation:
- ***[Designing neural networks: zero to micrograd](https://monicaspisar.com/posts/micrograd/)***
- ***[Tour de micrograd](https://monicaspisar.com/posts/micrograd-tour/)***

### Use
To experiment with a very simple binary classifier model that has randomly generated inputs and targets:
```bash
git clone git@github.com:msyvr/micrograd-python.git
cd micrograd-python
python train.py
```
Parameters for step size, number of epochs, activation function, etc., can be updated in `train.py` to see the effects of each on model convergence to targets.

### Evaluating the model
Model performance can be evaluated as a function of:
1. number of layers
2. nodes per layer
3. loss function

Separately, given a configuration of the above parameters, training efficiency can be evaluated as a function of:
1. step size
2. (maximum) training epochs


## TODO
