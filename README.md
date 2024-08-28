## Implementing [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)

Micrograd is based around building a neural network whose nodes are 'Value' class objects.

A Value class object includes implementations of linear algebraic functions and select nonlinear (activation) functions.

A Value object tracks the operations by which it was derived via pointers to its 'child(ren)', and implementation of local gradient derivation based on both its origin operation and children.

Note re 'child' terminology: From the perspective of starting at the neural network's output and working toward the input during gradient descent, the nodes' origin-tracking pointers are directed toward their contributor nodes. Hence, those contributors are denoted as a node's 'children'. Those contributor nodes seem, to me, more analogous to 'parents' when the neural network is viewed functionally, so I often have to remind myself to 'flip' that interpretation.

Mini-glossary:
- autograd: automatic gradient
- backpropagation: efficient computation of chained 
derivatives of outputs to (multiple layers of) inputs
- forward pass: compute the nn output for a particular input
'''