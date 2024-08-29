import random
import math
import matplotlib.pyplot as plt
from nn import MLP
from autograd import Value

def train():
    '''
    This NN is set up as a binary classifier. 
    Each input list maps to a single output. 
    The targets list length is, thus, the number of 
    inputs used to train the model.
    '''
    num_inputs = 5
    len_input = 3

    # Define the neural network    
    layer_nodes = [4, 4, 1]
    activation_function = ''
    if activation_function != 'relu':
        activation_function = 'tanh'
    
    # Gradient descent parameters
    step_size = 0.05;
    num_epochs = 100;
    # option: loss_function    
    tolerance = 0.05;
    
    # Eval loops
    eval_loops = 10

    # Set up the run!
    print(f'step={step_size} : max epochs={num_epochs} : {tolerance=} : activation function={activation_function} : repeat?={eval_loops} \n')    
    epochs = []
    losses = []
    
    # Eval loops.    
    for _ in range(eval_loops):

        # Generate inputs and targets.
        inputs = []    
        for _ in range(num_inputs):
            new_input = [round(random.uniform(-3., 3.)) for _ in range(len_input)]
            inputs.append(new_input)  
        targets = [round(random.uniform(-1., 1.)) for _ in range(num_inputs)]
        # print(f'{targets=}\n{inputs=}')

        # Generate neural net.        
        net = MLP(len_input, layer_nodes, activation_function, step_size)

        # Gradient descent loops.
        for epoch in range(num_epochs):
            
            # Forward pass.
            nn_guesses = [net(input) for input in inputs]
            
            # Evaluate loss.
            # TODO options for which loss function
            square_errors = [(nn_guess - target)**2 for nn_guess, target in zip(nn_guesses, targets)]
            summed_square_errors = Value(0)
            for se in square_errors:
                summed_square_errors += se
                
            # Break at performance metric or max epochs.
            metric = math.sqrt(summed_square_errors.data) / num_inputs
            # TODO consider exp backoff strategy for step(error)       
            if metric <= tolerance or epoch == num_epochs - 1:
                epochs.append(epoch + 1)
                losses.append(summed_square_errors)
                break
               
            # Backprop to get local gradient wrt each parameter: d(output)/d(parameter).
            net.zero_grad()
            summed_square_errors.backward()        
            # Update model weights and biases.
            net.update_parameters()
                       
        # Optional: 
        # Return the model details
    
    for epoch, summed_square_errors in zip(epochs, losses):
        print(f'{epoch=} : loss={summed_square_errors.data}')
 
if __name__ == "__main__":
    train()