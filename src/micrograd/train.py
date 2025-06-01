import random
import math
import matplotlib.pyplot as plt
from nn import MLP
from autograd import Value

def generate_training_data(num_inputs, len_input):        
    inputs = []    
    for _ in range(num_inputs):
        new_input = [round(random.uniform(-3., 3.)) for _ in range(len_input)]
        inputs.append(new_input)  
    targets = [round(random.uniform(-1., 1.)) for _ in range(num_inputs)]
    return [inputs, targets]
    

def train(training_data, layer_nodes, activation, loss, tolerance, step_size, num_epochs):
    '''
    The network trains on inputs with corresponding targets. 
    In other words, the training data is a collection of 
    (input vector, output value) pairs.

    Training is done `num_evals` times to evaluate training for a fixed
    parameter set: { `num_epochs`, `tolerance`, `activation`, `loss` }.
           
    The post-training network parameters define a neural network that,
    nominally, can predict appropriate outputs for new inputs.
    
    '''
    
    # Set up the run
    num_evals = len(training_data)
    print(f'Parameters:\n{step_size=} || max epochs={num_epochs} || {tolerance=} || {activation=} || {num_evals=} \n')    
    fig, axs = plt.subplots(num_evals)
    fig.suptitle(f"Prediction loss (rms) vs training epoch\n{activation=}, {loss=}\n{step_size=}, epochs={num_epochs}, {tolerance=}")
    
    # Eval loops.    
    for loop in range(len(training_data)):
        inputs = training_data[loop][0]
        targets = training_data[loop][1]
        # print(f'{targets=}\n{inputs=}')

        # Generate neural net.        
        net = MLP(len(inputs[0]), layer_nodes, activation, step_size)
        # Training epochs.
        epochs = []
        losses = []
        losses_rms = []
        losses_rms_max = 0
        for epoch in range(num_epochs):
            # Forward pass.
            nn_guesses = [net(input) for input in inputs]
            # Evaluate loss.
            # TODO select loss function: errors = get_error(loss_function)
            square_errors = [(nn_guess - target)**2 for nn_guess, target in zip(nn_guesses, targets)]
            summed_errors = Value(0)
            for se in square_errors:
                summed_errors += se
                
            # Determine performance metric to break (else at max epochs).
            loss_rms = math.sqrt(summed_errors.data) / num_inputs
            # print(f"{loop=}, {epoch=}, loss={loss_rms}")

            epochs.append(epoch + 1)
            losses_rms.append(loss_rms)
            losses.append(summed_errors)

            # TODO consider exp backoff strategy for step(error)       
            if loss_rms <= tolerance:
                print(f'loss: {loss_rms} is <= tolerance: {tolerance} at epoch:{epoch}')
                break
            elif epoch == num_epochs - 1:
                print(f'final loss: {loss_rms} vs tolerance: {tolerance}')
                break            
               
            # Backprop to get local gradient wrt each parameter: d(output)/d(parameter).
            net.zero_grad()
            summed_errors.backward()        
            # Update model weights and biases.
            net.update_parameters()
                                   
        # Optional: 
        # Return the model details
        
        # Create the loss vs epoch plot for the current eval loop.
        # print(f'{epochs=} : {losses_rms=}')
        axs[loop].scatter(epochs, losses_rms)
        if max(losses_rms) > losses_rms_max:
            losses_rms_max = max(losses_rms)

    custom_xlim = (0, num_epochs)
    custom_ylim = (0, math.ceil(losses_rms_max*10/1)/10)
        
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    plt.show()
 
if __name__ == "__main__":
    
    # TODO parse args, define defaults 
    
    # Training data
    num_inputs = 5
    len_input = 3
    num_evals = 4    
    training_data = [generate_training_data(num_inputs, len_input) for _ in range(num_evals)]
       
    # Neural network   
    layer_nodes = [4, 4, 1] 
    # activation = 'relu'
    activation = 'tanh'    
    # loss = 'cc'
    loss = 'rms'
        
    # Gradient descent
    # TODO adapt tolerance for specific loss functions
    xtolerance = 0.01
    tolerance = math.sqrt(xtolerance/num_inputs)       
    step_size = 0.005
    num_epochs = 1000
    
    num_step_sizes = 4
    delta_step_size = step_size/20
    step_sizes = [step_size + x*delta_step_size for x in range(num_step_sizes)]
    
    num_tolerances = 5
    delta_tolerance = tolerance/20
    tolerances = [tolerance - x*delta_tolerance for x in range(num_tolerances)]
    
    ## Run evals over combinations of training parameters.
    # for step, tol in zip(step_sizes, tolerances):
    #     for tol in range(num_tol_steps):
    #         curr_step_size = step_size
    #         if 
    #         train(training_data, layer_nodes, activation, loss, tolerance, step_size, num_epochs)
    
    train(training_data, layer_nodes, activation, loss, tolerance, step_size, num_epochs)