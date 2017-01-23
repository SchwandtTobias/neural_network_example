
# coding: utf-8


# Original code comes from a demo NN program from the YouTube video https://youtu.be/h3l4qz76JhQ. The program creates an neural network that 
# simulates the exclusive OR function with two inputs and one output. 

# Another website for more testing is here:
# http://experiments.mostafa.io/public/ffbpann/

# That is the graph represented in this script.

#   L0          L1            L2
#
#        +--+-- H1 ------+
#       /  /              \
#      /  /                \
#   I1 --+----- H2 ------+  \    
#      \/                 \  \
#      /\                   - 0
#   I2 --+----- H3 --------/ / 
#      \  \                 /
#       \  \               /
#        +--+-- H4 ------+
#



# The program creates an neural network that simulates the exclusive OR function with two inputs and one output. 
import numpy as np


# In[1]: Sigmoid function
# The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this neural 
# net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with. 
# In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate. 
# 
# The implementation of this function does double duty. If the _CalculateDerivative = True flag is passed in, the function instead calculates the 
# derivative of the function, which is used in the error backpropogation step. 

def Linear(_Input, _CalculateDerivative = False):
  if _CalculateDerivative == True:
    return 1
  
  return _Input


def Sigmoid(_Input, _CalculateDerivative = False):
  if _CalculateDerivative == True:
  	return np.exp(_Input) / (np.exp(_Input) + 1) ** 2
  
  return 1 / (1 + np.exp(-_Input))


def TanH(_Input, _CalculateDerivative = False):
  if _CalculateDerivative == True:
    return 1 - pow(_Input, 2)    # Alternative: 1 - _Input ** 2

  return np.tanh(_Input)


ActivationFunction = Sigmoid


# In[2]: Input data
# The following code creates the input matrix. Although not mentioned in the video, the third column is for accommodating the 
# bias term and is not part of the input. 
InputData = np.array([[0,0,1],
                      [0,1,1],
                      [1,0,1],
                      [1,1,1]])


# In[3]: Output data
# The output of the XOR function follows. 
OutputData = np.array([[0],
                       [1],
                       [1],
                       [0]])


# In[4]:
# The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.
np.random.seed(1)
 

# In[5]: Synapses
# Now we intialize the weights to random values. Synapses0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix 
# because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). Synapses1 are the weights between the 
# hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there 
# is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not 
# to work well when all the weights start at the same value.
Synapses0 = 2 * np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
Synapses1 = 2 * np.random.random((4,1)) - 1  # 4x2 matrix of weights. (4 nodes x 1 output)


# In[6]: Main loop / training step
# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases. 
for IndexOfLoop in range(60000):  
    # Calculate forward through the network.
    Layer0 = InputData
    Layer1 = ActivationFunction(np.dot(Layer0, Synapses0))
    Layer2 = ActivationFunction(np.dot(Layer1, Synapses1))
    
    # Back propagation of errors using the chain rule. 
    ErrorOfLayer2 = OutputData - Layer2

    DeltaOfLayer2 = ErrorOfLayer2 * ActivationFunction(np.dot(Layer1, Synapses1), _CalculateDerivative = True)
    
    ErrorOfLayer1 = DeltaOfLayer2.dot(Synapses1.T)
    
    DeltaOfLayer1 = ErrorOfLayer1 * ActivationFunction(np.dot(Layer0, Synapses0), _CalculateDerivative = True)
    
    # Update weights (no learning rate term)
    Synapses1 += Layer1.T.dot(DeltaOfLayer2)
    Synapses0 += Layer0.T.dot(DeltaOfLayer1)

    if(IndexOfLoop % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        # print("Error L1: " + str(np.mean(np.abs(ErrorOfLayer1))) + " (" + str(np.mean(DeltaOfLayer1)) + ")")
        print("Error L2: " + str(np.mean(np.abs(ErrorOfLayer2))) + " (" + str(np.mean(DeltaOfLayer2)) + ")")

# See how the final output closely approximates the true output [0, 1, 1, 0]. If you increase the number of interations in the 
# training loop the final output will be even closer. 
print ("Output after training")
print (Layer2)


# In[7]: Using the trained AI
# See now the result of the AI after training
Input  = np.array([[0,1,1]]);
Layer1 = ActivationFunction(np.dot(Input, Synapses0))
Layer2 = ActivationFunction(np.dot(Layer1, Synapses1))

print ("Output of trained AI")
print (Layer2)