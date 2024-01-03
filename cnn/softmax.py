import numpy as np

class Softmax:
    # A fully connected layer with softmax acivations

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce variance of our initial values

        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)


    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.

        Params :
        ---------
        input : an array with any dimensions.
        '''

        input = input.flatten()
        
        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        exp = np.exp(totals)

        return exp/np.sum(exp, axis = 0)
