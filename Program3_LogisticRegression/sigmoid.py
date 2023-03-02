import math
import numpy as np
#Function to perform the sigmoid function on the passed in value, list, or numpy array/matrix
def sigmoid(input):
    f = lambda x: 1 / (1 + math.pow(math.e, -x))  #lambda expression to easily reuse later for each type
    
    #if input is float or int, just perform the calculation and return result
    if type(input) is int or type(input) is float or type(input) is np.float64:
        output = f(input)
        return output
    
    #if input is a list (array),  iterate through the list, perform calculation on each value, and return
    elif type(input) is list: # if 
        output = []
        for x in input:
            output.append(f(x))
        return output
    
    #if input is a numpy array/matrix, iterate through every value, perform calculation on each value, and return
    elif type(input) is np.ndarray or type(input) is np.matrix:
        output = np.array(input)
        for i in range(len(input)):
            for j in range(len(input[i])):
                output[i,j] = f(input[i,j])
        return output
        
    
    