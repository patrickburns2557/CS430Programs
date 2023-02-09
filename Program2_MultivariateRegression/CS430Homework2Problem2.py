from pathlib import Path
import matplotlib.pyplot as plt ######################################################
import numpy as np
import math

#X1 = size of houses (ft^2)
#X2 = # of bedrooms
#X3 = price of house

################################SEPERATE ARRAYS FOR EACH DATA TO USE IN GRADIENT DESCENT
dataX1 = []
dataX2 = []
dataY = []
data = []
numPoints = 0
theta0 = 0
theta1 = 0
alpha = 0.01
epsilon = 0.0001

#Read in data and save to an array for X and an array for Y
with open(Path(__file__).with_name("data1.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")
        line[0] = float(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])
        data.append(line)

        
        dataX1.append(float(line[0]))
        dataX2.append(float(line[1]))
        dataY.append(float(line[2]))

        numPoints += 1
        line = input.readline()

####################################################################NORMALIZE DATA        
for i in range(numPoints):
    pass

#convert the data matrix into a numpy matrix
data = np.matrix(data)




#initially set as an arbitrary large value
testedEpsilon = 999
#keep looping the gradient descent algorithm until the tested episolon is lower than the target epsilon
while testedEpsilon > epsilon:
    summation0 = 0
    summation1 = 0
    for i in range(numPoints):
        summation0 += theta0 + theta1*dataX1[i] - dataY[i]
        summation1 += (theta0 + theta1*dataX1[i] - dataY[i])*dataX1[i]
    
    #calculate the new theta0 and theta1 based on the cost function and alpha
    tempTheta0 = theta0 - (alpha*(summation0 / numPoints))
    tempTheta1 = theta1 - (alpha*(summation1 / numPoints))

    #Calculate epsilon
    testedEpsilon = math.sqrt(math.pow((tempTheta0-theta0), 2) + math.pow((tempTheta1-theta1), 2))
    
    #simultaneous update of theta0 and theta1
    theta0 = tempTheta0
    theta1 = tempTheta1