from pathlib import Path
import numpy as np
import math
import statistics

#X1 = size of houses (ft^2)
#X2 = # of bedrooms
#Y  = price of house

dataX1 = []
dataX2 = []
dataY = []
data = []
numPoints = 0
theta0 = 0
theta1 = 0
theta2 = 0
alpha = 0.01
epsilon = 0.0001

#Read in data and save to an array for X1, an array for X2, and an array for Y
with open(Path(__file__).with_name("data1.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")        
        dataX1.append(float(line[0]))
        dataX2.append(float(line[1]))
        dataY.append(float(line[2]))

        line = input.readline()
numPoints = len(dataX1)

#Normalize data
stdevX1 = statistics.stdev(dataX1)
stdevX2 = statistics.stdev(dataX2)
stdevY = statistics.stdev(dataY)
meanX1 = statistics.mean(dataX1)
meanX2 = statistics.mean(dataX2)
meanY = statistics.mean(dataY)
for i in range(numPoints):
    dataX1[i] = (dataX1[i] - meanX1) / stdevX1
    dataX2[i] = (dataX2[i] - meanX2) / stdevX2
    dataY[i] = (dataY[i] - meanY) / stdevY

    #save the X values into an array of arrays, with 1 as the first value for x0, to be stored into a numpy matrix
    data.append([float(1), dataX1[i], dataX2[i]])



#####################################
#PART 1: Gradient Descent
#####################################

#initially set as an arbitrary large value
testedEpsilon = 999
#keep looping the gradient descent algorithm until the tested episolon is lower than the target epsilon
while testedEpsilon > epsilon:
    summation0 = 0
    summation1 = 0
    summation2 = 0
    for i in range(numPoints):
        summation0 += theta0 + theta1*dataX1[i] + theta2*dataX2[i] - dataY[i]
        summation1 += (theta0 + theta1*dataX1[i] + theta2*dataX2[i] - dataY[i])*dataX1[i]
        summation2 += (theta0 + theta1*dataX1[i] + theta2*dataX2[i] - dataY[i])*dataX2[i]
    
    #calculate the new theta0, theta1, and theta2 based on the cost function and alpha
    tempTheta0 = theta0 - (alpha*(summation0 / numPoints))
    tempTheta1 = theta1 - (alpha*(summation1 / numPoints))
    tempTheta2 = theta2 - (alpha*(summation2 / numPoints))

    #Calculate epsilon
    testedEpsilon = math.sqrt(math.pow((tempTheta0-theta0), 2) + math.pow((tempTheta1-theta1), 2) + math.pow((tempTheta2-theta2), 2))
    
    #simultaneous update of theta's
    theta0 = tempTheta0
    theta1 = tempTheta1
    theta2 = tempTheta2

print("Gradient Descent: ")
print("theta0: " + str(theta0))
print("theta1: " + str(theta1))
print("theta2: " + str(theta2))



print("\n")
#####################################
#PART 2: Normal Equation
#####################################


#convert the data matrix into a numpy matrix
X = np.matrix(data)
Y = np.matrix(dataY)
Y = Y.transpose()





XT = X.transpose()

XTX = np.matmul(XT, X)

XTXInv = np.linalg.inv(XTX)

XTXInvXT = np.matmul(XTXInv, XT)

theta = np.matmul(XTXInvXT, Y)

print("Normal Equation:")
print("theta0: " + str(theta[0,0]))
print("theta1: " + str(theta[1,0]))
print("theta2: " + str(theta[2,0]))



predx1 = (1650 - meanX1)/stdevX1
predx2 = (3 - meanX2)/stdevX2

predgrady = theta0 + (theta1*predx1) + (theta2*predx2)
predgrady = (predgrady*stdevY) + meanY #"un-normalize" the result to get the actual prediction
print("Gradient Descent guess: " + str(predgrady))


prednormy = theta[0,0] + (theta[1,0]*predx1) + (theta[2,0]*predx2)
prednormy = (prednormy*stdevY) + meanY #"un-normalize" the result to get the actual prediction
print("Normal Equation guess:  " + str(prednormy))