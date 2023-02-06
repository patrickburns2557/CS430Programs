from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

dataX = []
dataY = []
numPoints = 0
theta0 = 0
theta1 = 0
alpha = 0.01
epsilon = 0.0001

#Read in data and save to an array for X and an array for Y
with open(Path(__file__).with_name("data.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")
        dataX.append(float(line[0]))
        dataY.append(float(line[1]))

        line = input.readline()
numPoints = len(dataX)


#initially set as an arbitrary large value
testedEpsilon = 999
#keep looping the gradient descent algorithm until the tested episolon is lower than the target epsilon
while testedEpsilon > epsilon:
    summation0 = 0
    summation1 = 0
    for i in range(numPoints):
        summation0 += theta0 + theta1*dataX[i] - dataY[i]
        summation1 += (theta0 + theta1*dataX[i] - dataY[i])*dataX[i]
    
    #calculate the new theta0 and theta1 based on the cost function and alpha
    tempTheta0 = theta0 - (alpha*(summation0 / numPoints))
    tempTheta1 = theta1 - (alpha*(summation1 / numPoints))

    #Calculate epsilon
    testedEpsilon = math.sqrt(math.pow((tempTheta0-theta0), 2) + math.pow((tempTheta1-theta1), 2))
    
    #simultaneous update of theta0 and theta1
    theta0 = tempTheta0
    theta1 = tempTheta1
print("Values of theta found using gradient descent:")
print("theta0: " + str(theta0))
print("theta1: " + str(theta1) + "\n")

#equation of the line
f = lambda x: theta1*x + theta0

#Use the found equation of the line to predict the new values of profits in areas with 35,000 people and 70,000 people
pred1x = 35
pred2x = 70
pred1y = f(pred1x)
pred2y = f(pred2x)
print("Predictions based on equation found from gradient descent:")
print("Profits in a city with a population of 35,000: $" + str(pred1y*10000))
print("Profits in a city with a population of 70,000: $" + str(pred2y*10000))


#plot the training data samples
plt.scatter(dataX, dataY, c="red", marker="x", linewidths=0.5, label="Training data")

#make the line plot go from the smallest x-value to the largest x-value
x = np.array([min(dataX), max(dataX)])
plt.plot(x, f(x), c='blue', label="Linear regression")


plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend(loc='best')
plt.show()