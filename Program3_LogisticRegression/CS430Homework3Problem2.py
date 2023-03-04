from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
#from scipy.optimize import minimize
import scipy
from sigmoid import *

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
epsilon = 0.000001


#####################################
# Loading Data
#####################################

#Read in data and save to an array for X1, an array for X2, and an array for Y
with open(Path(__file__).with_name("data2.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")        
        dataX1.append(float(line[0]))
        dataX2.append(float(line[1]))
        dataY.append(float(line[2]))
        #Save the X values into an array of array, with 1 as the first value for x0, to be stored into a numpy matrix
        data.append([float(1), float(line[0]), float(line[1])])

        line = input.readline()
X = np.array(data)
numPoints = len(dataX1)



#####################################
# Plotting Data Values
#####################################
#Separate the admitted from the non-admitted to plot with different markers
X1Pass = []
X2Pass = []
X1Fail = []
X2Fail = []
for i in range(numPoints):
    if dataY[i]:
        X1Pass.append(dataX1[i])
        X2Pass.append(dataX2[i])
    else:
        X1Fail.append(dataX1[i])
        X2Fail.append(dataX2[i])

plt.scatter(X1Pass, X2Pass, c="blue", marker="+", label="Admitted")
plt.scatter(X1Fail, X2Fail, c="yellow", marker="o", edgecolors="black", label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend(loc="best")



#####################################
# Cost
#####################################
#Function to calculate the cost given a set of theta's and data matrix X
def costFunction(theta, X):
    summation = 0.0
    #Calculate theta*XTranspose
    XT = X.transpose()
    thetaXT = np.matmul(theta, XT)

    #Get the summation of the cost function, where sigmoid(thetaXT) = h(x)
    for i in range(numPoints):
        summation += -dataY[i]*math.log(sigmoid(thetaXT[i])) - (1-dataY[i])*math.log(1 - sigmoid(thetaXT[i]))

    #Divide summation by the number of data points
    cost = summation / numPoints
    return cost

theta = np.array([0,0,0]) # Initialize theta parameters to all zero's
cost = costFunction(theta, X)
print("Cost with all thetas initialized to zero: " + str(cost))



#####################################
# Gradient
#####################################
theta = np.array([0,0,0]) #Initialize theta parameters to all zero's
gradient = np.array([0,0,0])
summation0, summation1, summation2 = 0,0,0

#calculate theta*XTranspose
XT = X.transpose()
thetaXT = np.matmul(theta, XT)

for i in range(numPoints):
    summation0 += (sigmoid(thetaXT[i]) - dataY[i])*X[i,0]
    summation1 += (sigmoid(thetaXT[i]) - dataY[i])*X[i,1]
    summation2 += (sigmoid(thetaXT[i]) - dataY[i])*X[i,2]
gradient[0] = summation0
gradient[1] = summation1
gradient[2] = summation2

gradient = gradient/numPoints

print("Gradient with all thetas initialized to zero: " + str(gradient))



#####################################
# Optimization
#####################################
print()
#use SciPy's minimization function to optimize the theta values for the costFunction
#Pass in costFunction to minimize, theta values to optimize, use the Nelder-Mead Simplex algorithm, and pass in X as the data argument
optimizedCost = scipy.optimize.minimize(costFunction, theta, method='nelder-mead', args=(X), options={'disp': True})
print()

print("Optimized theta values: " + str(optimizedCost.x))
print("Optimized cost value:   " + str(optimizedCost.fun))
theta = optimizedCost.x



#####################################
# Plotting Decision Boundary
#####################################
# Solved θ0 + θ1*x1 + θ2*x2 = 0 for x2 to get the below equation to graph
f = lambda x: (-theta[0] - theta[1]*x)/theta[2]

x = np.array([30,100])
y = np.array([30,100])

plt.plot(x, f(x), c='blue', label='decision boundary')
plt.show()