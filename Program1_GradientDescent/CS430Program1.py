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
NUMLOOPS = 0

x_squared = []
xy = []

#Read in data and save to an array for X and an array for Y
with open(Path(__file__).with_name("data.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")
        dataX.append(float(line[0]))
        dataY.append(float(line[1]))

        line = input.readline()
numPoints = len(dataX)


testedEpsilon = 999

while testedEpsilon > epsilon:
    NUMLOOPS += 1

    summation0 = 0
    summation1 = 0
    for i in range(numPoints):
        summation0 += theta0 + theta1*dataX[i] - dataY[i]
        summation1 += (theta0 + theta1*dataX[i] - dataY[i])*dataX[i]
    tempTheta0 = theta0 - (alpha*(summation0 / numPoints))
    tempTheta1 = theta1 - (alpha*(summation1 / numPoints))

    testedEpsilon = math.sqrt(math.pow((tempTheta0-theta0), 2) + math.pow((tempTheta1-theta1), 2))
    


    theta0 = tempTheta0
    theta1 = tempTheta1
'''
    print("======================")
    print("LOOP " + str(NUMLOOPS))
    print("JValue = " + str(jValue))
    print("TestedEpsilon = " + str(testedEpsilon))
    print("Theta0 = " + str(theta0))
    print("Theta1 = " + str(theta1))
    print("======================\n")
'''

f = lambda x: theta1*x + theta0

x = np.array([min(dataX), max(dataX)])



plt.plot(x, f(x), c='purple', label="lin reg")
















#save lists for x^2 and x*y
for i in range(len(dataX)):
    x_squared.append(dataX[i]*dataX[i])
    xy.append(dataX[i]*dataY[i])

x_sum = 0
y_sum = 0
x_squared_sum = 0
xy_sum = 0

#find sums for x, y, x^2, and x*y
for i in range(len(dataX)):
    x_sum += dataX[i]
    y_sum += dataY[i]
    x_squared_sum += x_squared[i]
    xy_sum += xy[i]

print("=============================")

#for i in range(len(data)):
#    print(str(data[i][0]) + " " + str(data[i][1]) + " " + str(xy[i]) + " " + str(x_squared[i]))

print("=============================\n")

print("x sum: " + str(x_sum))
print("y sum: " + str(y_sum))
print("xy sum: " + str(xy_sum))
print("x2 sum: " + str(x_squared_sum))

print("")

slope = ((numPoints * xy_sum) - (x_sum * y_sum))/((numPoints * x_squared_sum) - (x_sum * x_sum))

intercept = (y_sum - (slope * x_sum))/(numPoints)

print("Slope: " + str(slope))
print("Intercept: " + str(intercept))

print("n = " + str(len(dataX)))

plt.scatter(dataX, dataY, c ="red", marker="x", linewidths=0.5)

f = lambda x: slope*x + intercept

x = np.array([min(dataX), max(dataX)])

plt.plot(x, f(x), c='g')


#plt.plot(X, X*slope + intercept, 'g')
#plt.plot(X, X*(-1) + 3, 'r')

#plt.xlim(4, 24)
#plt.ylim(-5, 25)



pred1x = 35
pred2x = 70

pred1y = f(pred1x)
pred2y = f(pred2x)

plt.scatter(pred1x, pred1y, c='orange', marker='^')
plt.scatter(pred2x, pred2y, c='black', marker='^')

pred1y2 = theta0 + theta1*pred1x
pred2y2 = theta0 + theta1*pred2x

plt.scatter(pred1x, pred1y2, c='blue', marker='o')
plt.scatter(pred2x, pred2y2, c='black', marker='o')

print("least squares:")
print("1: " + str(pred1y))
print("2: " + str(pred2y))
print("gradient descent:")
print("1: " + str(pred1y2))
print("2: " + str(pred2y2))






######################################
print("Number of loops: " + str(NUMLOOPS))


#plt.xlim(0,100)
#plt.ylim(0,100)
plt.scatter(dataX, dataY, c="red", marker="x", linewidths=0.5)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend(loc='best')
plt.show()