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


yBar = 0
xBar = 0
for i in range(numPoints):
    yBar += dataY[i]/numPoints
    xBar += dataX[i]/numPoints

totalSum1 = 0
totalSum2 = 0
for i in range(numPoints):
    totalSum1 += (dataX[i]*(dataY[i] - yBar))
    totalSum2 += (dataX[i]*(dataX[i] - xBar))

theta1 = totalSum1 / totalSum2

theta0 = yBar - (xBar*theta1)

xSum = 0
for i in range(numPoints):
    xSum += dataX[i]

print("theta 1: " + str(theta1))
print("theta 0: " + str(theta0))