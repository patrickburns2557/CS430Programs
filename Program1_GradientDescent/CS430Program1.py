from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

dataX = []
dataY = []
x_squared = []
xy = []
numPoints = 0

#Read in data and save to an array for X and an array for Y
with open(Path(__file__).with_name("data.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")
        x = float(line[0])
        dataX.append(x)
        y = float(line[1])
        dataY.append(y)

        line = input.readline()
numPoints = len(dataX)

#for point in data:
#   print("(" + str(point[0]) + "," + str(point[1]) + "), ", end='')


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

#plt.xlim(-5, 25)

#plt.plot(dataX, dataX*slope + intercept, 'r')

plt.show()