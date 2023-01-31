from pathlib import Path
import matplotlib.pyplot as plt


data = []
xlist = []
ylist = []

with open(Path(__file__).with_name("data.txt"), 'r') as input:
    line = input.readline()
    while line:
        line = line.split(",")
        x = float(line[0])
        xlist.append(x)
        y = float(line[1])
        ylist.append(y)
        data.append([x,y])
        line = input.readline()

#for point in data:
#   print("(" + str(point[0]) + "," + str(point[1]) + "), ", end='')

x_squared = []
xy = []

for point in data:
    x_squared.append(point[0]*point[0])
    xy.append(point[0]*point[1])

x_sum = 0
y_sum = 0
x_squared_sum = 0
xy_sum = 0

for i in range(len(data)):
    x_sum += data[i][0]
    y_sum += data[i][1]
    x_squared_sum += x_squared[i]
    xy_sum += xy[i]

print("=============================")

for i in range(len(data)):
    print(str(data[i][0]) + " " + str(data[i][1]) + " " + str(xy[i]) + " " + str(x_squared[i]))

print("=============================\n")

print("x sum: " + str(x_sum))
print("y sum: " + str(y_sum))
print("xy sum: " + str(xy_sum))
print("x2 sum: " + str(x_squared_sum))

print("")

slope = ((len(data) * xy_sum) - (x_sum * y_sum))/((len(data)*x_squared_sum) - (x_sum * x_sum))

intercept = (y_sum - (slope * x_sum))/(len(data))

print("Slope: " + str(slope))
print("Intercept: " + str(intercept))

print("n = " + str(len(data)))

plt.scatter(xlist, ylist, c ="red", marker="x", linewidths=0.5)

plt.show()