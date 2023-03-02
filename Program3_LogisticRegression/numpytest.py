import numpy as np
import math

A = np.array([[1, 2, 3], [4,5,6],[7,8,9]])
print(A)
print("\n\n")

for i in range(len(A)):
    print(A[i])
    print("========")
print("\n\n")

for i in range(len(A)):
    for j in range(len(A[i])):
        print(A[i,j])

print(str(math.log10(100)))