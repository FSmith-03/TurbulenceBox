# This is a Python file named whatever you like. It demonstrates
#   automatic `fmodpy` wrapping.

import fmodpy
import numpy as np

code = fmodpy.fimport("matrix_mult.f90")
a = np.array([
    [1,2,3,4,5],
    [1,2,3,4,5]
], dtype=float, order='F')

b = np.array([
    [1,2],
    [3,4],
    [5,6],
    [7,8],
    [9,10]
], dtype=float, order='F')

print()
print("a:")
print(a)

print()
print("b:")
print(b)

print()
print("Numpy result")
print(np.matmul(a,b))

print()
print("Fortran result")
print(code.matrix_multiply(a,b))


print()
help(code.matrix_multiply)