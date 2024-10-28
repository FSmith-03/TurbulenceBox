import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
import timeit
import plotly.graph_objects as go
import maffioli_sensor_line as msl

#Constants
L = 1
tol = 0.05
x_boundary = 500
y_boundary = 2.95
z_boundary = 2.95
Nxf = 2*x_boundary/tol
Nyz = 2*y_boundary/tol
#Convert Nxf to an integer to be used in an array input
Nx = int(Nxf)
Nyz = int(Nyz)
limit = 20
N_E = msl.spacefiller(x_boundary, y_boundary, z_boundary)
#msl.rotation_check()
msl.manyeddies(N_E, limit, boundary=x_boundary)
#msl.plot3d(3)