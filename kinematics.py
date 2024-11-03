import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
import timeit
import plotly.graph_objects as go

def spacefiller(x_boundary, y_boundary, z_boundary):
    N = 8*x_boundary*y_boundary*z_boundary
    N = int(N)
    return N

def constants():
    L = 1
    tol = 0.05
    x_boundary = 100
    y_boundary = 2.95
    z_boundary = 2.95
    Nxf = 2*x_boundary/tol
    Nyz = 2*y_boundary/tol
    #Convert Nxf to an integer to be used in an array input
    Nx = int(Nxf)
    Nyz = int(Nyz)
    limit = 20
    N_E = spacefiller(x_boundary, y_boundary, z_boundary)
    return L, tol, x_boundary, y_boundary, z_boundary, Nxf, Nyz, Nx, Nyz, limit, N_E

def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

# Rotation matrix for rotation around y-axis
def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

# Rotation matrix for rotation around z-axis
def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

#Total rotation matrix, takes input of the three angles
def total_matrix(theta_x, theta_y, theta_z):
    return rotation_matrix_x(theta_x) @ rotation_matrix_y(theta_y) @ rotation_matrix_z(theta_z)

def total_matrix_check(theta_x, theta_y, theta_z):
    R_22 = np.cos(theta_x)*np.cos(theta_z) - np.sin(theta_x)*np.sin(theta_y)*np.sin(theta_z)
    check = np.isclose(R_22, total_matrix(theta_x, theta_y, theta_z)[1, 1], atol=1e-5)
    if check == True:
        print('The rotation matrix is correct')
    else:
        print('The rotation matrix is incorrect')
    return

#Random angles for the rotation matrix
def random_angles():
   theta = random.uniform(0, 2*m.pi)
   phi = random.uniform(0, 2*m.pi)
   psi = random.uniform(0, 2*m.pi)
   return theta, phi, psi

#Random position for the eddy
def random_position(boundary=0, online=False):
    L = constants()[0]
    x_boundary = constants()[2]
    y_boundary = constants()[3]
    z_boundary = constants()[4]
    if boundary == 0:
        x = random.uniform(-x_boundary*L, x_boundary*L)
        y = random.uniform(-y_boundary*L, y_boundary*L)
        z = random.uniform(-z_boundary*L, z_boundary*L)
    else:
        x = random.uniform(-boundary*L, boundary*L)
        y = random.uniform(-boundary*L, boundary*L)
        z = random.uniform(-boundary*L, boundary*L)
    a = np.array([x, y, z])
    if online == True:
        a = np.array([x, 0, 0])
    a = a[:, np.newaxis]
    return a

#Set of all sensor positions on the x-axis
def positionvectorxaxis(boundary=0):
    L = constants()[0]
    x_boundary = constants()[2]
    Nx = constants()[7]
    if boundary == 0:
        x = np.linspace(-x_boundary*L, x_boundary*L, Nx)
    else:
        x = np.linspace(-boundary*L, boundary*L, Nx)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    pos_vector = np.vstack((x, y, z))
    return pos_vector

#Rotate x axis sensor positions
def positionvectorrotated(theta_x, theta_y, theta_z):
    R = total_matrix(theta_x, theta_y, theta_z)
    pos_vector = positionvectorxaxis()
    pos_vector_rotated = R @ pos_vector
    return pos_vector_rotated

def positionvectordisplaced(pos_vector, a):
    a = np.array(a).reshape(3, 1)
    pos_vector_displaced = pos_vector + a
    return pos_vector_displaced

def velocityline(pos_vector):
    L = constants()[0]
    x = pos_vector[0, :]  # Ensures x, y, z are 1D arrays of shape (200,)
    y = pos_vector[1, :]
    z = pos_vector[2, :]
    u = -y * np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    v = x * np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    w = np.zeros_like(u)
    return u, v, w

def velocity_test_function(pos_vector):
    L = constants()[0]
    x = pos_vector[0]
    y = pos_vector[1]
    z = pos_vector[2]
    u = np.exp(-x**2+y**2)
    v = 0
    w = 0
    return u, v, w

#Enhanced velocity line function that does not calculate the velocity at every point but only the local coordinates
def velocitylineenhanced(pos_vector):
    L = constants()[0]
    x = pos_vector[0]
    y = pos_vector[1]
    z = pos_vector[2]
    
    # Create a boolean mask where |x| < 2.95L
    mask = np.abs(x) < 4*L
    
    # Initialize u, v, and w as zero arrays
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    w = np.zeros_like(x)
    
    # Apply calculations only where the mask is True
    factor = np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    u[mask] = -y[mask] * factor[mask]
    v[mask] = x[mask] * factor[mask]
    # w remains zeros as per the original logic
    
    return u, v, w

def isotropic_turbulence_test_plot(pos_vector, u, v):
    tol = constants()[1]
    limit = 10
    max_index = int(limit/tol)
    x = pos_vector[0:max_index]
    N_u = len(u[0:max_index])
    s = np.arange(0, N_u)
    r = np.linspace(tol, max_index*tol, max_index)
    f = []
    for i in range(N_u):
        product_list = []
        for j in range(N_u - s[i]):
            product_list.append(u[j]*u[j+s[i]])
        #print("Product list mean for", i, "is", np.mean(product_list))
        f.append(np.mean(product_list))
    f = f/f[0]
    fig, ax = plt.subplots()
    ax.plot(r, f)
    ax.set_xlabel('s')
    ax.set_ylabel('f')
    plt.title('Isotropic turbulence')
    plt.show()