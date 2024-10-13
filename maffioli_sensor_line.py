#We have a sensor line on the x axis from -40,000L to 40,000L

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m

#Constants
L = 1
tol = 0.05
x_boundary = 1000
y_boundary = 2.95
z_boundary = 2.95
Nxf = 2*x_boundary/tol
Nyz = 2*y_boundary/tol
#Convert Nxf to an integer to be used in an array input
Nx = int(Nxf)

# Rotation matrix for rotation around x-axis
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
    return rotation_matrix_z(theta_z) @ rotation_matrix_y(theta_y) @ rotation_matrix_x(theta_x)

#Random angles for the rotation matrix
def random_angles():
   theta = random.uniform(0, 2*m.pi)
   phi = random.uniform(0, 2*m.pi)
   psi = random.uniform(0, 2*m.pi)
   return theta, phi, psi

#Random position for the eddy
def random_position():
    x = random.uniform(-x_boundary*L, x_boundary*L)
    y = random.uniform(-y_boundary*L, y_boundary*L)
    z = random.uniform(-z_boundary*L, z_boundary*L)
    a = np.array([x, y, z])
    return a

#Set of all sensor positions on the x-axis
def positionvectorxaxis():
    x = np.linspace(-x_boundary*L, x_boundary*L, Nx)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    pos_vector = np.vstack((x, y, z))
    return pos_vector

#Position vector rotated, takes input of the three angles
def positionvectorrotated(theta_x, theta_y, theta_z):
    R = total_matrix(theta_x, theta_y, theta_z)
    pos_vector = positionvectorxaxis()
    pos_vector_rotated = R @ pos_vector
    return pos_vector_rotated

#Vector displacement, takes input of a position vector and a displacement vector
def positionvectordisplaced(pos_vector, a):
    a = a[:, np.newaxis]
    pos_vector_displaced = pos_vector + a
    return pos_vector_displaced

#Velocity line on the x-axis, takes input of a position vector
def velocityline(pos_vector):
    x = pos_vector[0]
    y = pos_vector[1]
    z = pos_vector[2]
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    w = np.zeros_like(u)
    return u, v, w

#Plotter for the sensor line on the x-axis, takes input of the position vector and the velocity components
def xaxisplotter(pos_vector, u, v, w):
    x = pos_vector[0]
    abs_velocity = np.sqrt(u**2 + v**2 + w**2)
    if np.size(x) == np.size(abs_velocity):
        fig, ax = plt.subplots()
        ax.plot(x, abs_velocity)
        ax.set_xlabel('x')
        ax.set_ylabel('abs_velocity')
        plt.title('Sensor line on x-axis')
        plt.show()
    else:
        print('Size mismatch, x has size', np.size(x), 'and abs_velocity has size', np.size(abs_velocity))

#Unit test of a single eddy which can be manually placed at any position or angle
def main():
    #Generate random angles
    theta_x, theta_y, theta_z = random_angles()
    theta_x, theta_y, theta_z = 0, m.pi/2, 0
    #Rotate the sensor line
    pos_vector_rotated = positionvectorrotated(theta_x, theta_y, theta_z)
    #Displace the sensor line
    pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, np.array([50, 0, 0]))
    #Calculate the velocity components on the rotated and displaced sensor line
    u, v, w = velocityline(pos_vector_displaced)
    #Generate the an untransformed sensor line on the x-axis
    x0 = positionvectorxaxis()
    #Plot the transformed velocities on the untransformed sensor line on the x-axis
    xaxisplotter(x0, u, v, w)

#main()

def manyeddies(N_eddies):
    #Generate random angles
    theta_x, theta_y, theta_z = random_angles()
    #Rotate the sensor line
    pos_vector_rotated = positionvectorrotated(theta_x, theta_y, theta_z)
    #Displace the sensor line
    pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, np.array([0, 0, 0]))
    #Calculate the velocity components on the rotated and displaced sensor line
    u, v, w = velocityline(pos_vector_displaced)
    for N in range(N_eddies):
        #Generate random angles
        theta_x, theta_y, theta_z = random_angles()
        #Rotate the sensor line
        pos_vector_rotated = positionvectorrotated(theta_x, theta_y, theta_z)
        #Generate random position
        a = random_position()
        #print(a)
        #Displace the sensor line
        pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, a)
        #Calculate the velocity components on the rotated and displaced sensor line
        u_new, v_new, w_new = velocityline(pos_vector_displaced)
        u = np.add(u, u_new)
        v = np.add(v, v_new)
        w = np.add(w, w_new)
        print(u_new)
    #Generate the an untransformed sensor line on the x-axis
    x0 = positionvectorxaxis()
    #Plot the transformed velocities on the untransformed sensor line on the x-axis
    xaxisplotter(x0, u, v, w)

manyeddies(100000)

    






