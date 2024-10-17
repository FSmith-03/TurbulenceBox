#We have a sensor line on the x axis from -40,000L to 40,000L

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
import timeit
import plotly.graph_objects as go

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
Nyz = int(Nyz)
# Rotation matrix for rotation around x-axis

def spacefiller(x_boundary, y_boundary, z_boundary):
    N = 8*x_boundary*y_boundary*z_boundary
    N = int(N)
    return N

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
    a = a[:, np.newaxis]
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

#Position vector rotated, takes input of the three angles
def positionvectorrotated2(theta_x, theta_y, theta_z, a):
    R = total_matrix(theta_x, theta_y, theta_z)
    pos_vector = positionvectorxaxis()
    print(a)
    translated_vector = pos_vector - a
    pos_vector_rotated = R @ translated_vector
    return pos_vector_rotated

#Vector displacement, takes input of a position vector and a displacement vector
def positionvectordisplaced(pos_vector, a):
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

#Enhanced velocity line function that does not calculate the velocity at every point but only the local coordinates
def velocitylineenhanced(pos_vector):
    x = pos_vector[0]
    y = pos_vector[1]
    z = pos_vector[2]
    
    # Create a boolean mask where |x| < 2.95L
    mask = np.abs(x) < 2.95
    
    # Initialize u, v, and w as zero arrays
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    w = np.zeros_like(x)
    
    # Apply calculations only where the mask is True
    factor = np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    u[mask] = -y[mask] * factor[mask]
    v[mask] = x[mask] * factor[mask]
    # w remains zeros as per the original logic
    
    return u, v, w
#Plotter for the sensor line on the x-axis, takes input of the position vector and the velocity components
def xaxisplotter(pos_vector, u):
    x = pos_vector[0]
    if np.size(x) == np.size(u):
        fig, ax = plt.subplots()
        ax.plot(x, u)
        ax.set_xlabel('x/L')
        ax.set_ylabel('abs_velocity')
        plt.title('Sensor line on x-axis')
        plt.show()
    else:
        print('Size mismatch, x has size', np.size(x), 'and abs_velocity has size', np.size(u))

#Plot eddy positions
def eddyplotter(A_list, axis):
    x_list = A_list[:, 0]
    y_list = A_list[:, 1]
    z_list = A_list[:, 2]
    r_list = np.sqrt(y_list**2 + z_list**2)
    fig, ax = plt.subplots()
    if axis == 'x':
        plotted_list = y_list
    elif axis == 'y':
        plotted_list = z_list
    elif axis == 'r':
        plotted_list = r_list
    ax.scatter(x_list, plotted_list, c='r', s=0.4)
    ax.set_xlabel('x/L')
    ax.set_ylabel('r/L')
    plt.title('Eddy Positions')
    plt.show()

def plot3d(a, theta_x, theta_y, theta_z):
    x_boundary = 5
    y_boundary = 5
    z_boundary = 5
    Nx = 100
    Nyz = 100

    x, y, z = np.meshgrid(np.linspace(-x_boundary*L, x_boundary*L, Nx), 
                          np.linspace(-y_boundary*L, y_boundary*L, Nyz), 
                          np.linspace(-z_boundary*L, z_boundary*L, Nyz))
    
    # Flatten the meshgrid arrays
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    
    # Combine rotation matrices
    R = rotation_matrix_z(theta_z) @ rotation_matrix_y(theta_y) @ rotation_matrix_x(theta_x)
    
    # Apply the combined rotation matrix
    rotated_points = R @ points
    
    # Translate the rotated points by vector a
    translated_points = rotated_points + np.array(a).reshape(3, 1)
    
    # Reshape back to the original shape
    x_r = translated_points[0].reshape(x.shape)
    y_r = translated_points[1].reshape(y.shape)
    z_r = translated_points[2].reshape(z.shape)
    
    u = -y_r * np.exp(-(x_r**2 + y_r**2 + z_r**2) / (2*L**2))
    v = x_r * np.exp(-(x_r**2 + y_r**2 + z_r**2) / (2*L**2))
    w = np.zeros_like(u)
    mag_v = np.sqrt(u**2 + v**2 + w**2)
    # Create isosurface plot
    x_boundary = 100
    y_boundary = 2.95
    z_boundary = 2.95
    Nxf = 2*x_boundary/tol
    Nyz = 2*y_boundary/tol
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=mag_v.flatten(),
        isomin=mag_v.min(),
        isomax=mag_v.max(),
        surface_count=10,  # Number of isosurfaces
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    
    fig.show()

a = [1, 1, 1]
#plot3d(a, m.pi/4, m.pi/4, 0)

#Plot the structure function
def structure_function(pos_vector, u):
    x = pos_vector[0]
    r = np.linspace(tol, (Nx-1)*tol, Nx-1)
    S = np.zeros_like(r)
    total = 0
    for n in range(Nx-1):
        index = int(round((Nx-1)/(n+1)))
        for i in range(index):
            total += (u[i]-u[i+(n+1)])**2
        S[n] = total/(Nx-1)
        total = 0
    fig, ax = plt.subplots()
    print(r)
    ax.plot(r, S)
    ax.set_xlabel('r')
    ax.set_ylabel('S')
    plt.title('Structure function')
    plt.show()

#Unit test of a single eddy which can be manually placed at any position or angle
def main():
    #Generate random angles
    theta_x, theta_y, theta_z = random_angles()
    theta_x, theta_y, theta_z = 0, 0, 0
    #Rotate the sensor line
    pos_vector_rotated = positionvectorrotated(theta_x, theta_y, theta_z)
    #Displace the sensor line
    pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, np.array([0, 0, 0]))
    #Calculate the velocity components on the rotated and displaced sensor line
    u, v, w = velocityline(pos_vector_displaced)
    #Generate the an untransformed sensor line on the x-axis
    x0 = positionvectorxaxis()
    #Plot the transformed velocities on the untransformed sensor line on the x-axis
    xaxisplotter(x0, u, v, w)

#main()

def main2():
    #Generate random angles
    theta_x, theta_y, theta_z = random_angles()
    a = random_position()
    #Rotate the sensor line
    pos_vector_rotated = positionvectorrotated2(theta_x, theta_y, theta_z, a)
    #Displace the sensor line
    pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, a)
    #Calculate the velocity components on the rotated and displaced sensor line
    u, v, w = velocityline(pos_vector_rotated)
    #Generate the an untransformed sensor line on the x-axis
    x0 = positionvectorxaxis()
    #Plot the transformed velocities on the untransformed sensor line on the x-axis
    xaxisplotter(x0, u, v, w)

#main2()
def manyeddies(N_eddies):
    #Generate random angles
    theta_x, theta_y, theta_z = random_angles()
    #Rotate the sensor line
    pos_vector_rotated = positionvectorrotated(theta_x, theta_y, theta_z)
    #Generate random position
    A_list = []
    a = random_position()
    A_list.append(a)
    #Displace the sensor line
    pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, a)
    #Calculate the velocity components on the rotated and displaced sensor line
    u, v, w = velocityline(pos_vector_displaced)
    for N in range(N_eddies):
        #Generate random angles
        theta_x, theta_y, theta_z = random_angles()
        theta_x, theta_y, theta_z = 0, 0, 0
        #Rotate the sensor line
        pos_vector_rotated = positionvectorrotated(theta_x, theta_y, theta_z)
        #Generate random position
        a = random_position()
        A_list.append(a)
        #Displace the sensor line
        pos_vector_displaced = positionvectordisplaced(pos_vector_rotated, a)
        #Calculate the velocity components on the rotated and displaced sensor line
        u_new, v_new, w_new = velocitylineenhanced(pos_vector_displaced)
        u = np.add(u, u_new)
        v = np.add(v, v_new)
        w = np.add(w, w_new)
    #Generate the an untransformed sensor line on the x-axis
    x0 = positionvectorxaxis()
    #Plot the transformed velocities on the untransformed sensor line on the x-axis
    xaxisplotter(x0, u)
    #Plot the eddy positions
    A_list = np.array(A_list)
    structure_function(x0, u)
    #Calculate the structure function
    #structure_function(x0, u, v, w)

manyeddies(spacefiller(x_boundary, y_boundary, z_boundary))

#time_taken = timeit.timeit(lambda: manyeddies(100000), number=1)
#print(f"Time taken: {time_taken} seconds")   






