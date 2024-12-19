import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
import timeit
import plotly.graph_objects as go
import kinematics as k


def xaxisplotter(pos_vector, u):
    x = pos_vector[0]
    if np.size(x) == np.size(u):
        fig, ax = plt.subplots()
        ax.plot(x, u)
        ax.set_xlabel('x/L')
        ax.set_ylabel('Velocity Component')
        plt.title('Sensor line on x-axis')
        plt.show()
    else:
        print('Size mismatch, x has size', np.size(x), 'and abs_velocity has size', np.size(u))

def eddyplotter(A_list, axis):
    x_list = np.empty(len(A_list))
    y_list = np.empty(len(A_list))
    z_list = np.empty(len(A_list))
    for i in range(len(A_list)):
        x_list[i] = (A_list[i][0])
        y_list[i] = (A_list[i][1])
        z_list[i] = (A_list[i][2])
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

def plot3d(N_E):
    L = k.constants()[0]
    tol = k.constants()[1]
    x_boundary = 5
    y_boundary = 5
    z_boundary = 5
    Nx = 50
    Nyz = 50

    x, y, z = np.meshgrid(np.linspace(-x_boundary*L, x_boundary*L, Nx), 
                          np.linspace(-y_boundary*L, y_boundary*L, Nyz), 
                          np.linspace(-z_boundary*L, z_boundary*L, Nyz))
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    w = np.zeros_like(z)
    # Flatten the meshgrid arrays
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    for i in range(N_E):
        a = np.array([0, 0, 0])
        theta_x, theta_y, theta_z = np.array([0, 0, 0])
        # Combine rotation matrices
        R = k.rotation_matrix_z(theta_z) @ k.rotation_matrix_y(theta_y) @ k.rotation_matrix_x(theta_x)
        
        # Apply the combined rotation matrix
        rotated_points = R @ points
        
        # Translate the rotated points by vector a
        translated_points = rotated_points + np.array(a).reshape(3, 1)
        
        # Reshape back to the original shape
        x_r = translated_points[0].reshape(x.shape)
        y_r = translated_points[1].reshape(y.shape)
        z_r = translated_points[2].reshape(z.shape)
        
        u_0 = -y_r * np.exp(-(x_r**2 + y_r**2 + z_r**2) / (2*L**2))
        v_0 = x_r * np.exp(-(x_r**2 + y_r**2 + z_r**2) / (2*L**2))
        w_0 = np.zeros_like(u)
        u = np.add(u, u_0)
        v = np.add(v, v_0)
        w = np.add(w, w_0)
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

plot3d(1)
def plot3d_change_order(N_E, theta=0):
    L = k.constants()[0]
    tol = k.constants()[1]
    x_boundary = 10
    y_boundary = k.constants()[3]
    z_boundary = k.constants()[4]
    Nx = k.constants()[7]
    Nyz = k.constants()[8]

    x, y, z = np.meshgrid(np.linspace(-x_boundary*L, x_boundary*L, Nx), 
                          np.linspace(-y_boundary*L, y_boundary*L, Nyz), 
                          np.linspace(-z_boundary*L, z_boundary*L, Nyz))
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    w = np.zeros_like(z)
    # Flatten the meshgrid arrays
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    for i in range(N_E):
        a = k.random_position()
        if theta == 0:
            theta_x, theta_y, theta_z = k.random_angles()
        else:
            theta_x, theta_y, theta_z = theta, theta, theta
        # Combine rotation matrices
        R = k.rotation_matrix_z(theta_z) @ k.rotation_matrix_y(theta_y) @ k.rotation_matrix_x(theta_x)
        
        
        # Translate the rotated points by vector a
        translated_points = points + np.array(a).reshape(3, 1)
        
        # Reshape back to the original shape
        x_r = translated_points[0].reshape(x.shape)
        y_r = translated_points[1].reshape(y.shape)
        z_r = translated_points[2].reshape(z.shape)
        
        u_0 = -y_r * np.exp(-(x_r**2 + y_r**2 + z_r**2) / (2*L**2))
        v_0 = x_r * np.exp(-(x_r**2 + y_r**2 + z_r**2) / (2*L**2))
        w_0 = np.zeros_like(u)
        velocity_vector = np.vstack((u_0.ravel(), v_0.ravel(), w_0.ravel()))
        velocity_vector_rotated = R @ velocity_vector
        u_rotated = velocity_vector_rotated[0].reshape(x.shape)
        v_rotated = velocity_vector_rotated[1].reshape(y.shape)
        w_rotated = velocity_vector_rotated[2].reshape(z.shape)
        u = np.add(u, u_rotated)
        v = np.add(v, v_rotated)
        w = np.add(w, w_rotated)
    mag_v = np.sqrt(u**2 + v**2 + w**2)
    # Create isosurface plot
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

def rotation_check(axis_rotated_translated, N_E):
    # Generate random angles
    
    # Create a 3D vector line
    x, y, z = axis_rotated_translated[:,0:2]
    # Rotate the vector line
    # Plot the rotated vector line
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label="Original 3D Vector Line")
    for i in range(N_E):
        print(i)
        x_rot, y_rot, z_rot = axis_rotated_translated[:,3*i:3*i+2]
        ax.plot(x_rot, y_rot, z_rot, label="Rotated 3D Vector Line")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotated 3D Vector Line')
    ax.legend()
    plt.show()

def isotropic_turbulence_plot(r, f, g, max_index):
    r = r[0:max_index]
    f = f[0:max_index]
    g = g[0:max_index]
    fig, ax = plt.subplots()
    ax.plot(r, f, label='f')
    ax.plot(r, g, label='g')
    ax.set_xlabel('r/L')
    ax.set_ylabel('Correlation function')
    plt.legend()
    plt.title('Correlation Functions')

def energy_spectrum_plot(E_k, k_array):
    fig, ax = plt.subplots()
    ax.plot(k_array, E_k)
    ax.set_xlabel('k')
    ax.set_ylabel('E(k)')
    plt.title('Energy Spectrum')

def structure_plotter(r, f, max_index):
    f = f[0:max_index]
    r = r[0:max_index]
    fig, ax = plt.subplots()
    ax.plot(r, f)
    ax.set_xlabel('r/L')
    ax.set_ylabel('f')
    plt.title('Structure function')

def theoretical_f(r, f, max_index):
    f = f[0:max_index]
    L_e = k.constants()[0]
    f_theoretical = np.exp(-r**2/L_e**2)
    fig, ax = plt.subplots()
    ax.plot(r, f, label='f', c='r')
    ax.plot(r, f_theoretical, label='Theoretical f', c = 'b', linestyle='--')
    ax.set_xlabel('r/L_e')
    ax.set_ylabel('f')
    plt.legend()
    plt.title('Structure function')

def theoretical_g(r, g, max_index):
    g = g[0:max_index]
    L_e = k.constants()[0]
    g_theoretical = (1 - r**2/L_e**2) * np.exp(-r**2/L_e**2)
    fig, ax = plt.subplots()
    ax.plot(r, g, label='g', c='r')
    ax.plot(r, g_theoretical, label='Theoretical g', c = 'b', linestyle='--')
    ax.set_xlabel('r/L_e')
    ax.set_ylabel('g')
    plt.legend()
    plt.title('Structure function')
