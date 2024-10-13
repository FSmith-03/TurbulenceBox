import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m

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

def total_matrix(theta_x, theta_y, theta_z):
    return rotation_matrix_z(theta_z) @ rotation_matrix_y(theta_y) @ rotation_matrix_x(theta_x)

def random_angles():
   theta = random.uniform(0, 2*m.pi)
   phi = random.uniform(0, 2*m.pi)
   psi = random.uniform(0, 2*m.pi)
   return theta, phi, psi

def vectorline():
    # Define the vector line: start point (x0, y0, z0) and direction (dx, dy, dz)
    start_point = np.array([0, 0, 0])
    direction = np.array([0, 0, 1])

    # Create parameter t
    t = np.linspace(-3, 3, 100)

    # Define the 3D line based on the vector equation: r(t) = r0 + t * direction
    x = start_point[0] + t * direction[0]
    y = start_point[1] + t * direction[1]
    z = start_point[2] + t * direction[2]
    return x, y, z

def rotatedline(x, y, z, theta_x, theta_y, theta_z):
    # Rotate the vector line
    rotation_matrix = total_matrix(theta_x, theta_y, theta_z)
    v_vector = np.vstack([x, y, z])
    rotated_points = rotation_matrix @ v_vector
    x_rot = rotated_points[0, :]
    y_rot = rotated_points[1, :]
    z_rot = rotated_points[2, :]
    return x_rot, y_rot, z_rot

def coordinates(theta_x, theta_y, theta_z):
    #characteristic eddy length scale
    L = 1

    #create grid array in 3D space
    N = 100
    scale = 10
    x = np.linspace(-scale*L, scale*L, N)
    y = np.linspace(-scale*L, scale*L, N)
    z = np.linspace(-scale*L, scale*L, N)
    position_vector = np.vstack([x, y, z])
    rotation_matrix = total_matrix(theta_x, theta_y, theta_z)
    rotated_points = rotation_matrix @ position_vector
    x_r = rotated_points[0, :]
    y_r = rotated_points[1, :]
    z_r = rotated_points[2, :]
    return x_r, y_r, z_r
def velocity_field(x, y, z):
    #characteristic eddy length scale
    L = 1
    #create grid array in 3D space
    N = 100
    scale = 10
    x, y, z = np.meshgrid(x, y, z)
    #calculate the velocities following the Townsend model
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    w = np.zeros_like(u)
    return u, v, w

def rotated_velocity_field(x_range, y_range, z_range):
    # Characteristic eddy length scale
    L = 1
    theta_x, theta_y, theta_z = m.pi/2, 0, 0
    N = 100  # Number of grid points in each dimension

    # Create 3D grid in space
    x, y, z = np.meshgrid(np.linspace(x_range[0], x_range[1], N),
                          np.linspace(y_range[0], y_range[1], N),
                          np.linspace(z_range[0], z_range[1], N))
    
    # Calculate the velocities following the Townsend model
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2 * L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2 * L**2))
    w = np.zeros_like(u)
    
    # Flatten and stack velocity components for rotation
    v_vector = np.vstack([u.ravel(), v.ravel(), w.ravel()])
    
    # Apply rotations in sequence
    v_vector_rot = rotation_matrix_x(theta_x) @ rotation_matrix_y(theta_y) @ rotation_matrix_z(theta_z) @ v_vector
    
    # Reshape the rotated velocity components back to the original grid shape
    u_rot, v_rot, w_rot = v_vector_rot[0, :].reshape(N, N, N), \
                          v_vector_rot[1, :].reshape(N, N, N), \
                          v_vector_rot[2, :].reshape(N, N, N)
    
    return u_rot, v_rot, w_rot

def velocity_field_plotter():
    #characteristic eddy length scale
    L = 1

    #create grid array in 3D space
    N = 100
    scale = 10
    theta_x, theta_y, theta_z = m.pi/2, 0, 0
    x, y, z = np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N)
    u_r, v_r, w_r = rotated_velocity_field(x, y, z)
    u, v, w = velocity_field(x, y, z)
    x, y, z = np.meshgrid(x, y, z)
    mag_v_r = np.sqrt(u_r**2 + v_r**2 + w_r**2)
    mag_v = np.sqrt(u**2 + v**2 + w**2)
    ref_v = 0.2
    # Create a mask for the surface of constant velocity
    mask_r = np.abs(mag_v_r - ref_v) < 0.05  # Threshold for surface
    mask = np.abs(mag_v - ref_v) < 0.05  # Threshold for surface
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the constant velocity surface
    ax.scatter(x[mask_r], y[mask_r], z[mask_r], c='b', s=0.04, alpha=0.5)
    ax.scatter(x[mask], y[mask], z[mask], c='r', s=0.04, alpha=0.5)
    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Surface of Constant Velocity in 3D Space (Townsend Eddy Model)')
    
    # Set limits and labels
    M = 0.5
    ax.set_xlim(-M*scale*L, M*scale*L)
    ax.set_ylim(-M*scale*L, M*scale*L)
    ax.set_zlim(-M*scale*L, M*scale*L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Constant velocity surface for Townsend model eddy (Velocity = {ref_v})')
    plt.show()  

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the constant velocity surface
    ax.scatter(x[mask], y[mask], z[mask], c='r', s=0.04, alpha=0.5)
    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Surface of Constant Velocity in 3D Space (Townsend Eddy Model)')
    
    # Set limits and labels
    M = 0.5
    ax.set_xlim(-M*scale*L, M*scale*L)
    ax.set_ylim(-M*scale*L, M*scale*L)
    ax.set_zlim(-M*scale*L, M*scale*L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Constant velocity surface for Townsend model eddy (Velocity = {ref_v})')
    plt.show()
velocity_field_plotter()