import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m

import numpy as np

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

# Apply 3D rotation
def rotate_meshgrid(x, y, z, theta_x, theta_y, theta_z):
    # Flatten the meshgrid to apply the rotation easily
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    
    # Construct the combined rotation matrix
    rotation_matrix = rotation_matrix_z(theta_z) @ rotation_matrix_y(theta_y) @ rotation_matrix_x(theta_x)
    
    # Apply the rotation
    rotated_points = rotation_matrix @ points
    
    # Reshape back to the original meshgrid shape
    x_rot = rotated_points[0, :].reshape(x.shape)
    y_rot = rotated_points[1, :].reshape(y.shape)
    z_rot = rotated_points[2, :].reshape(z.shape)
    
    return x_rot, y_rot, z_rot

# Example usage
x, y, z = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))

# Rotate by 45 degrees (pi/4 radians) around x, y, and z axes
theta_x = np.pi / 2
theta_y = 0
theta_z = 0

x_rot, y_rot, z_rot = rotate_meshgrid(x, y, z, theta_x, theta_y, theta_z)

def townsend_rotate_scatter(N_eddies):
    #characteristic eddy length scale
    L = 1
    theta_x = np.pi / 4
    theta_y = np.pi / 4
    theta_z = np.pi / 4
    #create grid array in 3D space
    N = 100
    scale = 10
    x, y, z = np.meshgrid(np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N))
    x, y, z = rotate_meshgrid(x, y, z, theta_x, theta_y, theta_z)
    #calculate the velocities following the Townsend model
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    w = np.zeros_like(u)
    #Find random locations for the eddies
    eddy_locations = [[0,0,0]]
    for i in range(N_eddies-1):
        x1 = random.uniform(-1,1)*scale*L
        y1 = random.uniform(-1,1)*scale*L
        z1 = random.uniform(-1,1)*scale*L
        eddy_locations.append([x1, y1, z1])
        #calculate the velocities following the Townsend model for offset eddies
        u += -(y-y1) * np.exp(-((x-x1)**2 + (y-y1)**2 + (z-z1)**2) / (2*L**2))
        v += (x-x1) * np.exp(-((x-x1)**2 + (y-y1)**2 + (z-z1)**2) / (2*L**2))
        w += np.zeros_like(u)

    mag_v = np.sqrt(u**2 + v**2 + w**2)
    ref_v = 0.2
    # Create a mask for the surface of constant velocity
    mask = np.abs(mag_v - ref_v) < 0.05  # Threshold for surface

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the constant velocity surface
    ax.scatter(x[mask], y[mask], z[mask], c='b', s=0.04, alpha=0.5)

    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Surface of Constant Velocity in 3D Space (Townsend Eddy Model)')
    print(eddy_locations)
    arrow_length = 5 * L  # Define the arrow length (based on eddy scale)
    for loc in eddy_locations:
        x1, y1, z1 = loc
        ax.quiver(x1, y1, z1, 0, 0, arrow_length, color='r', linewidth=2)  # Arrow aligned along the z-axis
    
    # Set limits and labels
    M = 1
    ax.set_xlim(-M*scale*L, M*scale*L)
    ax.set_ylim(-M*scale*L, M*scale*L)
    ax.set_zlim(-M*scale*L, M*scale*L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Constant velocity surface for Townsend model eddy (Velocity = {ref_v})')
    plt.show()    
    return x1, y1, z1

townsend_rotate_scatter(2)
