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

x, y, z = vectorline()
theta_x, theta_y, theta_z = random_angles()
x_rot, y_rot, z_rot = rotatedline(*vectorline(), theta_x, theta_y, theta_z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the rotated vector line
ax.plot(x, y, z, label="Original 3D Vector Line")
ax.plot(x_rot, y_rot, z_rot, label="Rotated 3D Vector Line")

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rotated 3D Vector Line')
# Set equal limits for all axes to center the plot
max_range = 10  # Define the range for the axes limits
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])
# Show legend and plot
ax.legend()
plt.show()