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

def velocity_field():
    #characteristic eddy length scale
    L = 1

    #create grid array in 3D space
    N = 10
    scale = 2
    x, y, z = np.meshgrid(np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N))
    #calculate the velocities following the Townsend model
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    w = np.zeros_like(u)
    return u,v,w

def rotated_velocity_field(theta_x, theta_y, theta_z):
    u,v,w = velocity_field()
    rotation_matrix = total_matrix(theta_x, theta_y, theta_z)
    v_vector = np.vstack([u.ravel(), v.ravel(), w.ravel()])
    rotated_points = rotation_matrix @ v_vector
    u_rot = rotated_points[0, :].reshape(u.shape)
    v_rot = rotated_points[1, :].reshape(v.shape)
    w_rot = rotated_points[2, :].reshape(w.shape)
    return u_rot, v_rot, w_rot

#print("This is x rotation", rotated_velocity_field(m.pi/2, 0, 0))
#print("This is y rotation", rotated_velocity_field(0, m.pi/2, 0))

u_rot1, v_rot1, w_rot1 = rotated_velocity_field(m.pi/2, 0, 0)
u_rot2, v_rot2, w_rot2 = rotated_velocity_field(0, m.pi/2, 0)
u1 = u_rot1.ravel()
u2 = u_rot2.ravel()
print("This is u1", u1)
print("This is u2", u2)

# Plotting
def plotter(u, v, w):
    L = 1
    #create grid array in 3D space
    N = 10
    scale = 2
    mag_v = np.sqrt(u**2 + v**2 + w**2)
    ref_v = 0.2
    # Create a mask for the surface of constant velocity
    mask = np.abs(mag_v - ref_v) < 0.05  # Threshold for surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N))
    # Plot the constant velocity surface
    ax.scatter(x[mask], y[mask], z[mask], c='b', s=0.04, alpha=0.5)

    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Surface of Constant Velocity in 3D Space (Townsend Eddy Model)')
    """
    print(eddy_locations)
    arrow_length = 5 * L  # Define the arrow length (based on eddy scale)
    for loc in eddy_locations:
        x1, y1, z1 = loc
        ax.quiver(x1, y1, z1, 0, 0, arrow_length, color='r', linewidth=2)  # Arrow aligned along the z-axis
    """
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

plotter(u_rot1, v_rot1, w_rot1)
plotter(u_rot2, v_rot2, w_rot2)