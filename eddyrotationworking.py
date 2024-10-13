import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
#constants
L = 1
N = 100
scale = 10
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

def positionvector():
    x = np.linspace(-scale*L, scale*L, N)
    y = np.linspace(-scale*L, scale*L, N)
    z = np.linspace(-scale*L, scale*L, N)
    return x, y, z

def velocitymesh(x, y, z):
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
    u = -y_mesh * np.exp(-(x_mesh**2 + y_mesh**2 + z_mesh**2) / (2*L**2))
    v = x_mesh * np.exp(-(x_mesh**2 + y_mesh**2 + z_mesh**2) / (2*L**2))
    w = np.zeros_like(u)
    return u, v, w

def velocity_vectors(u, v, w):
    # Combine u, v, and w into a single 3D vector at each point
    u_1D = u.reshape(N**3)
    v_1D = v.reshape(N**3)
    w_1D = w.reshape(N**3)
    velocity_vector = np.vstack((u_1D, v_1D, w_1D))
    return velocity_vector

def rotated_velocity_vectors(velocity_vector, theta_x, theta_y, theta_z):
    # Rotate the velocity vectors
    rotation_matrix = total_matrix(theta_x, theta_y, theta_z)
    
    # Apply rotation matrix to each vector, velocity_vector is (3, N)
    rotated_velocity = rotation_matrix @ velocity_vector
    
    # Reshape the rotated velocity back to (3, N)
    return rotated_velocity

def reshape_velocity_to_meshgrid(rotated_velocity, x_mesh, y_mesh, z_mesh):
    # Reshape rotated velocity components to match the shape of the meshgrid
    u_rot = rotated_velocity[0, :].reshape(x_mesh.shape)
    v_rot = rotated_velocity[1, :].reshape(x_mesh.shape)
    w_rot = rotated_velocity[2, :].reshape(x_mesh.shape)
    return u_rot, v_rot, w_rot

def plot_velocity_field(x_mesh, y_mesh, z_mesh, u, v, w, u_rot, v_rot, w_rot, ref_v, original=False):

    # Calculate the magnitude of the rotated velocity field
    mag_v = np.sqrt(u_rot**2 + v_rot**2 + w_rot**2)
    
    # Create a mask for the surface of constant velocity
    mask = np.abs(mag_v - ref_v) < 0.1  # Threshold for surface
    
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if original:
        mag_v_original = np.sqrt(u**2 + v**2 + w**2)
        mask_original = np.abs(mag_v_original - ref_v) < 0.1
        ax.scatter(x_mesh[mask_original], y_mesh[mask_original], z_mesh[mask_original], c='r', s=0.04, alpha=0.5)
    # Plot the constant velocity surface using 3D mask (this ensures mask has same shape as x_mesh, y_mesh, z_mesh)
    ax.scatter(x_mesh[mask], y_mesh[mask], z_mesh[mask], c='b', s=0.04, alpha=0.5)
    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title(f'Constant velocity surface for Townsend model eddy (Velocity = {ref_v})')
    plt.show()

def plot_velocity_comparison(u, u_rot, v, v_rot, w, w_rot):
    u_1D = u.reshape(N**3)
    v_1D = v.reshape(N**3)
    w_1D = w.reshape(N**3)
    u_rot_1D = u_rot.reshape(N**3)
    v_rot_1D = v_rot.reshape(N**3)
    w_rot_1D = w_rot.reshape(N**3)
    x_axis = np.arange(N**3)
    plt.figure(figsize=(10, 8))
    plt.plot(x_axis, u_1D, label='Original u')
    plt.plot(x_axis, u_rot_1D, label='Rotated u')
    plt.xlabel('Index')
    plt.ylabel('Velocity')
    plt.title('Comparison of original and rotated velocity components in u direction')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.plot(x_axis, v_1D, label='Original v')
    plt.plot(x_axis, v_rot_1D, label='Rotated v')
    plt.xlabel('Index')
    plt.ylabel('Velocity')
    plt.title('Comparison of original and rotated velocity components in v direction')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.plot(x_axis, w_1D, label='Original w')
    plt.plot(x_axis, w_rot_1D, label='Rotated w')
    plt.xlabel('Index')
    plt.ylabel('Velocity')
    plt.title('Comparison of original and rotated velocity components in w direction')
    plt.legend()
    plt.show()

def plot_velocity_magnitude_x_axis(x_mesh, y_mesh, z_mesh, u_rot, v_rot, w_rot):
    """
    Plot the velocity magnitude along the x-axis where y = 0 and z = 0.
    """
    # Find the indices where y = 0 and z = 0
    mask = np.isclose(y_mesh, 0, atol=0.5) & np.isclose(z_mesh, 0, atol=0.5)
    # Extract x coordinates and the corresponding velocity components
    x_along_x_axis = x_mesh[mask]
    u_along_x_axis = u_rot[mask]
    v_along_x_axis = v_rot[mask]
    w_along_x_axis = w_rot[mask]

    # Calculate the velocity magnitude along the x-axis
    velocity_magnitude = np.sqrt(u_along_x_axis**2 + v_along_x_axis**2 + w_along_x_axis**2)
    print(velocity_magnitude)
    # Plot the velocity magnitude along the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(x_along_x_axis, velocity_magnitude, label='Velocity Magnitude')
    plt.xlabel('X axis')
    plt.ylabel('Velocity Magnitude')
    plt.title('Velocity Magnitude along the X-axis (y = 0, z = 0)')
    plt.grid(True)
    plt.legend()
    plt.show()

def translate_velocity_field(x_mesh, y_mesh, z_mesh, a, b, c):
    """
    Translates the velocity field to be centered on position (a, b, c).
    This function adds (a, b, c) to the x, y, z meshgrid.
    """
    x_trans = x_mesh + a
    y_trans = y_mesh + b
    z_trans = z_mesh + c
    return x_trans, y_trans, z_trans

def main():
    # Generate the position vector
    x, y, z = positionvector()
    
    # Generate the velocity field
    u, v, w = velocitymesh(x, y, z)
    
    # Combine u, v, and w into a single 3D vector at each point
    velocity_vector = velocity_vectors(u, v, w)
    
    # Generate random angles for rotation
    theta_x, theta_y, theta_z = m.pi/2, 0, 0
    
    # Rotate the velocity vectors
    rotated_velocity = rotated_velocity_vectors(velocity_vector, theta_x, theta_y, theta_z)
    
    # Reshape the rotated velocity components back to the original grid shape
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
    x_trans, y_trans, z_trans = translate_velocity_field(x_mesh, y_mesh, z_mesh, 0, 0, 0)
    u_rot, v_rot, w_rot = reshape_velocity_to_meshgrid(rotated_velocity, x_mesh, y_mesh, z_mesh)
    
    # Plot the rotated velocity field
    ref_v = 0.2
    #plot_velocity_field(x_trans, y_trans, z_trans, u, v, w, u_rot, v_rot, w_rot, ref_v, True)
    plot_velocity_comparison(u, u_rot, v, v_rot, w, w_rot)
    #plot_velocity_magnitude_x_axis(x_mesh, y_mesh, z_mesh, u_rot, v_rot, w_rot)
main()