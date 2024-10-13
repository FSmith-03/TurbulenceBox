import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
def townsend_plot():
    #characteristic eddy length scale
    L = 1

    #create grid array in 3D space
    N = 100
    scale = 10
    x, y, z = np.meshgrid(np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N))
    #calculate the velocities following the Townsend model
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    w = np.zeros_like(u)
    mag_v = np.sqrt(u**2 + v**2 + w**2)
    #plot the velocities
    ref_v = 0.2
    # Generate the isosurface for the constant velocity
    verts, faces, _, _ = measure.marching_cubes(mag_v, ref_v, spacing=(1, 1, 1))

    # Plot the constant velocity surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    verts = verts - np.array([N/2, N/2, N/2])  # Center the mesh at the origin
    print(verts[faces])
    # Create a collection of triangles for plotting
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    mesh.set_facecolor([0, 1, 1])  # Set color to cyan
    ax.add_collection3d(mesh)

    # Set limits and labels
    M = 10
    ax.set_xlim(-M*scale*L, M*scale*L)
    ax.set_ylim(-M*scale*L, M*scale*L)
    ax.set_zlim(-M*scale*L, M*scale*L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Constant velocity surface for Townsend model eddy (Velocity = {ref_v})')
    #Add a slider to change the axes limits
    # Function to update the plot when the slider is changed
    def update(val):
        M = slider.val
        ax.set_xlim(-M*scale*L, M*scale*L)
        ax.set_ylim(-M*scale*L, M*scale*L)
        ax.set_zlim(-M*scale*L, M*scale*L)
        fig.canvas.draw_idle()

    # Create a slider for adjusting the value of M
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'M', 1, 20, valinit=M, valstep=1)

    # Update the plot when the slider value changes
    slider.on_changed(update)
    plt.show()
    return 

#townsend_plot()

def townsend_multiple(N_eddies):
    #characteristic eddy length scale
    L = 1

    #create grid array in 3D space
    N = 100
    scale = 10
    x, y, z = np.meshgrid(np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N))
    #calculate the velocities following the Townsend model
    u = -y * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    v = x * np.exp(-(x**2 + y**2 + z**2) / (2*L**2))
    w = np.zeros_like(u)
    #Find random locations for the eddies
    eddy_locations = []
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
        #plot the velocities
    ref_v = 0.2
    # Generate the isosurface for the constant velocity
    verts, faces, _, _ = measure.marching_cubes(mag_v, ref_v, spacing=(1, 1, 1))

    # Plot the constant velocity surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    verts = verts - np.array([N/2, N/2, N/2])  # Center the mesh at the origin
    # Create a collection of triangles for plotting
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    mesh.set_facecolor([0, 1, 1])  # Set color to cyan
    ax.add_collection3d(mesh)
    # Plot arrows through the axis of each individual eddy

    arrow_length = 3 * L  # Define the arrow length (based on eddy scale)
    for loc in eddy_locations:
        x1, y1, z1 = loc
        ax.quiver(x1, y1, z1, 0, 0, arrow_length, color='r', linewidth=2)  # Arrow aligned along the z-axis
    
    # Set limits and labels
    M = 10
    ax.set_xlim(-M*scale*L, M*scale*L)
    ax.set_ylim(-M*scale*L, M*scale*L)
    ax.set_zlim(-M*scale*L, M*scale*L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Constant velocity surface for Townsend model eddy (Velocity = {ref_v})')
    plt.show()    
    return x1, y1, z1

def townsend_multiple_scatter(N_eddies):
    #characteristic eddy length scale
    L = 1

    #create grid array in 3D space
    N = 100
    scale = 10
    x, y, z = np.meshgrid(np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N), np.linspace(-scale*L, scale*L, N))
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


print(townsend_multiple_scatter(10))