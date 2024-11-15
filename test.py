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
import plotting as p


def two_eddy_test():
    N_E = 3
    print("Number of eddies: ", N_E)
    Nx = k.constants()[7]
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    xaxis = k.positionvectorxaxis()
    a_list = []
    for i in range(N_E):
        print("Eddy number: ", i)
        thetax, thetay, thetaz = k.random_angles()
        if i == 0:
            thetax = 0
            thetay = 0
            thetaz = 0
        if i == 1:
            thetax = m.pi/4
            thetay = 0
            thetaz = 0
        if i == 2:
            thetax = 0
            thetay = m.pi/4
            thetaz = 0
        R = k.total_matrix(thetax, thetay, thetaz)
        a = k.random_position(online=True)
        a_list.append(a.reshape(3))
        print("a: ", a[0])
        #print("a", a[:, np.newaxis])
        xaxis_translated = xaxis - a
        #print(xaxis_translated.shape)
        xaxis_rotated = R @ xaxis_translated
        #print("xaxis_rotated: ", xaxis_rotated)
        #print("xaxis_translated", xaxis_translated)
        u, v, w = k.velocityline(xaxis_rotated)
        #print("u: ", max(u))
        R = k.total_matrix(-thetax, -thetay, -thetaz)
        u_rotated, v_rotated, w_rotated = R @ np.array([u, v, w])
        #print("u_rotated: ", u_rotated)
        #print(u_rotated)
        u_total += u_rotated
        v_total += v_rotated
        w_total += w_rotated
    furthest_eddies = [a for a in a_list if a[0] == max(a_list, key=lambda pos: pos[0])[0]]
    print("Furthest eddy: ", furthest_eddies)
    p.xaxisplotter(xaxis, u_total)
    p.xaxisplotter(xaxis, v_total)
    p.xaxisplotter(xaxis, w_total)
    p.eddyplotter(a_list, 'x')
    return


def main():
    N_E = k.constants()[10]
    Nx = k.constants()[7]
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    a_list = []
    for i in range(N_E):
        xaxis = k.positionvectorxaxis()
        thetax, thetay, thetaz = k.random_angles()
        R = k.total_matrix(thetax, thetay, thetaz)
        a = k.random_position()
        print(a)    
        a_list.append(a)
        xaxis_rotated = R @ xaxis
        xaxis_translated = xaxis_rotated + a
        u, v, w = k.velocitylineenhanced(xaxis_translated)
        u_total += u
        v_total += v
        w_total += w
    p.xaxisplotter(xaxis, u_total)
    p.eddyplotter(a_list, 'x')
    return

#main()     


def diagnostic():
    N_E = k.constants()[10]
    N_E = 8
    Nx = k.constants()[7]
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    xaxis = k.positionvectorxaxis()
    xaxis_translated_list = []
    translation_positions = []
    
    # Initialize the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the original x-axis
    ax.plot(
        xaxis[0], 
        xaxis[1], 
        xaxis[2],
        label='Original Axis',
        color='black',
        linestyle='--'
    )
    
    for i in range(N_E):
        # Get random angles
        thetax, thetay, thetaz = k.random_angles()
        R = k.total_matrix(thetax, thetay, thetaz)
        
        # Generate a random position and rotate the axis
        a = k.random_position(online=False)
        xaxis_rotated = R @ xaxis
        xaxis_translated = xaxis_rotated + np.array(a).reshape(3, 1)
        
        # Append the translated axis and translation vector
        xaxis_translated_list.append(xaxis_translated)
        translation_positions.append(a)

        # Plot the rotated and translated axis in 3D
        ax.plot(
            xaxis_translated[0], 
            xaxis_translated[1], 
            xaxis_translated[2],
            label=f'Axis {i+1}'
        )
        
        # Calculate velocities
        u, v, w = k.velocityline(xaxis_translated)
        u_total += u
        v_total += v
        w_total += w
    
    # Convert translation_positions to a numpy array for easy plotting
    translation_positions = np.array(translation_positions)
    
    # Plot all translation points at once
    ax.scatter(
        translation_positions[:, 0], 
        translation_positions[:, 1], 
        translation_positions[:, 2],
        color='red', 
        marker='o',
        label='Translation Points'
    )

    # Set plot labels and show
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    # Print final list of translated axes
    p.xaxisplotter(xaxis, u_total)
    p.xaxisplotter(xaxis, v_total)
    p.xaxisplotter(xaxis, w_total)
    print(translation_positions)


def diagnostic_change_order():
    N_E = k.constants()[10]
    Nx = k.constants()[7]
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    xaxis = k.positionvectorxaxis()
    xaxis_translated_list = []
    translation_positions = []
    
    for i in range(N_E):
        # Get random angles
        thetax, thetay, thetaz = k.random_angles()
        R = k.total_matrix(thetax, thetay, thetaz)
        
        # Generate a random position and rotate the axis
        a = k.random_position(online=False)
        xaxis_translated = xaxis + np.array(a).reshape(3, 1)
        
        # Append the translated axis and translation vector
        xaxis_translated_list.append(xaxis_translated)
        translation_positions.append(a)

        # Plot the rotated and translated axis in 3D
        
        # Calculate velocities
        u, v, w = k.velocityline(xaxis_translated)
        velocity_vector = np.vstack((u, v, w))
        velocity_vector_rotated = R @ velocity_vector
        u_rotated = velocity_vector_rotated[0]
        v_rotated = velocity_vector_rotated[1]
        w_rotated = velocity_vector_rotated[2]
        u_total += u_rotated
        v_total += v_rotated
        w_total += w_rotated
    
    # Convert translation_positions to a numpy array for easy plotting
    translation_positions = np.array(translation_positions)

    # Print final list of translated axes
    p.xaxisplotter(xaxis, u_total)
    p.xaxisplotter(xaxis, v_total)
    p.xaxisplotter(xaxis, w_total)
    p.isotropic_turbulence_plot(xaxis, u_total, v_total)
    print(translation_positions)

#diagnostic_change_order()

def diagnostic_test_velocity(N_E):
    Nx = k.constants()[7]
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    xaxis = k.positionvectorxaxis()
    xaxis_translated_list = []
    translation_positions = []
    
    for i in range(N_E):
        # Get random angles
        thetax, thetay, thetaz = k.random_angles()
        R = k.total_matrix(thetax, thetay, thetaz)
        thetax, thetay, thetaz = 0, 0, 0
        print(thetax, thetay, thetaz)
        # Generate a random position and rotate the axis
        a = k.random_position(online=False)
        a = np.array([2, 0, 0])
        xaxis_translated = xaxis + np.array(a).reshape(3, 1)
        
        # Append the translated axis and translation vector
        xaxis_translated_list.append(xaxis_translated)
        translation_positions.append(a)

        # Plot the rotated and translated axis in 3D
        
        # Calculate velocities
        u, v, w = k.velocity_test_function(xaxis_translated)
        print(v)
        velocity_vector = np.vstack((u, v, w))
        velocity_vector_rotated = R @ velocity_vector
        u_rotated = velocity_vector_rotated[0]
        check = np.isclose(u, u_rotated)
        print(check)
        v_rotated = velocity_vector_rotated[1]
        print(v_rotated)
        w_rotated = velocity_vector_rotated[2]
        u_total += u_rotated
        v_total += v_rotated
        w_total += w_rotated
    
    # Convert translation_positions to a numpy array for easy plotting
    translation_positions = np.array(translation_positions)

    # Print final list of translated axes
    p.xaxisplotter(xaxis, u_total)
    p.xaxisplotter(xaxis, v_total)
    p.xaxisplotter(xaxis, w_total)

#diagnostic_test_velocity(1)

def diagnostic_test_velocity_2(N_E):
    N_E = k.constants()[10]
    Nx = k.constants()[7]
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    xaxis = k.positionvectorxaxis()
    xaxis_translated_list = []
    translation_positions = []
    
    for i in range(N_E):
        # Get random angles
        thetax, thetay, thetaz = k.random_angles()
        R = k.total_matrix(thetax, thetay, thetaz)

        # Generate a random position and rotate the axis
        a = k.random_position(online=False)
        xaxis_rotated = R @ xaxis
        xaxis_translated = xaxis_rotated - np.array(a).reshape(3, 1)
        
        # Append the translated axis and translation vector
        xaxis_translated_list.append(xaxis_translated)
        translation_positions.append(a)
        # Calculate velocities
        u, v, w = k.velocityline(xaxis_translated)
        u_rotated, v_rotated, w_rotated = R @ np.array([u, v, w])
        u_total += u_rotated
        v_total += v_rotated
        w_total += w_rotated
    # Convert translation_positions to a numpy array for easy plotting
    # Print final list of translated axes
    p.xaxisplotter(xaxis, u_total)
    p.xaxisplotter(xaxis, v_total)
    p.xaxisplotter(xaxis, w_total)

#diagnostic_test_velocity_2(10)

def point_test():
    L = k.constants()[0]
    x0 = np.array([2, 0, 0])
    thetax, thetay, thetaz = 0, 0, 0
    R = k.total_matrix(thetax, thetay, thetaz)
    x0_rotated = R @ x0
    x,y,z = x0_rotated
    u = -y * np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    v = x * np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    w = 0
    print(R)
    u,v,w = R @ np.array([u,v,w])
    return u,v,w
#print(point_test())
#p.plot3d(3)

def velocities_faster_plot():
    starttime = timeit.default_timer()
    u, v, w = k.velocities_faster()
    print("Elapsed time: ", timeit.default_timer() - starttime)
    xaxis = k.positionvectorxaxis()
    r, f, g, f_s, max_index = k.correlation_functions_vect(xaxis, u, v)
    p.theoretical_f(r, f, max_index)
    p.theoretical_g(r, g, max_index)
    plt.show()

def velocities_base_plot():
    starttime = timeit.default_timer()
    u, v, w = k.velocities_base()
    print("Elapsed time: ", timeit.default_timer() - starttime)
    xaxis = k.positionvectorxaxis()
    r, f, g, f_s, max_index = k.correlation_functions_vect(xaxis, u, v)
    p.theoretical_f(r, f, max_index)
    p.theoretical_g(r, g, max_index)
    plt.show()

def velocities_faster_2_plot():
    starttime = timeit.default_timer()
    u, v, w = k.velocities_faster_3()
    print("Elapsed time: ", timeit.default_timer() - starttime)
    xaxis = k.positionvectorxaxis()
    r, f, g, f_s, max_index = k.correlation_functions_vect(xaxis, u, v)
    p.theoretical_f(r, f, max_index)
    p.theoretical_g(r, g, max_index)
    plt.show()

#velocities_faster_2_plot()

def magnitude_test():
    xaxis = k.positionvectorxaxis()
    x = xaxis[0]
    N = 100
    U_0 = np.zeros(len(x))
    U_1 = np.zeros(len(x))
    for i in range(N):
        thetax, thetay, thetaz = k.random_angles()
        R = k.total_matrix(thetax, thetay, thetaz)
        a = k.random_position(online=False)
        xaxis_translated = xaxis - a
        xaxis_rotated = R @ xaxis_translated
        u0, v0, w0 = k.velocityline(xaxis_translated)
        u1, v1, w1 = k.velocityline(xaxis_rotated)
        U_0 += np.sqrt(u0**2 + v0**2 + w0**2)
        U_1 += np.sqrt(u1**2 + v1**2 + w1**2)
    print(np.isclose(U_0, U_1))
    ax, fig = plt.subplots()
    fig.plot(x, U_0, label='Original')
    fig.plot(x, U_1, label='Rotated')
    plt.legend()
    plt.show()

magnitude_test()