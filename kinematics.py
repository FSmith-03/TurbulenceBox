import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
import math as m
import timeit
import plotly.graph_objects as go

def spacefiller(x_boundary, y_boundary, z_boundary):
    N = 8*x_boundary*y_boundary*z_boundary
    N = int(N)
    return N

def constants():
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
    limit = 20
    N_E = spacefiller(x_boundary, y_boundary, z_boundary)
    return L, tol, x_boundary, y_boundary, z_boundary, Nxf, Nyz, Nx, Nyz, limit, N_E

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
    return rotation_matrix_x(theta_x) @ rotation_matrix_y(theta_y) @ rotation_matrix_z(theta_z)

def total_matrix_check(theta_x, theta_y, theta_z):
    R_22 = np.cos(theta_x)*np.cos(theta_z) - np.sin(theta_x)*np.sin(theta_y)*np.sin(theta_z)
    check = np.isclose(R_22, total_matrix(theta_x, theta_y, theta_z)[1, 1], atol=1e-5)
    if check == True:
        print('The rotation matrix is correct')
    else:
        print('The rotation matrix is incorrect')
    return

#Random angles for the rotation matrix
def random_angles():
   theta = random.uniform(0, 2*m.pi)
   phi = random.uniform(0, 2*m.pi)
   psi = random.uniform(0, 2*m.pi)
   return theta, phi, psi

#Random position for the eddy
def random_position(boundary=0, online=False):
    L = constants()[0]
    x_boundary = constants()[2]
    y_boundary = constants()[3]
    z_boundary = constants()[4]
    if boundary == 0:
        x = random.uniform(-x_boundary*L, x_boundary*L)
        y = random.uniform(-y_boundary*L, y_boundary*L)
        z = random.uniform(-z_boundary*L, z_boundary*L)
    else:
        x = random.uniform(-boundary*L, boundary*L)
        y = random.uniform(-boundary*L, boundary*L)
        z = random.uniform(-boundary*L, boundary*L)
    a = np.array([x, y, z])
    if online == True:
        a = np.array([x, 0, 0])
    a = a[:, np.newaxis]
    return a

#Set of all sensor positions on the x-axis
def positionvectorxaxis(boundary=0):
    L = constants()[0]
    x_boundary = constants()[2]
    Nx = constants()[7]
    if boundary == 0:
        x = np.linspace(-x_boundary*L, x_boundary*L, Nx)
    else:
        x = np.linspace(-boundary*L, boundary*L, Nx)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    pos_vector = np.vstack((x, y, z))
    return pos_vector

#Rotate x axis sensor positions
def positionvectorrotated(theta_x, theta_y, theta_z):
    R = total_matrix(theta_x, theta_y, theta_z)
    pos_vector = positionvectorxaxis()
    pos_vector_rotated = R @ pos_vector
    return pos_vector_rotated

def positionvectordisplaced(pos_vector, a):
    a = np.array(a).reshape(3, 1)
    pos_vector_displaced = pos_vector + a
    return pos_vector_displaced

def velocityline(pos_vector):
    L = constants()[0]
    x = pos_vector[0, :]  # Ensures x, y, z are 1D arrays of shape (200,)
    y = pos_vector[1, :]
    z = pos_vector[2, :]
    u = -y * np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    v = x * np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    w = np.zeros_like(u)
    return u, v, w

def velocity_test_function(pos_vector):
    L = constants()[0]
    x = pos_vector[0]
    y = pos_vector[1]
    z = pos_vector[2]
    u = np.exp(-x**2+y**2)
    v = 0
    w = 0
    return u, v, w

#Enhanced velocity line function that does not calculate the velocity at every point but only the local coordinates
def velocitylineenhanced(pos_vector):
    L = constants()[0]
    x = pos_vector[0]
    y = pos_vector[1]
    z = pos_vector[2]
    
    # Create a boolean mask where |x| < 2.95L
    mask = np.abs(np.sqrt(x**2 + y**2 + z**2)) < 4*L
    
    # Initialize u, v, and w as zero arrays
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    w = np.zeros_like(x)
    
    # Apply calculations only where the mask is True
    factor = np.exp(2*-(x**2 + y**2 + z**2) / (L**2))
    u[mask] = -y[mask] * factor[mask]
    v[mask] = x[mask] * factor[mask]
    # w remains zeros as per the original logic
    
    return u, v, w, mask

def correlation_functions(pos_vector, u, v):
    #Load tolerance and limit of plotting
    tol = constants()[1]
    limit = 10
    #Calculate the maximum index of the plotted arrays
    max_index = int(limit/tol)
    #Assign the maximum index of the velocity array
    N_u = len(u)
    #Create an aray of integers from 0 to N_u to be used as the total number of compared points
    s = np.arange(0, N_u)
    #Create an array of spacings to be plotted
    r = np.linspace(tol, max_index*tol, max_index)
    #Create empty arrays for the correlation functions
    f = []
    g = []
    f_s = []

    ## please parallelize this loop for 4 cores
    #Iterate over the number of velocity points
    for i in range(N_u):
        #Reset the product list for each spacing
        product_list_f = []
        product_list_g = []
        product_list_f_s = []
        #Iterate over the number of compared points
        for j in range(N_u - s[i]):
            #Store the product of the velocity components at the compared points
            product_list_f.append(u[j]*u[j+s[i]])
            product_list_g.append(v[j]*v[j+s[i]])
            product_list_f_s.append((u[j]-u[j+s[i]])**2)
        #print("Product list mean for", i, "is", np.mean(product_list))
        #Store the mean of the product list in the correlation function arrays
        f.append(np.mean(product_list_f))
        g.append(np.mean(product_list_g))
        f_s.append(np.mean(product_list_f_s))
    #Normalize the correlation functions
    if f[0] != 0:
        f = f/f[0]
    else:
        f = f
    if g[0] != 0:
        g = g/g[0]
    else:
        g = g
    return r, f, g, f_s, max_index

import numpy as np

def correlation_functions_vect(xaxis, u_total, v_total):
    tol = constants()[1]
    N_u = len(u_total)
    limit = 10
    s = np.arange(N_u)
    max_index = int(limit/tol)
    r = np.linspace(tol, max_index*tol, max_index)
    f = []
    g = []
    f_s = []
    
    for i in range(N_u):
        shift = s[i]
        if shift < N_u:
            u_shifted = np.roll(u_total, -shift)
            v_shifted = np.roll(v_total, -shift)
            #print(u_total)
            #print(u_shifted)
            valid_range = slice(0, N_u - shift)
            #print(valid_range)
            product_list_f = u_total[valid_range] * u_shifted[valid_range]
            product_list_g = v_total[valid_range] * v_shifted[valid_range]
            product_list_f_s = (u_total[valid_range] - u_shifted[valid_range])**2
            
            f.append(np.mean(product_list_f))
            g.append(np.mean(product_list_g))
            f_s.append(np.mean(product_list_f_s))
    
    # Normalize the correlation functions
    if f[0] != 0:
        f = f / f[0]
    if g[0] != 0:
        g = g / g[0]
    
    return r, f, g, f_s, max_index

def townsend_structure(f, r):
    f = np.array(f)
    f = f[0:len(r)]
    dfdr = np.gradient(f, r)
    townsend = -3/2  * dfdr
    return townsend
    

"""
def correlation_functions_fast_legacy(pos_vector, u, v):
    # Load tolerance and limit of plotting
    tol = constants()[1]
    limit = 10
    # Calculate the maximum index of the plotted arrays
    max_index = int(limit / tol)
    # Assign the maximum index of the velocity array
    N_u = len(u)
    # Create an array of spacings to be plotted
    r = np.linspace(tol, max_index * tol, max_index)
    
    # Compute the pairwise products using outer products for u and v
    product_u = np.outer(u, u)
    product_v = np.outer(v, v)
    
    # Initialize arrays for correlation functions f and g
    f = np.zeros(max_index)
    g = np.zeros(max_index)
    
    # Calculate the mean of products for each spacing
    for k in range(1, max_index):
        # Select the k-th diagonal for each matrix and compute the mean
        f[k] = np.mean(np.diag(product_u, k))
        g[k] = np.mean(np.diag(product_v, k))
    
    # Normalize the correlation functions
    if f[0] != 0:
        f = f / f[0]
    if g[0] != 0:
        g = g / g[0]
    
    return r, f[:max_index], g[:max_index]
"""

def find_nth_crossing(f_values, n):
    """
    Finds the nth crossing point of a function across the x-axis.

    Parameters:
    - f_values (list or np.array): The evaluated values of f at discrete x points.
    - n (int): The crossing number to find (1 for first, 2 for second, etc.)

    Returns:
    - int: The index of the nth crossing point if it exists, else -1.
    """
    # Count the number of crossings found
    crossing_count = 0
    
    # Iterate over pairs of consecutive values to detect sign changes
    for i in range(1, len(f_values)):
        # Check for sign change between f_values[i-1] and f_values[i]
        if f_values[i-1] * f_values[i] < 0:
            crossing_count += 1
            # If this is the nth crossing, return the index
            if crossing_count == n:
                return i - 1  # Return the index of the point just before the crossing

    # If the nth crossing wasn't found, return -1
    return -1

def f_and_g_filter(f, g):
    # Find 2nd intercept for g and 1st for f
    g_index = find_nth_crossing(g, 2)
    f_index = find_nth_crossing(f, 1)
    # Set values beyond this index to 0
    f[f_index:] = 0
    g[g_index:] = 0
    return f, g


from scipy.integrate import simps, trapz
def energy_spectrum(r, f, g):
    tol = constants()[1]
    # Assuming r_array, f_array, and g_array are given
    u_rms = 1  # root-mean-square velocity (value needed)
    r_array = np.linspace(tol, len(f)*tol, len(f))  # distances (r values)
    f_array = f  # array of f(r) values
    g_array = g  # array of g(r) values

    # Calculate R(r)
    R_array = u_rms**2 * (g_array + f_array / 2)

    # Define a range of k values (wavenumbers)
    k_array = np.linspace(0.1, 10, len(r))  # Adjust range and density as needed

    # Reshape arrays to enable broadcasting for matrix operations
    # r_array has shape (N,), where N is the number of r values
    # k_array has shape (M,), where M is the number of k values
    # We need to broadcast to a shape (M, N) for simultaneous calculation
    k_grid, r_grid = np.meshgrid(k_array, r_array, indexing='ij')  # shape (M, N)
    sin_kr = np.sin(k_grid * r_grid)  # shape (M, N)

    # Compute the integrand for all k and r values
    integrand = R_array * r_grid * sin_kr  # shape (M, N)

    # Integrate over r for each k using Simpson's rule (or np.trapz for trapezoidal rule)
    # The axis=1 specifies integration over the r dimension
    E_k = (2 / np.pi) * trapz(integrand * k_grid, r_array, axis=1)  # shape (M,)

    # E_k now contains the energy spectrum values at each wavenumber in k_array
    return E_k, k_array

from scipy.fft import fft

def energy_spec_fft(r, f, g):
    R = g + f/2
    N = len(r)
    k = np.linspace(0, 8, N)
    return

def isotropic_turbulence_test_plot(pos_vector, u, v):
    tol = constants()[1]
    limit = 10
    max_index = int(limit/tol)
    x = pos_vector[0:max_index]
    N_u = len(u[0:max_index])
    s = np.arange(0, N_u)
    r = np.linspace(tol, max_index*tol, max_index)
    f = []
    for i in range(N_u):
        product_list = []
        for j in range(N_u - s[i]):
            product_list.append(u[j]*u[j+s[i]])
        #print("Product list mean for", i, "is", np.mean(product_list))
        f.append(np.mean(product_list))
    f = f/f[0]
    fig, ax = plt.subplots()
    ax.plot(r, f)
    ax.set_xlabel('s')
    ax.set_ylabel('f')
    plt.title('Isotropic turbulence')
    plt.show()

#Find the first and last idexes that the eddy has influence on
def velocity_range(mask):
    first_index = np.argmax(mask)
    last_index = len(mask) - np.argmax(mask[::-1])
    return first_index, last_index

#Find the first and last indexes that the eddy has influence on from the position vector
def masker(pos_vectors):
    mask = np.abs(np.sqrt(pos_vectors[0]**2 + pos_vectors[1]**2 + pos_vectors[2]**2)) < 4
    first_index = np.argmax(mask)
    last_index = len(mask) - np.argmax(mask[::-1])
    return mask, first_index, last_index

def velocities_faster():
    #Load constants
    N_E = 10000
    print("Number of eddies: ", N_E)
    Nx = constants()[7]
    #Initialize arrays for velocity components
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    #Generate x-axis
    xaxis = positionvectorxaxis()
    #Iterate over all eddies
    for i in range(N_E):
        #Print eddy number for every 1000 eddies
        if (i+1) % 1000 == 0:
            print("Eddy number: ", i+1)
        #Generate random angles
        thetax, thetay, thetaz = random_angles()
        #Calculate rotation matrix
        R = total_matrix(thetax, thetay, thetaz)
        #Generate random position
        a = random_position()
        #Translate x-axis
        xaxis_translated = xaxis - a
        #Rotate translated x-axis
        xaxis_rotated = R @ xaxis_translated
        #Calculate velocity components
        u, v, w, mask = velocitylineenhanced(xaxis_rotated)
        first_index, last_index = velocity_range(mask)
        u = u[first_index:last_index]
        v = v[first_index:last_index]
        w = w[first_index:last_index]
        #Rotate velocity components by inverse angles
        R = total_matrix(-thetax, -thetay, -thetaz)
        u_rotated, v_rotated, w_rotated = R @ np.array([u, v, w])
        #Add rotated velocity components to total velocity components
        u_total[first_index:last_index] += u_rotated
        v_total[first_index:last_index] += v_rotated
        w_total[first_index:last_index] += w_rotated
    return u_total, v_total, w_total

#Currently the fastest version of the function filters out points that are not within the eddy's influence
def velocities_faster_2():
    #Load constants
    N_E = constants()[10]
    print("Number of eddies: ", N_E)
    Nx = constants()[7]
    #Initialize arrays for velocity components
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    #Generate x-axis
    xaxis = positionvectorxaxis()
    #Iterate over all eddies
    for i in range(N_E):
        #Print eddy number for every 1000 eddies
        if (i+1) % 1000 == 0:
            print("Eddy number: ", i+1)
        #Generate random angles
        thetax, thetay, thetaz = random_angles()
        #Calculate rotation matrix
        R = total_matrix(thetax, thetay, thetaz)
        #Generate random position
        a = random_position()
        #Translate x-axis
        xaxis_translated = xaxis - a
        mask, first_index, last_index = masker(xaxis_translated)
        xaxis_translated_trimmed = xaxis_translated[:, first_index:last_index]
        #Rotate translated x-axis
        xaxis_rotated = R @ xaxis_translated_trimmed
        #Calculate velocity components
        u, v, w, _ = velocitylineenhanced(xaxis_rotated)
        #Rotate velocity components by inverse angles
        R = total_matrix(-thetax, -thetay, -thetaz)
        u_rotated, v_rotated, w_rotated = R @ np.array([u, v, w])
        #Add rotated velocity components to total velocity components
        u_total[first_index:last_index] += u_rotated
        v_total[first_index:last_index] += v_rotated
        w_total[first_index:last_index] += w_rotated
    return u_total, v_total, w_total

def velocities_faster_3():
    #Load constants
    N_E = constants()[10]
    print("Number of eddies: ", N_E)
    Nx = constants()[7]
    #Initialize arrays for velocity components
    u_total_matrix = np.zeros((Nx, N_E))
    v_total_matrix = np.zeros((Nx, N_E))
    w_total_matrix = np.zeros((Nx, N_E))
    #Generate x-axis
    xaxis = positionvectorxaxis()
    #Iterate over all eddies
    for i in range(N_E):
        #Print eddy number for every 1000 eddies
        if (i+1) % 1000 == 0:
            print("Eddy number: ", i+1)
        #Generate random angles
        thetax, thetay, thetaz = random_angles()
        #Calculate rotation matrix
        R = total_matrix(thetax, thetay, thetaz)
        #Generate random position
        a = random_position()
        #Translate x-axis
        xaxis_translated = xaxis - a
        mask, first_index, last_index = masker(xaxis_translated)
        xaxis_translated_trimmed = xaxis_translated[:, first_index:last_index]
        #Rotate translated x-axis
        xaxis_rotated = R @ xaxis_translated_trimmed
        #Calculate velocity components
        u, v, w = velocitylineenhanced(xaxis_rotated)
        #Rotate velocity components by inverse angles
        R = total_matrix(-thetax, -thetay, -thetaz)
        u_rotated, v_rotated, w_rotated = R @ np.array([u, v, w])
        #Add rotated velocity components to total velocity components
        u_total_matrix[first_index:last_index, i] = u_rotated
        v_total_matrix[first_index:last_index, i] = v_rotated
        w_total_matrix[first_index:last_index, i] = w_rotated
    u_total = np.sum(u_total_matrix, axis=1)
    v_total = np.sum(v_total_matrix, axis=1)
    w_total = np.sum(w_total_matrix, axis=1)
    return u_total, v_total, w_total

def velocities_base():    #Load constants
    N_E = 10000
    print("Number of eddies: ", N_E)
    Nx = constants()[7]
    #Initialize arrays for velocity components
    u_total = np.zeros(Nx)
    v_total = np.zeros(Nx)
    w_total = np.zeros(Nx)
    #Generate x-axis
    xaxis = positionvectorxaxis()
    #Iterate over all eddies
    for i in range(N_E):
        #Print eddy number for every 1000 eddies
        if (i+1) % 1000 == 0:
            print("Eddy number: ", i+1)
        #Generate random angles
        thetax, thetay, thetaz = random_angles()
        #Calculate rotation matrix
        R = total_matrix(thetax, thetay, thetaz)
        #Generate random position
        a = random_position()
        #Translate x-axis
        xaxis_translated = xaxis - a
        #Rotate translated x-axis
        xaxis_rotated = R @ xaxis_translated
        #Calculate velocity components
        u, v, w, mask = velocitylineenhanced(xaxis_rotated)
        #Rotate velocity components by inverse angles
        R = total_matrix(-thetax, -thetay, -thetaz)
        u_rotated, v_rotated, w_rotated = R @ np.array([u, v, w])
        #Add rotated velocity components to total velocity components
        u_total += u_rotated
        v_total += v_rotated
        w_total += w_rotated
    return u_total, v_total, w_total