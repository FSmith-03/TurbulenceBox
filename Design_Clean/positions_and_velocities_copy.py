import numpy as np
import rotation_matrix as rm
import numba as nb
import random_generator as rg
from joblib import Parallel, delayed
import fmodpy
import matplotlib.pyplot as plt

code = fmodpy.fimport("routines_simplified.f90")
#print(dir(code))
def sensor_line(x_boundary, Nxf):
    x = np.linspace(-x_boundary, x_boundary, Nxf)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    pos_vector = np.vstack([x, y, z])
    return pos_vector

def eddy_range(pos_vectors):
    factor = pos_vectors[0]**2 + pos_vectors[1]**2 + pos_vectors[2]**2
    mask = np.sqrt(factor) < 4
    first_index = np.argmax(mask)
    last_index = len(mask) - np.argmax(mask[::-1])
    return first_index, last_index, factor[first_index:last_index]

def velocity_generator(xaxis_trimmed, factor):
    u = -xaxis_trimmed[1] * np.exp(-factor * 2)
    v = xaxis_trimmed[0] * np.exp(-factor * 2)
    w = np.zeros_like(u)
    return u, v, w

def u_2_average(u_total):
    u_2 = u_total**2
    u_2_average = np.mean(u_2)
    return u_2_average

def total_velocities(Nx, x_boundary, N_E, theta_list, a_list):
    xaxis = sensor_line(x_boundary, Nx)
    input = np.array([x_boundary, Nx, int(N_E)])
    velocities_total = code.main_calculation(input, a_list, theta_list)
    
    print("Shape of velocities_total:", np.shape(velocities_total))

    # Separate components
    velocities_list = velocities_total[2]
    u_total, v_total, w_total = velocities_list[:,0], velocities_list[:,1], velocities_list[:,2]
    return u_total, v_total, w_total


a_list = rg.random_positions(1000, 2.95, 2.95, 69620)
a_list = np.ascontiguousarray(a_list)
theta_list = rg.random_angles(69620)
theta_list = np.ascontiguousarray(theta_list)
#total_velocities(40000, 1000, 69620, a_list, theta_list)

def process_single_eddy(Nx, x_boundary, theta_list, a_list, i):
    u_result, v_result, w_result = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
    xaxis = sensor_line(x_boundary, Nx)
    a  = a_list[:,i][:, np.newaxis]
    eddy_pos_translated = xaxis - a
    first_index, last_index, factor = eddy_range(eddy_pos_translated)
    eddy_pos_trimmed = eddy_pos_translated[:, first_index:last_index]
    thetax, thetay, thetaz = theta_list[i]
    R = rm.rotation_total(thetax, thetay, thetaz)
    R_inv = rm.rotation_total(-thetax, -thetay, -thetaz)
    eddy_pos_rotated = R @ eddy_pos_trimmed
    u, v, w = velocity_generator(eddy_pos_rotated, factor)
    u_rotated, v_rotated, w_rotated = R_inv @ np.array([u, v, w])
    u_result[first_index:last_index] = u_rotated
    v_result[first_index:last_index] = v_rotated
    w_result[first_index:last_index] = w_rotated
    return u_result, v_result, w_result

def total_velocities_parallel(Nx, x_boundary, N_E, theta_list, a_list, n_jobs=-1):
    """
    Compute the total velocities using parallel processing.
    """
    xaxis = sensor_line(x_boundary, Nx)
    
    # Parallel processing
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_single_eddy)(Nx, x_boundary, theta_list, a_list, i) for i in range(N_E)
    )
    
    # Aggregate results
    u_total, v_total, w_total = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
    for u, v, w in results:
        u_total += u
        v_total += v
        w_total += w
    
    return u_total, v_total, w_total
    
