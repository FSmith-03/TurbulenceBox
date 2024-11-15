import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from matplotlib.widgets import Slider
import random
from numba import njit, prange
import random
import math as m
import timeit
import plotly.graph_objects as go
import kinematics as k
import plotting as p
import pandas as pd

def write_to_csv(data, filename="output.csv"):
    # Convert each array in `data` to a DataFrame column
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data written to {filename}")

def main():
    #Load constants
    N_E = k.constants()[10]
    #Initialize arrays for velocity components
    u_total, v_total, w_total = k.velocities_faster_2()
    xaxis = k.positionvectorxaxis()
    #Plot correlation and structure functions
    #Calculating correlation functions
    r, f, g, f_s, max_index = k.correlation_functions_vect(xaxis, u_total, v_total)
    write_to_csv_variable = False
    if write_to_csv_variable == True:
        print("Writing data to output1_20000.csv")
        write_to_csv({
        "u_total": u_total,
        "v_total": v_total,
        "w_total": w_total,
        }, "output1_20000.csv")
        print("Writing data to output2_20000.csv")
        write_to_csv({
        "f": f,
        "g": g,
        "f_s": f_s,
        }, "output2_20000.csv")
    p.isotropic_turbulence_plot(r, f, g, max_index)
    p.theoretical_f(r, f, max_index)
    p.theoretical_g(r, g, max_index)
    f_filtered, g_filtered = k.f_and_g_filter(f, g)
    print("Plotting Structure Function")
    p.structure_plotter(r, f_s, max_index)
    print("Calculating Townsend's Structure Function")
    townsend = k.townsend_structure(f, r)
    print("Plotting Townsend's Structure Function")
    p.structure_plotter(r, townsend, max_index)
    E_k, k_array = k.energy_spectrum(r, f_filtered, g_filtered)
    print("Plotting Energy Spectrum")
    p.energy_spectrum_plot(E_k, k_array)
    plt.show()
    return

@njit
def random_angles_fast(N):
    theta = np.random.uniform(0, 2*m.pi, size=N)
    phi = np.random.uniform(0, 2*m.pi, size=N)
    psi = np.random.uniform(0, 2*m.pi, size=N)
    return theta, phi, psi

@njit
def total_matrix_fast(theta_x, theta_y, theta_z):
    # Implement inline matrix multiplication for performance.
    return np.array([
        [np.cos(theta_y) * np.cos(theta_z), -np.cos(theta_y) * np.sin(theta_z), np.sin(theta_y)],
        [np.cos(theta_x) * np.sin(theta_z) + np.sin(theta_x) * np.sin(theta_y) * np.cos(theta_z),
         np.cos(theta_x) * np.cos(theta_z) - np.sin(theta_x) * np.sin(theta_y) * np.sin(theta_z),
         -np.sin(theta_x) * np.cos(theta_y)],
        [np.sin(theta_x) * np.sin(theta_z) - np.cos(theta_x) * np.sin(theta_y) * np.cos(theta_z),
         np.sin(theta_x) * np.cos(theta_z) + np.cos(theta_x) * np.sin(theta_y) * np.sin(theta_z),
         np.cos(theta_x) * np.cos(theta_y)]
    ])

def main_optimized():
    # Load constants once
    N_E = k.constants()[10]
    print(N_E)
    Nx = k.constants()[7]
    xaxis = k.positionvectorxaxis()
    
    # Initialize total velocity arrays
    u_total, v_total, w_total = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
    
    # Generate random angles and positions for all eddies at once
    print("Generating random angles and positions")
    thetax, thetay, thetaz = random_angles_fast(N_E)
    positions = np.array([k.random_position() for _ in range(N_E)])
    print('Beginning loop')
    for i in prange(N_E):  # Parallel loop over eddies
        if i+1 % 1000 == 0:
            print("Eddy number: ", i+1)
        # Rotation matrix calculation
        R = total_matrix_fast(thetax[i], thetay[i], thetaz[i])
        
        # Translate and rotate x-axis
        xaxis_translated = xaxis - positions[i, :]
        xaxis_rotated = R @ xaxis_translated
        
        # Calculate and accumulate velocity components
        u, v, w = k.velocitylineenhanced(xaxis_rotated)
        u_total += u
        v_total += v
        w_total += w
    print("Writing data to output.csv")
    write_to_csv({
    "u_total": u_total,
    "v_total": v_total,
    "w_total": w_total
    }, "output_10000.csv")
    print("Plotting Isotropic Turbulence Correlations f and g")
    r, f, g = k.correlation_functions(xaxis, u_total, v_total)
    p.isotropic_turbulence_plot(r, f, g)
    print("Plotting Structure Function")
    p.structure_plotter(xaxis, u_total)
    return

#two_eddy_test()
main()