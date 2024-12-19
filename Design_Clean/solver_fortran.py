import input_settings as inp
import random_generator as rg
import positions_and_velocities_copy as pv
import numpy as np
import correlation_functions as cf
import matplotlib.pyplot as plt
import plot_correlations as pc
import energy_sepctrum as es
import fmodpy
import pandas as pd


def write_to_csv(data, filename="output.csv"):
    # Convert each array in `data` to a DataFrame column
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data written to {filename}")

def filenamer(choice):
    x_boundary = inp.constants()[2]
    if choice == 1:
        filename = f"velocitycomponents_{x_boundary}.csv"
    elif choice == 2:
        filename = f"correlationfunctions_{x_boundary}.csv"
    return filename

def solver():
    # Constants
    L_e, tol, x_boundary, y_boundary, z_boundary, _, _, Nx, _, plot_limit, N_E = inp.constants()
    # Random angles and positions
    print("Total number of eddies: ", N_E)
    print("Generating random angles and positions")
    theta_list = rg.random_angles(N_E)
    a_list = rg.random_positions(x_boundary, y_boundary, z_boundary, N_E)
    print(a_list.shape)
    # Total velocities
    print("Calculating total velocities")
    u_total, v_total, w_total = pv.total_velocities(Nx, x_boundary, N_E, tol, theta_list, a_list)
    # Write to csv
    print("Writing data to csv")
    write_to_csv({"u_total": u_total, "v_total": v_total, "w_total": w_total,}, filenamer(1))
    #Mean velocity squared
    u_2_average = pv.u_2_average(u_total)
    v_2_average = pv.u_2_average(v_total)
    w_2_average = pv.u_2_average(w_total)
    print(u_2_average, v_2_average, w_2_average)
    # Correlation functions
    print("Calculating correlation functions")
    r, f, g, f_s, max_index_plot = cf.correlation_functions_vect(u_total, w_total, tol, plot_limit, x_boundary, u_2_average, w_2_average)
    # Write to csv
    print("Writing data to csv")
    write_to_csv({"r": r, "f": f, "g": g, "f_s": f_s}, filenamer(2))
    f_filter, g_filter = es.f_and_g_filter(r, f, g)
    f = f_filter
    g = g_filter
    #Plot correlation and structure functions
    #Calculate Townsend and Signature functions
    print("Calculating Townsend and Signature functions")
    townsends, dvdr = cf.townsend_structure(f, r, u_2_average)
    signature_1 = cf.signature_function_1(f, r, u_2_average)
    signature_2 = cf.signature_function_2_1(f_s, r, u_2_average, max_index_plot)
    print("Plotting correlation and structure functions")
    pc.theoretical_f(r, f, max_index_plot, L_e)
    pc.theoretical_g(r, g, max_index_plot, L_e)
    pc.structure_plotter(r, f_s, max_index_plot, 'Structure Function')
    pc.structure_plotter(r, townsends, max_index_plot, 'Townsend Structure Function')
    pc.structure_plotter(r, signature_1, max_index_plot, 'Signature Function 1')
    pc.structure_plotter(r, signature_2, max_index_plot, 'Signature Function 2')
    plt.show()
    return

solver()
