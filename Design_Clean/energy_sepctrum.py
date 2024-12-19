import numpy as np
from scipy.integrate import simps, trapz
import matplotlib.pyplot as plt

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

# Define a function to filter f and g based on their second and first crossings
def f_and_g_filter(r, f, g):
    max_index = len(r)
    f_total = np.zeros(max_index)
    g_total = np.zeros(max_index)
    # Find 2nd intercept for g and 1st for f
    g_index = find_nth_crossing(g, 2)
    f_index = find_nth_crossing(f, 1)
    # Set values beyond this index to 0
    f_total[:f_index] = f[:f_index]
    g_total[:g_index] = g[:g_index]
    return f_total, g_total

def energy_spectrum(r, f, g, tol, u_2_avg):
    # Assuming r_array, f_array, and g_array are given
    r_array = r  # array of r values
    f_array = f  # array of f(r) values
    g_array = g  # array of g(r) values

    # Calculate R(r)
    R_array = u_2_avg**2 * (g_array + f_array / 2)

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

def energy_spectrum_plot(E_k, k_array):
    fig, ax = plt.subplots()
    ax.plot(k_array, E_k)
    ax.set_xlabel('k=π/r')
    ax.set_ylabel('E(k)')
    plt.title('Energy Spectrum')


def main():
    L_e = 1.0
    r = np.linspace(0, 5*L_e, int(5/0.05))
    f_theoretical = np.exp(-r**2/L_e**2)
    g_theoretical = (1 - r**2/L_e**2) * np.exp(-r**2/L_e**2)
    E_k, k_array = energy_spectrum(r, f_theoretical, g_theoretical, 0.05, 1.0)
    energy_spectrum_plot(E_k, k_array)
    plt.show()

#main()