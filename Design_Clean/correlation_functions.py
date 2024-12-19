import numpy as np
from numba import njit
import sympy as sp

# Function to calculate the correlation functions
# Note only need u^2 average as the turbulence is isotropic
def correlation_functions_vect(u_total, v_total, tol, plot_limit, x_boundary, u_2_average, v_2_average):
    N_u = len(u_total)
    s = np.arange(N_u)
    max_index = int(plot_limit/tol)
    r = np.linspace(0, 2*x_boundary, 2*int(x_boundary/tol))
    f = []
    g = []
    f_s = []
    
    for i in range(N_u):
        shift = s[i]
        if shift < N_u:
            u_shifted = np.roll(u_total, -shift)
            v_shifted = np.roll(v_total, -shift)
            valid_range = slice(0, N_u - shift)
            product_list_f = u_total[valid_range] * u_shifted[valid_range]
            product_list_g = v_total[valid_range] * v_shifted[valid_range]
            product_list_f_s = (u_total[valid_range] - u_shifted[valid_range])**2
            
            f.append(np.mean(product_list_f))
            g.append(np.mean(product_list_g))
            f_s.append(np.mean(product_list_f_s))
    
    # Normalize the correlation functions
    f = f / u_2_average
    g = g / v_2_average
    return r, f, g, f_s, max_index

# Townsend's structure function
def townsend_structure(f, r, u_2_average):
    delta_v = 2 * u_2_average * (1 - f)
    dvdr = np.gradient(delta_v, r)
    townsends = 3/4 * dvdr
    return townsends, dvdr

def signature_function_1(f, r, u_2_average):
    dfdr = np.gradient(f, r)
    df2dr2 = np.gradient(dfdr, r)
    signature_1 = 3/4 * u_2_average * (r * df2dr2 - dfdr)
    return signature_1

@njit
def g(x):
    return (7 * x**5 - 2 * x**7) * np.exp(-x**2)

@njit
def dgds(s, r):
    return ((35 / r**5) * s**4 - (14 / r**7) * s**6) * np.exp(-(s / r)**2) - (2 * s / r**2) * g(s / r)

@njit
def signature_function_2(delta_v_2, r, u_2_average, limit):
    # Use a slice of r directly, avoid reassigning
    s = r[:limit + 1]
    n_r = len(s)
    n_s = len(s)

    # Initialize arrays
    u_v = np.zeros(n_r)
    integrand = np.zeros((n_r, n_s))

    # Compute u_v and integrand
    for j in range(n_r):
        r_j = s[j]
        if r_j == 0:
            continue  # Skip if r_j is 0

        # Compute u_v for the current r_j
        u_v[j] = delta_v_2[-1] * g(s[-1] / r_j) - delta_v_2[0] * g(s[0] / r_j)

        # Compute the integrand for all s
        for i in range(n_s):
            integrand[j, i] = dgds(s[i], r_j) * delta_v_2[i]

    # Compute the signature function
    signature_2 = np.zeros(n_r)
    for j in range(n_r):
        signature_2[j] = u_v[j] - np.trapz(integrand[j, 1:], s[1:])  # Integrate over s[1:]

    return signature_2

@njit
def integral_by_parts(v, u, dv_ds, s, a=None, b=None):
    """
    Compute the integral by parts given v, u, s, and optional limits.

    Parameters:
        v: sympy expression, the function v(s)
        u: sympy expression, the function u(s)
        s: sympy symbol, the variable of integration
        a: lower limit of integration (default: None for indefinite integral)
        b: upper limit of integration (default: None for indefinite integral)

    Returns:
        sympy expression: The result of the integration by parts.
    """

    if a is not None and b is not None:  # Definite integral
        boundary_term = v[-1] * u[-1] - v[0] * u[0]
        integral_term = np.trapz(u * dv_ds, s)
        return boundary_term - integral_term
    else:  # Indefinite integral
        return v * u - np.trapz(u * dv_ds, s)


def signature_function_2_1(delta_v_2, r, u_2_average, limit):
    # Use a slice of r directly, avoid reassigning
    r = r[1:]
    s = r
    n_r = len(s)
    a = 0
    b = s[-1]
    signature_2 = np.zeros(n_r+1)
    for i,r_i in enumerate(r):
        g_v = g(s/r_i)
        dgds_v = dgds(s,r_i)
        integral = integral_by_parts(g_v,delta_v_2,dgds_v,s,a,b)
        signature_2[i+1] = integral
    return signature_2
    