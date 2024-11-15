import numpy as np


# Function to calculate the correlation functions
# Note only need u^2 average as the turbulence is isotropic
def correlation_functions_vect(u_total, v_total, tol, plot_limit, u_2_average, v_2_average):
    N_u = len(u_total)
    s = np.arange(N_u)
    max_index = int(plot_limit/tol)
    r = np.linspace(tol, max_index*tol, max_index)
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
