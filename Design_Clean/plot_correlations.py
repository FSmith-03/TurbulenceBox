import numpy as np
import matplotlib.pyplot as plt

def structure_plotter(r, f, max_index, title):
    f = f[0:max_index]
    r = r[0:max_index]
    max_location = r[np.argmax(f)]
    max_location = round(max_location, 4)
    fig, ax = plt.subplots()
    ax.plot(r, f)
    ax.axvline(x=max_location, color='red', linestyle='--', label=f'Location of max value: {max_location}')
    ax.set_xlabel('r/L')
    ax.set_ylabel(title)
    plt.title(title)
    plt.legend()

def theoretical_f(r, f, max_index, L_e):
    r = r[0:max_index]
    f = f[0:max_index]
    f_theoretical = np.exp(-r**2/L_e**2)
    fig, ax = plt.subplots()
    ax.plot(r, f, label='f', c='r')
    ax.plot(r, f_theoretical, label='Theoretical f', c = 'b', linestyle='--')
    ax.set_xlabel('r/L_e')
    ax.set_ylabel('f')
    plt.legend()
    plt.title('Longitudinal Correlation Function')

def theoretical_g(r, g, max_index, L_e):
    g = g[0:max_index]
    r = r[0:max_index]
    g_theoretical = (1 - r**2/L_e**2) * np.exp(-r**2/L_e**2)
    fig, ax = plt.subplots()
    ax.plot(r, g, label='g', c='r')
    ax.plot(r, g_theoretical, label='Theoretical g', c = 'b', linestyle='--')
    ax.set_xlabel('r/L_e')
    ax.set_ylabel('g')
    plt.legend()
    plt.title('Transverse Correlation Function')