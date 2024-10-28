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

def main():
    N_E = k.constants()[10]
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
        R = k.total_matrix(thetax, thetay, thetaz)
        a = k.random_position()
        a_list.append(a.reshape(3))
        #print("a: ", a)
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
    p.isotropic_turbulence_plot(xaxis, u_total, v_total)
    return

main()