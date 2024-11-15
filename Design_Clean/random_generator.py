import random
import numpy as np


def random_angles(N_eddies):
    angles_list = np.random.uniform(0, 2*np.pi, (N_eddies, 3))
    return angles_list


def random_positions(x_boundary, y_boundary, z_boundary, N_eddies):
    positions_list_x = np.random.uniform(-x_boundary, x_boundary, (N_eddies))
    positions_list_y = np.random.uniform(-y_boundary, y_boundary, (N_eddies))
    positions_list_z = np.random.uniform(-z_boundary, z_boundary, (N_eddies))
    positions_list = np.array([positions_list_x, positions_list_y, positions_list_z])
    return positions_list
