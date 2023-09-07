#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SIGMA_X = 0.25
SIGMA_Y = 0.25
SIGMA_I = 0.3

CONTOUR_LEVELS = np.geomspace(0.0001, 250, 100)

# Returns a random xy pair within the unit circle centered at the origin.
def random_unit_circle_coords():
    # Polar coordinates with a square-rooted r-value produce a uniform distribution
    r = np.sqrt(np.random.uniform(0, 1))
    theta = np.random.uniform(0, 2) * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

# Generates noisy measurements for all K landmarks given the true position.
def get_range_measurements(K, xy_true):
    return [generate_measurement(landmark_pos(i, K), xy_true) for i in range(K)]

# Returns the xy coordinate pair of the i'th landmark out of K total landmarks.
def landmark_pos(i, K):
    angle = 2 * np.pi / K * i
    x = np.cos(angle)
    y = np.sin(angle)
    return np.array([x, y])

# Generates a range measurement between the given landmark and true positions,
# with random Gaussian noise.
def generate_measurement(xy_landmark, xy_true):
    dTi = np.linalg.norm(xy_true-xy_landmark)
    while True:
        noise = np.random.normal(0, SIGMA_I)
        measurement = dTi + noise
        if measurement >= 0:
            return measurement

# Creates an equilevel contour plot for the MAP estimation objective function
# given a set of range measurements
def plot_equilevels(range_measurements, xy_true):
    # First, create a mesh grid of values from the objective function
    gridpoints = np.meshgrid(np.linspace(-2, 2, 128), np.linspace(-2, 2, 128))
    contour_values = MAP_objective(gridpoints, range_measurements)

    # Then, set up the plot
    plt.style.use('seaborn-white')

    ax = plt.gca()

    unit_circle = plt.Circle((0, 0), 1, color='#888888', fill=False)
    ax.add_artist(unit_circle)

    plt.contour(gridpoints[0], gridpoints[1], contour_values, cmap='plasma_r', levels=CONTOUR_LEVELS);

    for (i, r_i) in enumerate(range_measurements):
        (x, y) = landmark_pos(i, len(range_measurements))
        plt.plot((x), (y), 'o', color='g', markerfacecolor='none')
        # I added faint blue circles to demonstrate the range from each landmark
        range_circle = plt.Circle((x, y), r_i, color='#0000bb33', fill=False)
        ax.add_artist(range_circle)

    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title("MAP estimation objective contours, K = " + str(len(range_measurements)))

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.plot([xy_true[0]], [xy_true[1]], '+', color='r')
    plt.colorbar();
    plt.show()

# Calculates values of the MAP estimation objective function on a given mesh
# grid of input x-y coordinate pairs.
def MAP_objective(xy, range_measurements):
    # The shape of xy is (2, n, m), but it needs to be (n, m, 1, 2).
    xy = np.expand_dims(np.transpose(xy, axes=(1, 2, 0)), axis=len(np.shape(xy))-1)

    prior = np.matmul(xy, np.linalg.inv(np.array([[SIGMA_X**2, 0],[0, SIGMA_Y**2]])))
    prior = np.matmul(prior, np.swapaxes(xy, 2, 3))
    prior = np.squeeze(prior)
    # prior is now of shape (n, m).

    range_sum = 0

    for (i, r_i) in enumerate(range_measurements):
        xy_i = landmark_pos(i, len(range_measurements))
        d_i = np.linalg.norm(xy - xy_i[None, None, None, :], axis=3)
        range_sum += np.squeeze((r_i - d_i)**2 / SIGMA_I**2)

    return prior + range_sum


xy_true = random_unit_circle_coords()

for K in [1, 2, 3, 4]:
    range_measurements = get_range_measurements(K, xy_true)
    plot_equilevels(range_measurements, xy_true)
