import numpy as np
import random

# Function to introduce amplitude errors (modifies data in place)
def amplitude_errors(data, standard_deviation):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply amplitude error to both components (E_theta and E_phi)
            amplitude_error_theta = 1 + np.random.normal(0, standard_deviation)
            amplitude_error_phi = 1 + np.random.normal(0, standard_deviation)
            data[i, j, 0] *= amplitude_error_theta
            data[i, j, 1] *= amplitude_error_phi

# Function to introduce phase errors (modifies data in place)
def phase_errors(data, standard_deviation):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error_theta = 2*np.pi * np.random.normal(0, standard_deviation)
            phase_error_phi = 2*np.pi * np.random.normal(0, standard_deviation)
            data[i, j, 0] *= np.exp(1j * phase_error_theta)
            data[i, j, 1] *= np.exp(1j * phase_error_phi)

def fixed_phase_error(data, error):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error_theta = random.choice([-1, 1]) * error
            phase_error_phi = random.choice([-1, 1]) * error
            data[i, j, 0] *= np.exp(1j * phase_error_theta)
            data[i, j, 1] *= np.exp(1j * phase_error_phi)