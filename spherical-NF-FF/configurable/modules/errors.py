import numpy as np

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
            phase_error_theta = np.random.normal(0, standard_deviation)
            phase_error_phi = np.random.normal(0, standard_deviation)
            data[i, j, 0] *= np.exp(1j * phase_error_theta)
            data[i, j, 1] *= np.exp(1j * phase_error_phi)