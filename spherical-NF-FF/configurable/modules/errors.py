import numpy as np
import random

# Function to introduce amplitude errors (modifies data in place)
def amplitude_errors_normal(data, standard_deviation):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply amplitude error to both components (E_theta and E_phi)
            amplitude_error_theta = 1 + np.random.normal(0, standard_deviation)
            amplitude_error_phi = 1 + np.random.normal(0, standard_deviation)
            data[i, j, 0] *= abs(amplitude_error_theta)
            data[i, j, 1] *= abs(amplitude_error_phi)

# Function to introduce phase errors (modifies data in place)
def phase_errors_normal(data, standard_deviation):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error_theta = 2*np.pi * np.random.normal(0, standard_deviation)
            phase_error_phi = 2*np.pi * np.random.normal(0, standard_deviation)
            data[i, j, 0] *= np.exp(1j * phase_error_theta)
            data[i, j, 1] *= np.exp(1j * phase_error_phi)

# Function to introduce amplitude errors (modifies data in place)
def amplitude_errors_uniform(data, max_error):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply amplitude error to both components (E_theta and E_phi)
            amplitude_error_theta = np.random.uniform(1-max_error, 1+max_error)
            amplitude_error_phi = np.random.uniform(1-max_error, 1+max_error)
            data[i, j, 0] *= abs(amplitude_error_theta)
            data[i, j, 1] *= abs(amplitude_error_phi)

# Function to introduce phase errors (modifies data in place)
def phase_errors_uniform(data, max_error):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error_theta = 2*np.pi * np.random.uniform(1-max_error, 1+max_error)
            phase_error_phi = 2*np.pi * np.random.uniform(1-max_error, 1+max_error)
            data[i, j, 0] *= np.exp(1j * phase_error_theta)
            data[i, j, 1] *= np.exp(1j * phase_error_phi)

# Function to introduce amplitude errors (modifies data in place)
def amplitude_same_errors_normal(data, standard_deviation):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply amplitude error to both components (E_theta and E_phi)
            amplitude_error = 1 + np.random.normal(0, standard_deviation)
            data[i, j, 0] *= abs(amplitude_error)
            data[i, j, 1] *= abs(amplitude_error)

# Function to introduce phase errors (modifies data in place)
def phase_same_errors_normal(data, standard_deviation):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error = 2*np.pi * np.random.normal(0, standard_deviation)
            data[i, j, 0] *= np.exp(1j * phase_error)
            data[i, j, 1] *= np.exp(1j * phase_error)

# Function to introduce amplitude errors (modifies data in place)
def amplitude_same_errors_uniform(data, max_error):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply amplitude error to both components (E_theta and E_phi)
            amplitude_error = np.random.uniform(1-max_error, 1+max_error)
            data[i, j, 0] *= abs(amplitude_error)
            data[i, j, 1] *= abs(amplitude_error)

# Function to introduce phase errors (modifies data in place)
def phase_same_errors_uniform(data, max_error):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error = 2*np.pi * np.random.uniform(1-max_error, 1+max_error)
            data[i, j, 0] *= np.exp(1j * phase_error)
            data[i, j, 1] *= np.exp(1j * phase_error)

def phase_errors_correlated(data, deviation_factor, max_error):
    if(max_error > 1):
        raise ValueError(f"To large value for max_error, this value may not exceed 1!")
    if(deviation_factor > 1):
        raise ValueError(f"To large value for deviation_factor, this value may not exceed 1!")
    max = 1 + max_error
    min = 1 - max_error
    error = 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            upOrDown = np.random.randint(0,2)
            if (upOrDown == 1):
                error += (max-error) * deviation_factor
            else:
                error += (min-error) * deviation_factor
            # Apply phase error to both components (E_theta and E_phi)
            data[i, j, 0] *= np.exp(1j * error)
            data[i, j, 1] *= np.exp(1j * error)

def amplitude_errors_correlated(data, deviation_factor, max_error):
    if(max_error >= 1):
        raise ValueError(f"To large value for max_error, this value may not be equal to or exceed 1!")
    if(deviation_factor >= 1):
        raise ValueError(f"To large value for deviation_factor, this value may not be equal to or exceed 1!")
    max = 1 + max_error
    min = 1 - max_error
    error = 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            upOrDown = np.random.randint(0,2)
            if (upOrDown == 1):
                error += (max-error) * deviation_factor
            else:
                error += (min-error) * deviation_factor
            # Apply phase error to both components (E_theta and E_phi)
            data[i, j, 0] *= error
            data[i, j, 1] *= error

def phase_errors_correlated_rev(data, deviation_factor, max_error):
    if(max_error > 1):
        raise ValueError(f"To large value for max_error, this value may not exceed 1!")
    if(deviation_factor > 1):
        raise ValueError(f"To large value for deviation_factor, this value may not exceed 1!")
    max = 1 + max_error
    min = 1 - max_error
    error = 1
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            upOrDown = np.random.randint(0,2)
            if (upOrDown == 1):
                error += (max-error) * deviation_factor
            else:
                error += (min-error) * deviation_factor
            # Apply phase error to both components (E_theta and E_phi)
            data[i, j, 0] *= np.exp(1j * error)
            data[i, j, 1] *= np.exp(1j * error)

def amplitude_errors_correlated_rev(data, deviation_factor, max_error):
    if(max_error >= 1):
        raise ValueError(f"To large value for max_error, this value may not be equal to or exceed 1!")
    if(deviation_factor >= 1):
        raise ValueError(f"To large value for deviation_factor, this value may not be equal to or exceed 1!")
    max = 1 + max_error
    min = 1 - max_error
    error = 1
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            upOrDown = np.random.randint(0,2)
            if (upOrDown == 1):
                error += (max-error) * deviation_factor
            else:
                error += (min-error) * deviation_factor
            # Apply phase error to both components (E_theta and E_phi)
            data[i, j, 0] *= error
            data[i, j, 1] *= error


def fixed_phase_error(data, error):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Apply phase error to both components (E_theta and E_phi)
            phase_error_theta = random.choice([-1, 1]) * error
            phase_error_phi = random.choice([-1, 1]) * error
            data[i, j, 0] *= np.exp(1j * phase_error_theta)
            data[i, j, 1] *= np.exp(1j * phase_error_phi)

def combine_data_for_position_error(dataArr):

    if len(dataArr) < 2:
        raise ValueError(f"Not possible to combine data from single data set")
    
    # Stack the arrays along a new axis to simplify random selection
    stacked_arrays = np.stack(dataArr)  # Shape: (3, rows, cols)

    # Generate a random index array to choose from the stacked arrays
    rows, cols, _ = dataArr[0].shape

    random_indices = np.random.randint(0, len(dataArr), size=(rows, cols))  # Random indices (0, 1, or 2)

    # Use advanced indexing to select random values from the stacked arrays
    output_array = stacked_arrays[random_indices, np.arange(rows)[:, None], np.arange(cols)]

    return output_array

def get_theta_phi_error_from_fine_set(array, new_shape, sample_theta=True, sample_phi=True):
    """
    Reduces the resolution of a 3D array along the first two axes by random sampling.
    
    Parameters:
        array (numpy.ndarray): Input 3D array with shape (d1, d2, d3).
        new_shape (tuple): Desired shape for the first two axes (new_d1, new_d2).
        sample_axis1 (bool): If True, samples randomly along axis 1.
        sample_axis2 (bool): If True, samples randomly along axis 2.
    
    Returns:
        numpy.ndarray: Output 3D array with reduced resolution.
        new_theta_values: Output a 1D array with correct theta resolution
        new_phi_values: Output a 1D array with correct phi resolution
        new_theta_stepSize: Output correct theta step size
        new_phi_stepSize: Output correct phi step size
    """

    d1, d2, d3 = array.shape
    new_d1, new_d2 = new_shape

    # if not (sample_theta or sample_phi):
    #     raise ValueError("At least one sampling mode (sample_theta or sample_phi) must be True.")

    # Define output array
    output = np.zeros((new_d1, new_d2, d3), dtype=array.dtype)

    # Generate new coordinates for sampling
    for i in range(new_d1):
        for j in range(new_d2):
            # Determine the sampling range in the original array
            start_i = (i * d1) // new_d1
            end_i = ((i + 1) * d1) // new_d1
            start_j = (j * d2) // new_d2
            end_j = ((j + 1) * d2) // new_d2

            # Sampling logic
            sampled_i = np.random.randint(start_i, end_i) if sample_theta else (start_i + end_i) // 2
            sampled_j = np.random.randint(start_j, end_j) if sample_phi else (start_j + end_j) // 2

            # Assign value to output
            output[i, j] = array[sampled_i, sampled_j]

    # determine new theta,phi values and stepsizes
    new_theta_values, new_theta_stepSize = np.linspace(0, 180, new_d1, retstep=True, endpoint=False)
    new_phi_values, new_phi_stepSize = np.linspace(0, 360, new_d2, retstep=True, endpoint=False)

    return output, new_theta_values, new_phi_values, new_theta_stepSize, new_phi_stepSize
