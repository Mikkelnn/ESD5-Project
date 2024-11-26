import numpy as np

def sum_NF_poles(nfData):
    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    # return (abs(nfData[:, :, 0])**2 + abs(nfData[:, :, 1])**2)**0.5
    return abs(nfData[:, :, 0]) + abs(nfData[:, :, 1])

def sum_NF_poles_sqrt(nfData):
    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    return (abs(nfData[:, :, 0])**2 + abs(nfData[:, :, 1])**2)**0.5
    #return abs(nfData[:, :, 0]) + abs(nfData[:, :, 1])

def HansenPreProcessing(nfData): # Implementation of eq 4.126
    nfDataNew = np.zeros((nfData.shape[0], nfData.shape[1], 2), dtype= complex)
    nfDataNew[:, :, 0] = (nfData[:, :, 0] - 1j*nfData[:, :, 1]) * (1./2.)   #+1
    nfDataNew[:, :, 1] = (nfData[:, :, 0] - 1j*nfData[:, :, 1]) * (1./2.)     #-1

    #nfDataNew = np.pad(nfDataNew, ((0,36-nfDataNew.shape[0]), (0,0), (0, 0)), mode='constant', constant_values=0) # this only works if the step size is 5 degrees

    return nfDataNew

def zero_pad_theta(nfData, theta_deg, theta_step_deg):
    
    num_zero_rows = len(theta_deg) - nfData.shape[0]
    if (num_zero_rows == 0):
        return nfData

    # Create zero rows with the same number of columns as the original array
    zero_rows = np.full((num_zero_rows, nfData.shape[1]), np.min(nfData))

    # Add the zero rows to the array
    return np.vstack((nfData, zero_rows))

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

