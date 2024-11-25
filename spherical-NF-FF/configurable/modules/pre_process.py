import numpy as np

def sum_NF_poles(nfData):
    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    return (abs(nfData[:, :, 0])**2 + abs(nfData[:, :, 1])**2)**0.5

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
    zero_rows = np.full((num_zero_rows, nfData.shape[1]), 0)

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

