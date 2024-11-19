import numpy as np

def sum_NF_poles(nfData):
    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    return (abs(nfData[:, :, 0])**2 + abs(nfData[:, :, 1])**2)**0.5



def zero_pad_theta(nfData, theta_deg, theta_step_deg):
    
    num_zero_rows = len(theta_deg) - nfData.shape[0]
    if (num_zero_rows == 0):
        return nfData

    # Create zero rows with the same number of columns as the original array
    zero_rows = np.full((num_zero_rows, nfData.shape[1]), np.min(nfData))

    # Add the zero rows to the array
    return np.vstack((nfData, zero_rows))

    