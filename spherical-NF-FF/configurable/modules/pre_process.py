import numpy as np

def HansenPreProcessing(nfData): # Implementation of eq 4.126
    nfDataNew = np.zeros(nfData.shape, dtype= complex) # copy of data
    nfDataNew[:, :, 0] = (nfData[:, :, 0] - 1j * nfData[:, :, 1]) * 0.5   #+1
    nfDataNew[:, :, 1] = (nfData[:, :, 0] + 1j * nfData[:, :, 1]) * 0.5   #-1

    return nfDataNew

def pad_theta(nfData, theta_step_deg):
    # calculate the number of theta values from 0 -> pi
    count = int(180 / theta_step_deg) + 1
    full_theta_range = np.linspace(0, 180, count)

    # calculate the row padding count
    num_padding_rows = len(full_theta_range) - nfData.shape[0]
    if (num_padding_rows == 0):
        return nfData, full_theta_range, num_padding_rows

    # copy the last row as padding
    padding_rows = np.repeat([nfData[-1]], num_padding_rows, axis=0)

    # Add the zero rows to the array
    result = np.vstack((nfData, padding_rows))
    return result, full_theta_range, num_padding_rows

