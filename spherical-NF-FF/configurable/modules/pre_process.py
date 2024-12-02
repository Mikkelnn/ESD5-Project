import numpy as np

def HansenPreProcessing(nfData): # Implementation of eq 4.126
    nfDataNew = np.zeros((nfData.shape[0], nfData.shape[1], 2), dtype= complex)
    nfDataNew[:, :, 0] = (nfData[:, :, 0] - 1j*nfData[:, :, 1]) * (1./2.)   #+1
    nfDataNew[:, :, 1] = (nfData[:, :, 0] + 1j*nfData[:, :, 1]) * (1./2.)     #-1

    #nfDataNew = np.pad(nfDataNew, ((0,36-nfDataNew.shape[0]), (0,0), (0, 0)), mode='constant', constant_values=0) # this only works if the step size is 5 degrees

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

