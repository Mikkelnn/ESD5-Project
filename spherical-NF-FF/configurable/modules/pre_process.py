import numpy as np

def HansenPreProcessing(nfData): # Implementation of eq 4.126
    nfDataNew = np.zeros((nfData.shape[0], nfData.shape[1], 2), dtype= complex)
    nfDataNew[:, :, 0] = (nfData[:, :, 0] - 1j*nfData[:, :, 1]) * (1./2.)   #+1
    nfDataNew[:, :, 1] = (nfData[:, :, 0] + 1j*nfData[:, :, 1]) * (1./2.)   #-1

    #nfDataNew = np.pad(nfDataNew, ((0,36-nfDataNew.shape[0]), (0,0), (0, 0)), mode='constant', constant_values=0) # this only works if the step size is 5 degrees

    return nfDataNew

def zero_pad_theta(nfData, theta_step_deg):

    count = int(180 / theta_step_deg) + 1
    full_theta_range = np.linspace(0, 180, count)

    num_zero_rows = len(full_theta_range) - nfData.shape[0]
    # num_zero_rows = len(theta_deg) - nfData.shape[0]
    if (num_zero_rows == 0):
        return nfData

    # Create zero rows with the same number of columns as the original array
    shape = list(nfData.shape)
    shape[0] = num_zero_rows
    zero_rows = np.full(tuple(shape), np.min(nfData))

    # Add the zero rows to the array
    result = np.vstack((nfData, zero_rows))
    return result, full_theta_range

