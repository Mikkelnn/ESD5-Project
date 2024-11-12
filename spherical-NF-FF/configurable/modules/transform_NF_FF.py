import numpy as np
from scipy.special import sph_harm

def normalize_near_field_data(nf_data):
    """
    Normalize the near-field data by scaling E_theta and E_phi to have a maximum value of 1.
    """

    # Extract components
    E_theta = nf_data[:, :, 0]
    E_phi = nf_data[:, :, 1]
    
    # Normalize E_theta and E_phi magnitudes
    E_theta /= np.max(np.abs(E_theta))
    E_phi /= np.max(np.abs(E_phi))
    
    # Update the nf_data array
    nf_data[:, :, 0] = E_theta
    nf_data[:, :, 1] = E_phi

    return nf_data

def spherical_far_field_transform(nf_data, theta_f, phi_f, max_l):
    """
    Compute and return the normalized far-field pattern from the near-field data.
    The sizes and ranges of theta and phi are determined from the nf_data array.
    """

    # Normalize the near-field data first
    #nf_data = normalize_near_field_data(nf_data)    
    #nf_data = np.abs(np.real(nf_data))

    # Create a meshgrid for the far-field angles
    #phi_f_grid, theta_f_grid = np.meshgrid(phi_f, theta_f, indexing='ij')
    phi_f_grid, theta_f_grid = np.meshgrid(theta_f, phi_f, indexing='ij')

    # Extract theta, phi, and electric field components
    E_theta = nf_data[:, :, 0].flatten()
    E_phi = nf_data[:, :, 1].flatten()

    # Compute spherical harmonic coefficients a_lm
    a_lm = np.zeros((max_l + 1, 2 * max_l + 1), dtype=complex)
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, theta_f_grid.flatten(), phi_f_grid.flatten())
            Y_lm_conj = np.conjugate(Y_lm)
            a_lm[l, m + l] = np.sum(E_theta * Y_lm_conj)  # Use E_theta only for simplicity

    # Calculate the far-field pattern from coefficients
    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm_f = sph_harm(m, l, phi_f_grid, theta_f_grid)
            E_far += a_lm[l, m + l] * Y_lm_f

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    E_far_roll = np.roll(E_far_magnitude, int(E_far_magnitude.shape[1] // 2), axis=1)
    #E_far_roll = np.roll(E_far_roll, int(E_far_roll.shape[0] // 2), axis=0)

    return E_far_roll