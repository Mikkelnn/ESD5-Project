import numpy as np
from scipy.special import sph_harm

def normalize_near_field_data(nf_data):
    """
    Normalize the near-field data by scaling E_theta and E_phi to have a maximum value of 1.
    """
    # Extract components
    E_theta = nf_data[:, 2]
    E_phi = nf_data[:, 3]
    
    # Normalize E_theta and E_phi magnitudes
    E_theta /= np.max(np.abs(E_theta))
    E_phi /= np.max(np.abs(E_phi))
    
    # Update the nf_data array
    nf_data[:, 2] = E_theta
    nf_data[:, 3] = E_phi

    return nf_data

def spherical_far_field_transform(nf_data, max_l):
    """
    Compute and return the normalized far-field pattern from the near-field data.
    The sizes and ranges of theta and phi are determined from the nf_data array.
    """
    # Determine theta and phi sizes from the nf_data shape
    num_points = nf_data.shape[0]
    theta_size = int(np.sqrt(num_points // 2))
    phi_size = 2 * theta_size

    # Define theta and phi ranges for far-field computation
    theta_f = np.linspace(0, np.pi, theta_size)  # Far-field theta range
    phi_f = np.linspace(0, 2 * np.pi, phi_size)  # Far-field phi range

    # Normalize the near-field data first
    nf_data = normalize_near_field_data(nf_data)
    
    # Extract theta, phi, and electric field components
    theta = nf_data[:, 0]
    phi = nf_data[:, 1]
    E_theta = nf_data[:, 2]
    E_phi = nf_data[:, 3]
    
    # Compute spherical harmonic coefficients a_lm
    a_lm = np.zeros((max_l + 1, 2 * max_l + 1), dtype=complex)
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            a_lm[l, m + l] = np.sum(E_theta * Y_lm)  # Use E_theta only for simplicity

    # Calculate the far-field pattern from coefficients
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            # Create a meshgrid for the far-field angles
            phi_f_grid, theta_f_grid = np.meshgrid(phi_f, theta_f, indexing='ij')
            Y_lm_f = sph_harm(m, l, phi_f_grid, theta_f_grid)
            E_far += a_lm[l, m + l] * Y_lm_f

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    return E_far_magnitude