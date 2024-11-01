import numpy as np
import pandas as pd


# Function to load the data and return a two-dimensional array of complex numbers
def load_data_lab_measurements(file_path):
    # Load the data, skipping the non-data rows
    nf_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=13, header=None)
    
    # Calculate theta and phi sizes
    theta_size = int(np.sqrt(nf_data.shape[0] / (4 * 2.5)))
    phi_size = int(theta_size * 2 * 2.5)
    
    # Create meshgrid for spherical coordinates
    theta = np.linspace(0, np.pi, theta_size)
    phi = np.linspace(0, 2 * np.pi, phi_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Initialize the total electric fields in the near field as a single array
    complex_field_data = np.zeros((theta_size, phi_size, 2), dtype=np.complex_)

    # Fill in the electric field arrays with complex numbers
    k = 0
    for i in range(theta_size):
        for j in range(phi_size):
            # E_theta component
            complex_field_data[i, j, 0] = nf_data.iloc[k, 3] + 1j * nf_data.iloc[k, 4]
            # E_phi component
            complex_field_data[i, j, 1] = (
                nf_data.iloc[k + int(nf_data.shape[0] / 2), 3] +
                1j * nf_data.iloc[k + int(nf_data.shape[0] / 2), 4]
            )
            k += 1

    return complex_field_data



#Newtons method for square root, used to get the square root of integers.
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def load_data_cst(file_path):
    # Load the data, skipping the row with dashes (assumed to be the second row)
    nf_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, header=None)

    # Calculate theta and phi sizes
    theta_size = int(isqrt(nf_data.shape[0] // 2))
    phi_size = theta_size * 2

    # Define theta and phi ranges for spherical coordinates
    theta = np.linspace(0, np.pi, theta_size)  # Polar angle
    phi = np.linspace(0, 2 * np.pi, phi_size)  # Azimuthal angle

    # Create meshgrid for spherical coordinates
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Initialize the total electric fields in the near field
    complex_field_data = np.zeros((theta_size, phi_size, 2), dtype=np.complex_)

    # Fill in the electric field arrays with complex numbers
    k = 0
    for i in range(theta_size):
        for j in range(phi_size):
            # Compute E_theta and E_phi as complex numbers
            e_theta_magnitude = nf_data.iloc[k, 3]
            e_theta_phase = nf_data.iloc[k, 4] * 2 * np.pi / 360  # Convert degrees to radians
            e_phi_magnitude = nf_data.iloc[k, 5]
            e_phi_phase = nf_data.iloc[k, 6] * 2 * np.pi / 360  # Convert degrees to radians

            # Construct the complex electric field components
            complex_field_data[i, j, 0] = e_theta_magnitude * np.cos(e_theta_phase) + 1j * e_theta_magnitude * np.sin(e_theta_phase)
            complex_field_data[i, j, 1] = e_phi_magnitude * np.cos(e_phi_phase) + 1j * e_phi_magnitude * np.sin(e_phi_phase)
            k += 1

    return complex_field_data