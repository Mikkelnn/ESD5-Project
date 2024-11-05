import numpy as np

def simulate_NF_dipole_array(frequency = 8e9, radius_nf = 0.2, wavelength_spacing_factor = 0.5, num_dipoles = 9, num_samples_theta = 360, num_samples_phi = 360):
    """
    Simulate near-field data for a dipole antenna array.
    
    Returns:
    - NON-normalized tuple of E-field magnitudes in the components: (theta, phi) 
    
    Parameters:
    - frequency: Frequency in Hz (8 GHz)
    - radius_nf: Radius in meters for near field (20 cm)
    - wavelength_spacing_factor: Spacing factor of wavelength between dipoles (1/2 wavelength spacing)
    - num_dipoles: Number of dipoles in the array
    - num_samples_theta: Number of samples for theta (0 to π)
    - num_samples_phi: Number of samples for phi (0 to 2π)
    """

    # Parameters
    wavelength = 3e8 / frequency  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wave number
    spacing = wavelength * wavelength_spacing_factor # Spacing between dipoles in meters

    # Centering the dipoles around the origin on the x-axis
    dipole_positions = (np.arange(num_dipoles) - num_dipoles // 2) * spacing  # x-coordinates of dipoles

    # Define theta and phi ranges for spherical coordinates
    theta = np.linspace(0, np.pi, num_samples_theta)  # Polar angle
    phi = np.linspace(0, 2 * np.pi, num_samples_phi)  # Azimuthal angle

    # Create meshgrid for spherical coordinates
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Initialize the total electric fields in the near field
    E_theta_total_nf = np.zeros_like(theta_grid, dtype=np.complex_)
    E_phi_total_nf = np.zeros_like(theta_grid, dtype=np.complex_)

    # Calculate the total electric fields from each dipole in the near field
    for pos in dipole_positions:
        for i in range(num_samples_theta):
            for j in range(num_samples_phi):
                # Calculate the distance to the observation point for each dipole in the near field
                observation_x = radius_nf * np.sin(theta_grid[i, j]) * np.cos(phi_grid[i, j])  # x-coordinate
                observation_y = radius_nf * np.sin(theta_grid[i, j]) * np.sin(phi_grid[i, j])  # y-coordinate
                observation_z = radius_nf * np.cos(theta_grid[i, j])  # z-coordinate
                distance = np.sqrt((observation_x - pos) ** 2 + observation_y ** 2 + observation_z ** 2)  # distance to each dipole
                
                # Calculate electric field contributions in the near field
                E_theta_nf = (1j * k * np.exp(-1j * k * distance) / (2 * np.pi * distance)) * np.sin(theta_grid[i, j])
                E_phi_nf = (1j * k * np.exp(-1j * k * distance) / (2 * np.pi * distance)) * np.cos(theta_grid[i, j])

                # Sum the contributions to the total electric field in the near field
                E_theta_total_nf[i, j] += E_theta_nf
                E_phi_total_nf[i, j] += E_phi_nf

    # Convert complex fields to magnitude (near field)
    E_theta_mag_nf = np.abs(E_theta_total_nf)
    E_phi_mag_nf = np.abs(E_phi_total_nf)

    return (E_theta_mag_nf, E_phi_mag_nf)

