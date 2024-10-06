import numpy as np
import matplotlib.pyplot as plt

def simulate_near_field_dipole(antenna_size, wavelength, plane_size, z_distance, num_points=500):
    k = 2 * np.pi / wavelength  # Wavenumber

    # Define a 2D grid for the measurement plane (in y-z plane at x = z_distance)
    y = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    z = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    Y, Z = np.meshgrid(y, z)

    # Coordinates of the dipole's center
    dipole_top = np.array([0, 0, antenna_size / 2])
    dipole_bottom = np.array([0, 0, -(antenna_size / 2)])

    # Calculate distance R from dipole to each point on the measurement plane
    R_top = np.sqrt((z_distance - dipole_top[0])**2 + (Y - dipole_top[1])**2 + (Z - dipole_top[2])**2)
    R_bottom = np.sqrt((z_distance - dipole_bottom[0])**2 + (Y - dipole_bottom[1])**2 + (Z - dipole_bottom[2])**2)

    # Electric field components (simplified near-field model)
    E_top = np.sin(np.arctan2(np.sqrt(Y**2 + Z**2), z_distance)) / R_top * np.exp(-1j * k * R_top)
    E_bottom = np.sin(np.arctan2(np.sqrt(Y**2 + Z**2), z_distance)) / R_bottom * np.exp(-1j * k * R_bottom)

    # Total near-field data (superposition of the fields from the top and bottom of the dipole)
    near_field = E_top + E_bottom
    
    return near_field, Y, Z

def nf_ff_transform(near_field, wavelength, plane_size):
    num_points = near_field.shape[0]
    k = 2 * np.pi / wavelength  # Wavenumber
    
    # Spatial frequency grid in the plane (for Fourier transform)
    delta_yz = plane_size / num_points  # Spatial resolution in the y and z directions
    ky = np.fft.fftfreq(num_points, d=delta_yz) * 2 * np.pi
    kz = np.fft.fftfreq(num_points, d=delta_yz) * 2 * np.pi
    KY, KZ = np.meshgrid(ky, kz)
    
    # Perform 2D Fourier transform (with shift to center frequencies)
    far_field_2D = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(near_field)))
    
    # Calculate corresponding far-field angles based on wavenumber
    theta_far = np.arctan2(np.sqrt(KY**2 + KZ**2), k)  # Far-field angle (theta)

    # Normalize the far-field pattern
    far_field_pattern = np.abs(far_field_2D) / np.max(np.abs(far_field_2D))
    
    return far_field_pattern, theta_far, ky, kz

def plot_far_field_heatmap(far_field_pattern, theta_far, ky, kz):
    """
    Plot the far-field pattern as a 2D heat map in the translated plane.
    
    Parameters:
    - far_field_pattern: The far-field amplitude data (2D array).
    - theta_far: The far-field angles in the elevation plane (in radians).
    - ky: Spatial frequency corresponding to the y direction.
    - kz: Spatial frequency corresponding to the z direction.
    """
    # Create a meshgrid for KY and KZ
    KY, KZ = np.meshgrid(ky, kz)

    # Normalize the far-field data for heatmap
    far_field_normalized = far_field_pattern / np.max(far_field_pattern)

    # Plot the heatmap using imshow
    plt.figure(figsize=(8, 6))
    plt.imshow(far_field_normalized, extent=[-1, 1, -1, 1], cmap='hot', aspect='auto')

    plt.colorbar(label='Far-field amplitude (normalized)')
    plt.title('Far-Field Radiation Pattern Heatmap')
    plt.xlabel('K_Y (1/m)')
    plt.ylabel('K_Z (1/m)')
    
    plt.show()

# Example usage
antenna_size = 5   # Largest dimension of the antenna in meters
wavelength = 10    # Wavelength in meters
plane_size = 10    # Size of the measurement plane in meters
num_points = 500   # Number of point along each axis
z_distance = 3.5   # Distance from the antenna to the measurement plane

# Step 1: Simulate the near-field data
near_field, Y, Z = simulate_near_field_dipole(antenna_size, wavelength, plane_size, z_distance, num_points)

# Step 2: Perform the near-field to far-field transformation
far_field_pattern, theta_far, ky, kz = nf_ff_transform(near_field, wavelength, plane_size)

# Step 3: Plot the far-field radiation pattern heatmap
plot_far_field_heatmap(far_field_pattern, theta_far, ky, kz)
