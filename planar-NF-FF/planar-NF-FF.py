import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

def simulate_near_field_dipole(antenna_size, wavelength, plane_size, x_distance, num_points=100):
    """
    Simulate near-field data for a vertical dipole antenna in the new configuration.
    
    Parameters:
    - antenna_size: Size of the dipole antenna (length of dipole, meters)
    - wavelength: Wavelength of the signal (meters)
    - plane_size: Size of the measurement plane (in y and z directions, meters)
    - x_distance: Distance from the antenna to the measurement plane (meters)
    - num_points: Number of sampling points along each axis
    
    Returns:
    - near_field: The complex near-field data (amplitude and phase) on the measurement plane
    """
    # Wavenumber
    k = 2 * np.pi / wavelength
    
    # Define a 2D grid for the measurement plane (in y-z plane, at x = x_distance)
    y = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    z = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    Y, Z = np.meshgrid(y, z)
    
    # Coordinates of the dipole's center: (0, 0, 0) with endpoints at (0, 0, -0.5) and (0, 0, 0.5)
    dipole_top = np.array([0, 0, antenna_size/2])
    dipole_bottom = np.array([0, 0, -(antenna_size/2)])
    
    # Calculate distance R from the dipole to each point on the measurement plane
    R_top = np.sqrt((x_distance - dipole_top[0])**2 + (Y - dipole_top[1])**2 + (Z - dipole_top[2])**2)
    R_bottom = np.sqrt((x_distance - dipole_bottom[0])**2 + (Y - dipole_bottom[1])**2 + (Z - dipole_bottom[2])**2)
    
    # Calculate the elevation angle theta from the dipole axis to the measurement plane
    theta_top = np.arctan2(np.sqrt((Y - dipole_top[1])**2 + (Z - dipole_top[2])**2), x_distance)
    theta_bottom = np.arctan2(np.sqrt((Y - dipole_bottom[1])**2 + (Z - dipole_bottom[2])**2), x_distance)
    
    # Approximate dipole near-field amplitude (simplified dipole model)
    near_field_amplitude_top = np.sin(theta_top) / R_top  # Contribution from the top part of the dipole
    near_field_amplitude_bottom = np.sin(theta_bottom) / R_bottom  # Contribution from the bottom part
        
    # Calculate the phase using spherical wave propagation from both ends
    near_field_phase_top = np.exp(-1j * k * R_top)
    near_field_phase_bottom = np.exp(-1j * k * R_bottom)
    
    # Combine the complex near-fields from the top and bottom of the dipole
    near_field = (near_field_amplitude_top * near_field_phase_top) + (near_field_amplitude_bottom * near_field_phase_bottom)
    

    # Limit the far-field data to the valid angular range (based on antenna size and plane)
    inner_angle = np.arctan2((plane_size / 2), z_distance)  # Maximum valid angle from the plane
    angle_min = np.pi/2 - inner_angle
    angle_max = np.pi/2 + inner_angle

    #valid_indices = np.where(np.abs(np.linspace(-np.pi/2, np.pi/2, num_points)) <= theta_max)[0]
    
    # Define angle ranges
    angle_range = np.linspace(angle_min, angle_max, num_points)  # Elevation angle range
    #phi_range = np.linspace(0, 2*np.pi, num_points)  # Azimuth angle range
    
    # E-plane: when phi = 0 (azimuth is constant)
    #theta_limited = theta_range[valid_indices]  # Corresponding limited angles for plotting
    #phi_limited = phi_range[valid_indices]  # Corresponding limited angles for plotting

    return near_field, angle_range


def far_field_pattern_limited(near_field):
    """
    Perform near-field to far-field transformation limited to a specific angular range and plot the far-field pattern.
    
    Parameters:
    - near_field: Complex near-field data simulated on the measurement plane.
    
    Returns:
    - far_field_pattern: The far-field amplitude and phase data.
    """
    # Simulate far-field pattern from near-field using Fourier-like approach
    # Assume the far-field pattern is proportional to a Fourier transform of the near-field
    # far_field_amplitude = np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(near_field), axis=0)))
    # far_field_amplitude = np.abs(np.fft.fftshift(np.fft.fft2(near_field)))  # 2D FFT for full far-field
    
    far_field_amplitude_1 = np.abs(np.fft.fftshift(np.fft.fft(near_field, axis=0)))
    far_field_amplitude_1 /= np.max(far_field_amplitude_1)  # Normalize

    far_field_amplitude_2 = np.abs(np.fft.fftshift(np.fft.fft(near_field, axis=1)))
    far_field_amplitude_2 /= np.max(far_field_amplitude_2)  # Normalize

    return [far_field_amplitude_1, far_field_amplitude_2]


def plot_far_field_image(far_field_amplitude):
    """
    Plot the far-field radiation pattern as a grayscale image.
    
    Parameters:
    - far_field_amplitude: The far-field amplitude data.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(far_field_amplitude[0], cmap='gray', extent=(-90, 90, -90, 90))  # Set the extent to represent angles
    plt.colorbar(label='Normalized Amplitude')
    plt.title('Far-Field Radiation Pattern (Grayscale Image) axis=0')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Elevation Angle (degrees)')
    plt.axis('equal')
    plt.grid(False)  # Disable the grid for better visibility

    plt.subplot(122)
    plt.imshow(far_field_amplitude[1], cmap='gray', extent=(-90, 90, -90, 90))  # Set the extent to represent angles
    plt.colorbar(label='Normalized Amplitude')
    plt.title('Far-Field Radiation Pattern (Grayscale Image) axis=1')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Elevation Angle (degrees)')
    plt.axis('equal')
    plt.grid(False)  # Disable the grid for better visibility

    plt.show()


def plot_far_field(far_field_amplitude, theta_limited, phi_limited, valid_indices):
    """
    Plot the far-field radiation pattern in 2D (E-plane).
    
    Parameters:
    - far_field_amplitude: The far-field amplitude data.
    - theta_far: The corresponding angles for the far-field pattern.
    """

    e_plane_magnitude = far_field_amplitude[1][:, num_points // 2]  # Cut at phi = 0
    e_plane_magnitude_limited = e_plane_magnitude[valid_indices]  # Limit to valid angular range
    
    # H-plane: when theta = 0 (elevation is constant)
    h_plane_magnitude = far_field_amplitude[0][num_points // 2, :]  # Cut at theta = 0
    h_plane_magnitude_limited = h_plane_magnitude[valid_indices]  # Limit to valid angular range
    
    # Plot the limited E-Plane and H-Plane patterns
    plt.figure(figsize=(10, 5))
    
    # E-plane polar plot
    plt.subplot(121, projection='polar')
    plt.polar(theta_limited, e_plane_magnitude_limited)
    plt.title('E-Plane (Limited Angle Range)')
    
    # H-plane polar plot
    plt.subplot(122, projection='polar')
    plt.polar(phi_limited, h_plane_magnitude_limited)
    plt.title('H-Plane (Limited Angle Range)')
    
    plt.show()


def save_near_field_data_with_angles(near_field, filename, angle_range):
    """
    Save near-field data using the precomputed theta and phi angles from the simulation.
    
    Parameters:
    - near_field: The simulated near-field data (complex values, amplitude and phase).
    - filename: The name of the file to save the data.
    - theta: Array of theta (elevation) angles in radians.
    - phi: Array of phi (azimuthal) angles in radians.
    
    Format:
    theta (rad)   phi (rad)   Eabs (V/m)   Ethetaabs (V/m)   Ephiabs (V/m)
    """

    with open(filename, 'w') as f:
        # Iterate over the grid points and save the data
        for i in range(near_field.shape[0]):
            for j in range(near_field.shape[1]):
                # Get theta and phi directly from the provided arrays
                theta_val = angle_range[i]
                phi_val = angle_range[j]
                
                # Extract the near-field amplitude and components (split complex field)
                Eabs = np.abs(near_field[i, j])  # Field magnitude
                Ethetaabs = np.real(near_field[i, j])  # Simplified: assuming Etheta is the real part
                Ephiabs = np.imag(near_field[i, j])    # Simplified: assuming Ephi is the imaginary part
                
                # Save the values in the file (space-separated)
                f.write(f"{theta_val:.6f} {phi_val:.6f} {Eabs:.6f} {Ethetaabs:.6f} {Ephiabs:.6f}\n")


# Define parameters
antenna_size = 5   # Largest dimension of the antenna in meters
wavelength = 10     # Wavelength in meters
plane_size = 10     # Size of the measurement plane in meters
num_points = 100     # Number of point along each axis
z_distance = 3.5     # Distance from the antenna to the measurement plane

# Step 1: Simulate Near-Field Data
near_field, angle_range = simulate_near_field_dipole(antenna_size, wavelength, plane_size, z_distance, num_points)

#filename = "./NF-data.txt"
#save_near_field_data_with_angles(near_field, filename, angle_range)

# Step 2: Transform the near-field to far-field data
far_field_amplitude = far_field_pattern_limited(near_field)

# Step 3: Plot the far-field radiation pattern with the plane's center aligned to 0 degrees
plot_far_field_image(far_field_amplitude)

#plot_far_field(far_field_amplitude, theta_limited, phi_limited, valid_indices)




