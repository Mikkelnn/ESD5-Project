import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

# Function to simulate near-field measurements and plot far-field patterns
def simulate_dipole_far_field(plane_size, num_points=100, z_distance=1, wavelength=0.1):
    """
    Simulate near-field data for a dipole antenna and transform to far-field.
    
    Parameters:
    - plane_size: Size of the near-field sampling plane (meters)
    - num_points: Number of sampling points along each axis
    - z_distance: Distance from the dipole antenna (near-field distance)
    - wavelength: Wavelength of the signal (meters)
    """
    # Constants
    k = 2 * np.pi / wavelength  # Wavenumber

    # Define a 2D grid for near-field data in x-y plane
    x = np.linspace(-plane_size / 2, plane_size / 2, num_points)  # X-coordinates (meters)
    y = np.linspace(-plane_size / 2, plane_size / 2, num_points)  # Y-coordinates (meters)
    X, Y = np.meshgrid(x, y)

    # Convert (X, Y, Z) to spherical coordinates to get theta, phi
    R = np.sqrt(X**2 + Y**2 + z_distance**2)  # Distance from dipole
    theta = np.arctan2(np.sqrt(X**2 + Y**2), z_distance)  # Elevation angle theta
    phi = np.arctan2(Y, X)  # Azimuth angle phi

    # Compute the near-field magnitude based on dipole radiation pattern
    near_field_amplitude = np.sin(theta) / R  # Dipole far-field modified by 1/R

    # Debug: Plot the near-field amplitude to check if it's symmetric
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, near_field_amplitude, 100, cmap='inferno')
    plt.colorbar(label="Near-field Amplitude (Dipole)")
    plt.title("Simulated Near-Field Amplitude of a Dipole Antenna")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()

    # Step 2: Apply Near-Field to Far-Field Transformation (FFT)
    fft_near_field = fft2(near_field_amplitude)
    fft_near_field_shifted = fftshift(fft_near_field)

    # Calculate far-field magnitude (normalize by the max value to prevent large scaling issues)
    far_field_magnitude = np.abs(fft_near_field_shifted)
    far_field_magnitude /= np.max(far_field_magnitude)  # Normalize

    # Step 3: Extract the E-Plane and H-Plane Radiation Patterns

    # Define angle ranges
    theta_range = np.linspace(-np.pi/2, np.pi/2, num_points)  # Elevation angle range
    phi_range = np.linspace(0, 2*np.pi, num_points)  # Azimuth angle range

    # E-plane: when phi = 0 (azimuth is constant)
    e_plane_magnitude = far_field_magnitude[:, num_points // 2]  # Cut at phi = 0

    # H-plane: when theta = 0 (elevation is constant)
    h_plane_magnitude = far_field_magnitude[num_points // 2, :]  # Cut at theta = 0

    # Debug: Plot the far-field magnitude to check if it's symmetric
    plt.figure(figsize=(6, 5))
    plt.contourf(far_field_magnitude, 100, cmap='viridis')
    plt.colorbar(label="Far-field Magnitude (normalized)")
    plt.title("Far-field Magnitude after FFT (Dipole)")
    plt.grid(True)
    plt.show()

    # Step 4: Plot Full 2D Polar Plots for E-Plane and H-Plane

    # E-plane polar plot
    plt.figure(figsize=(10, 5))

    plt.subplot(121, projection='polar')
    plt.polar(theta_range, e_plane_magnitude)
    plt.title('E-Plane')

    # H-plane polar plot
    plt.subplot(122, projection='polar')
    plt.polar(phi_range, h_plane_magnitude)
    plt.title('H-Plane')

    plt.show()

# Example usage: Simulate with a specific plane size
plane_size = 1.0  # Size of the near-field sampling plane in meters
simulate_dipole_far_field(plane_size)
