import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

samples = 100

# Sample near-field measurement data (example)
# Replace these with your actual measurements
#amplitude = np.random.rand(samples, samples)  # Example amplitude data
#x = np.linspace(1, samples, samples)  # X-coordinates (meters)
#y = np.linspace(1, samples, samples)  # Y-coordinates (meters)



# Calculate wavenumber (k) in the far-field region
wavelength = 0.1  # Assume wavelength (adjust as per actual data, e.g., 10 cm)
k = 2 * np.pi / wavelength  # Wavenumber
z_distance = 1  # Distance from AUT (meters)
dipole_length = wavelength / 2  # Half-wave dipole

# Define a 2D grid for near-field data in x-y plane
x = np.linspace(-0.5, 0.5, samples)  # X-coordinates (meters)
y = np.linspace(-0.5, 0.5, samples)  # Y-coordinates (meters)
X, Y = np.meshgrid(x, y)

# Convert (X, Y, Z) to spherical coordinates to get theta, phi
R = np.sqrt(X**2 + Y**2 + z_distance**2)  # Distance from dipole
theta = np.arctan2(np.sqrt(X**2 + Y**2), z_distance)  # Elevation angle theta
phi = np.arctan2(Y, X)  # Azimuth angle phi

# Compute the near-field magnitude based on dipole radiation pattern
# For simplicity, assume E(theta) = sin(theta) for far-field dipole pattern
# Apply a near-field modification based on 1/R decay
near_field_amplitude = np.sin(theta) / R  # Dipole far-field modified by 1/R for near-field

amplitude = near_field_amplitude

# Create meshgrid for x and y
X, Y = np.meshgrid(x, y)

# Perform 2D FFT on the amplitude data
fft_near_field = fft2(amplitude)
fft_near_field_shifted = fftshift(fft_near_field)  # Shift FFT result for easier plotting

# Create far-field coordinates (theta, phi)
theta = np.linspace(-np.pi/2, np.pi/2, samples)  # Elevation angles
phi = np.linspace(0, 2*np.pi, samples)  # Azimuth angles

# Far-field radiation pattern (magnitude)
far_field_magnitude = np.abs(fft_near_field_shifted)

# Create meshgrid for theta and phi for full far-field pattern
THETA, PHI = np.meshgrid(theta, phi)

# ** E-Plane (phi = 0) Radiation Pattern ** #
# E-plane is the radiation pattern at phi = 0
e_plane_magnitude = far_field_magnitude[:, int(samples/2)]  # Picking a slice at phi = 0 (or close to it)

# ** H-Plane (theta = 0) Radiation Pattern ** #
# H-plane is the radiation pattern at theta = 0
h_plane_magnitude = far_field_magnitude[int(samples/2), :]  # Picking a slice at theta = 0 (or close to it)

# Now plot the full 2D radiation pattern

# E-plane polar plot
plt.figure(figsize=(12, 6))
plt.subplot(121, projection='polar')
plt.polar(theta, e_plane_magnitude)
plt.title('E-plane Radiation Pattern (Phi = 0)')

# H-plane polar plot
plt.subplot(122, projection='polar')
plt.polar(phi, h_plane_magnitude)
plt.title('H-plane Radiation Pattern (Theta = 0)')
plt.show()

exit()

# Plot func:

from mpl_toolkits.mplot3d import Axes3D

# Create a spherical plot in the far-field region
THETA, PHI = np.meshgrid(theta, phi)
R = far_field_magnitude  # Using FFT results for far-field magnitude

# Convert spherical to Cartesian coordinates for 3D plot
X_3D = R * np.sin(THETA) * np.cos(PHI)
Y_3D = R * np.sin(THETA) * np.sin(PHI)
Z_3D = R * np.cos(THETA)

# Plotting the far-field pattern in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_3D, Y_3D, Z_3D, cmap='viridis')
plt.title("3D Far-Field Radiation Pattern")
plt.show()
