import numpy as np
import matplotlib.pyplot as plt

# Constants
wavelength = 0.1  # Assume a wavelength of 10 cm
k = 2 * np.pi / wavelength  # Wavenumber
z_distance = 1  # Distance from the dipole antenna (near-field distance)
dipole_length = wavelength / 2  # Half-wave dipole

# Define a 2D grid for near-field data in x-y plane
x = np.linspace(-0.5, 0.5, 100)  # X-coordinates (meters)
y = np.linspace(-0.5, 0.5, 100)  # Y-coordinates (meters)
X, Y = np.meshgrid(x, y)

# Convert (X, Y, Z) to spherical coordinates to get theta, phi
R = np.sqrt(X**2 + Y**2 + z_distance**2)  # Distance from dipole
theta = np.arctan2(np.sqrt(X**2 + Y**2), z_distance)  # Elevation angle theta
phi = np.arctan2(Y, X)  # Azimuth angle phi

# Compute the near-field magnitude based on dipole radiation pattern
# For simplicity, assume E(theta) = sin(theta) for far-field dipole pattern
# Apply a near-field modification based on 1/R decay
near_field_amplitude = np.sin(theta) / R  # Dipole far-field modified by 1/R for near-field

print(near_field_amplitude)

# Plot the simulated near-field data (amplitude)
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, near_field_amplitude, 100, cmap='inferno')
plt.colorbar(label="Near-field Amplitude (Dipole)")
plt.title("Simulated Near-Field Amplitude of a Dipole Antenna")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.show()

# This "near_field_amplitude" can be used as input to the transformation algorithm
