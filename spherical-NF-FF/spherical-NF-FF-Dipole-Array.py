import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# Constants
frequency = 8e9  # Frequency in Hz (8 GHz)
wavelength = 3e8 / frequency  # Wavelength in meters
k = 2 * np.pi / wavelength  # Wave number
radius_nf = 0.2  # Radius in meters for near field (20 cm)

# Parameters
num_dipoles = 9  # Number of dipoles in the array
spacing = wavelength / 2  # Spacing between dipoles in meters
num_samples_theta = 360  # Number of samples for theta (0 to π)
num_samples_phi = 360  # Number of samples for phi (0 to 2π)

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

# Normalize the near-field electric field magnitudes
E_theta_mag_nf /= np.max(E_theta_mag_nf)
E_phi_mag_nf /= np.max(E_phi_mag_nf)

# Plot the near-field and far-field patterns
plt.figure(figsize=(12, 12))

#'''
# Near field plot
ax1 = plt.subplot(2, 2, 1, projection='3d')

# Create a surface plot for E_theta near field
E_theta_surface = E_theta_mag_nf  # Use the magnitude for plotting

# Convert to Cartesian coordinates for 3D plotting
x = radius_nf * np.outer(np.sin(theta), np.cos(phi))  # X-coordinates
y = radius_nf * np.outer(np.sin(theta), np.sin(phi))  # Y-coordinates
z = radius_nf * np.outer(np.cos(theta), np.ones(num_samples_phi))  # Z-coordinates

# Color mapping based on E_theta magnitude
E_theta_surface_flat = E_theta_surface.flatten()  # Flatten the surface for color mapping
norm = plt.Normalize(E_theta_surface_flat.min(), E_theta_surface_flat.max())
colors = plt.cm.viridis(norm(E_theta_surface_flat)).reshape(E_theta_surface.shape + (4,))  # Reshape colors

# Create surface plot
surf = ax1.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, alpha=0.7, linewidth=0)

ax1.set_title('E_theta (Near Field)')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Z-axis')
#'''

# Average over azimuthal angle to create a traditional radiation pattern
E_theta_avg = np.mean(E_theta_mag_nf, axis=1)  # Average over phi for E_theta
E_phi_avg = np.mean(E_phi_mag_nf, axis=1)  # Average over phi for E_phi

# Polar plot for averaged results
ax2 = plt.subplot(2, 2, 2, projection='polar')
ax2.plot(theta, E_theta_avg, label='E_theta (Near Field)', color='blue', alpha=0.7)
ax2.plot(theta, E_phi_avg, label='E_phi (Near Field)', color='red', alpha=0.7)
ax2.set_title('Normalized Near-field Pattern')
ax2.set_ylim(0, 1.1)  # Set y-limit for better visibility
ax2.legend()

# Set parameters for far-field computation
max_l = 10  # Maximum order of spherical harmonics
theta_f = np.linspace(0, np.pi, num_samples_theta)  # Far-field theta (0 to π)
phi_f = np.linspace(0, 2 * np.pi, num_samples_phi)  # Far-field phi (0 to 2π)

# Create nf_data array for far-field computation
nf_data = np.zeros((num_samples_theta * num_samples_phi, 4))
nf_data[:, 0] = theta_grid.flatten()  # theta
nf_data[:, 1] = phi_grid.flatten()  # phi
nf_data[:, 2] = E_theta_mag_nf.flatten()  # E_theta
nf_data[:, 3] = E_phi_mag_nf.flatten()  # E_phi

# Compute spherical harmonic coefficients from near-field data
def compute_far_field(nf_data, max_l):
    theta = nf_data[:, 0]
    phi = nf_data[:, 1]
    E_theta = nf_data[:, 2]
    E_phi = nf_data[:, 3]
    
    a_lm = np.zeros((max_l + 1, 2 * max_l + 1), dtype=complex)
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            a_lm[l, m + l] = np.sum(E_theta * Y_lm)  # Use E_theta only
            
    return a_lm

# Function to calculate far-field pattern from coefficients
def far_field_pattern(a_lm, theta_f, phi_f, max_l):
    E_far_theta = np.zeros((len(theta_f), len(phi_f)), dtype=complex)
    E_far_phi = np.zeros((len(theta_f), len(phi_f)), dtype=complex)
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            phi_f_grid, theta_f_grid = np.meshgrid(phi_f, theta_f, indexing='ij')
            Y_lm_f = sph_harm(m, l, phi_f_grid, theta_f_grid)
            E_far_theta += a_lm[l, m + l] * Y_lm_f
            E_far_phi += a_lm[l, m + l] * Y_lm_f
            
    return np.abs(E_far_theta), np.abs(E_far_phi)

# Compute far-field patterns
a_lm = compute_far_field(nf_data, max_l)
E_theta_far, E_phi_far = far_field_pattern(a_lm, theta_f, phi_f, max_l)

# Normalize the far-field electric field magnitudes
E_theta_far_mod = np.abs(E_theta_far)
E_phi_far_mod = np.abs(E_phi_far)

E_theta_far_mod /= np.max(E_theta_far_mod)  # Normalize E_theta
E_phi_far_mod /= np.max(E_phi_far_mod)  # Normalize E_phi

# Far field plot
ax3 = plt.subplot(2, 2, 3, projection='polar')

# Plot E_theta and E_phi for the far field
ax3.plot(phi_f, E_theta_far_mod[0, :], label='E_theta (Far Field, θ=0)', alpha=0.7)
ax3.plot(phi_f, E_phi_far_mod[0, :], label='E_phi (Far Field, θ=0)', alpha=0.7)
ax3.plot(phi_f, E_theta_far_mod[-1, :], label='E_theta (Far Field, θ=π)', alpha=0.7)
ax3.plot(phi_f, E_phi_far_mod[-1, :], label='E_phi (Far Field, θ=π)', alpha=0.7)

ax3.set_title('Normalized Far-field Pattern')
ax3.legend()

plt.tight_layout()
plt.show()