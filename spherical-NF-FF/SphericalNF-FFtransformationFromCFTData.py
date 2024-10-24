import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import random
import pandas as pd

#Newtons method for square root, used to get the square root of integers.
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

# Load the data, skipping the row with dashes (assumed to be the second row)
file_path = 'C:/Users/Valdemar/Desktop/datasetNF.txt'
nfData = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, header = None)

# Display the loaded dataframe
#print(nfData)

thetaSize = int(isqrt(nfData.shape[0]/2))
phiSize = thetaSize * 2

# Define theta and phi ranges for spherical coordinates
theta = np.linspace(0, np.pi, int(thetaSize))  # Polar angle
phi = np.linspace(0, 2 * np.pi, int(phiSize))  # Azimuthal angle

# Create meshgrid for spherical coordinates
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

# Initialize the total electric fields in the near field
E_theta_total_nf = np.zeros_like(theta_grid, dtype=np.complex_)
E_phi_total_nf = np.zeros_like(theta_grid, dtype=np.complex_)

#Gets the data in the two grids so that we can do the transform.
k = 0
for i in range(thetaSize):
    for j in range(phiSize):
            E_theta_total_nf[i, j] += nfData.iloc[k, 3]*np.cos(nfData.iloc[k, 4]*2*np.pi/360)+1j*nfData.iloc[k, 3]*np.sin(nfData.iloc[k, 4]*2*np.pi/360)
            E_phi_total_nf[i, j] += nfData.iloc[k, 5]*np.cos(nfData.iloc[k, 6]*2*np.pi/360)+1j*nfData.iloc[k, 5]*np.sin(nfData.iloc[k, 6]*2*np.pi/360)
            k += 1

# Convert complex fields to magnitude (near field)
E_theta_mag_nf = np.abs(E_theta_total_nf)
E_phi_mag_nf = np.abs(E_phi_total_nf)

# Normalize the near-field electric field magnitudes
E_theta_mag_nf /= np.max(E_theta_mag_nf)
E_phi_mag_nf /= np.max(E_phi_mag_nf)

# Set parameters for far-field computation
max_l = 10  # Maximum order of spherical harmonics
theta_f = np.linspace(0, np.pi, thetaSize)  # Far-field theta (0 to π)
phi_f = np.linspace(0, 2*np.pi, phiSize)  # Far-field phi (0 to 2π) azimuth

# Create nf_data array for far-field computation
nf_data = np.zeros((thetaSize * phiSize, 4))
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
    E_far = np.zeros((len(theta_f), len(phi_f)), dtype=complex)
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            phi_f_grid, theta_f_grid = np.meshgrid(theta_f ,phi_f, indexing='ij')
            Y_lm_f = sph_harm(m, l, phi_f_grid, theta_f_grid)
            E_far += a_lm[l, m + l] * Y_lm_f
            
    return np.real(E_far)

# Compute far-field patterns
a_lm = compute_far_field(nf_data, max_l)
E_far = far_field_pattern(a_lm, theta_f, phi_f, max_l)

# Normalize the far-field electric field magnitudes
E_far_mod = np.abs(E_far) #Reactive power bounces back, hence we are only interested in real power.

E_far_mod /= np.max(E_far_mod)  # Normalize E_theta

# Plot the far-field patterns using the averaged data
ax1 = plt.subplot(1, 1, 1, projection='polar')
ax1.plot(phi_f, E_far_mod[0, :], label='E_phi (Far Field)', alpha=0.7)
ax1.plot(theta_f, E_far_mod[:, 0], label='E_theta (Far Field)', alpha=0.7)
ax1.set_title('Normalized Far-field Pattern')
ax1.legend()
plt.tight_layout()
plt.show()