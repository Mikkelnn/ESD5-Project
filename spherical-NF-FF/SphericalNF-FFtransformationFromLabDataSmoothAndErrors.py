import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.signal import savgol_filter  # Savitzky-Golay filter
import pandas as pd

standardDeviation = 0.1 # multiplies each value 1 + a random value from normal distribution with center 0.
# Load the data, skipping the non data rows
# file_path = 'C:/Users/Valdemar/Desktop/datasetFF30.txt'
file_path = './NF-FF-data/SH800_CBC_008000.CSV' # Do this, now it is a relative path! i.e. universal :)
nfData = pd.read_csv(file_path, delim_whitespace=True, skiprows=13, header = None)

# Display the loaded dataframe
#print(nfData)

thetaSize = int(np.sqrt(nfData.shape[0]/(4 * 2.5)))
phiSize = int(thetaSize * 2 * 2.5)

print(thetaSize)
print(phiSize)

# Define theta and phi ranges for spherical coordinates
theta = np.linspace(0, np.pi, int(thetaSize))  # Polar angle
phi = np.linspace(0, 2 * np.pi, int(phiSize))  # Azimuthal angle

# Create meshgrid for spherical coordinates
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

# Initialize the total electric fields in the near field
E_theta_total_nf = np.zeros_like(theta_grid, dtype=np.complex128)
E_phi_total_nf = np.zeros_like(theta_grid, dtype=np.complex128)

# Gets the data in the two grids so that we can do the transform.
k = 0
for i in range(thetaSize):
    for j in range(phiSize):
        #print(e1, e2, e3, e4)
        E_theta_total_nf[i, j] += nfData.iloc[k, 3] + 1j * nfData.iloc[k, 4]
        E_phi_total_nf[i, j] += nfData.iloc[k + int(nfData.shape[0] / 2), 3] + 1j * nfData.iloc[k + int(nfData.shape[0] / 2), 4]
        k += 1

# Convert complex fields to magnitude real part (near field)
E_theta_mag_nf = np.abs(np.real(E_theta_total_nf))
E_phi_mag_nf = np.abs(np.real(E_phi_total_nf))

# Set parameters for far-field computation
max_l = 25  # Maximum order of spherical harmonics
theta_f = np.linspace(0, np.pi, thetaSize)  # Far-field theta (0 to 5/6*π)
phi_f = np.linspace(0, 2 * np.pi, phiSize)  # Far-field phi (0 to 2π) azimuth

# Create nf_data array for far-field computation
nf_data = np.zeros((thetaSize * phiSize, 4))
nf_data[:, 0] = theta_grid.flatten()  # theta
nf_data[:, 1] = phi_grid.flatten()  # phi
nf_data[:, 2] = E_theta_mag_nf.flatten()  # E_theta
nf_data[:, 3] = E_phi_mag_nf.flatten()  # E_phi

for i in range(len(nf_data)):
    e1 = (1+np.random.normal(0, standardDeviation))
    e2 = (1+np.random.normal(0, standardDeviation))
    nf_data[i,2] = abs(nf_data[i,2] * e1)
    nf_data[i,3] = abs(nf_data[i,2] * e2)

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
            phi_f_grid, theta_f_grid = np.meshgrid(theta_f, phi_f, indexing='ij')
            Y_lm_f = sph_harm(m, l, phi_f_grid, theta_f_grid)
            E_far += a_lm[l, m + l] * Y_lm_f
            
    return E_far # Reactive power bounces back, hence we are only interested in real power.

# Compute far-field patterns
a_lm = compute_far_field(nf_data, max_l)
E_far_real = far_field_pattern(a_lm, theta_f, phi_f, max_l)

# Normalize the far-field electric field magnitudes
E_far_mod = np.abs(E_far_real) 
E_far_mod /= np.max(E_far_mod)  # Normalize E_theta

# Select data at 0 degrees and smooth it with Savitzky-Golay filter
n1 = E_far_mod[0, :]
n1 = np.roll(n1, int(len(n1) / 2))

# Apply Savitzky-Golay filter for smoothing
window_size = 11 # Choose an odd window size
poly_order = 2    # Polynomial order for smoothing
n1_smooth = savgol_filter(n1, window_size, poly_order)

# Plot the far-field patterns using the smoothed data
ax1 = plt.subplot(1, 1, 1)
ax1.plot(phi_f, 20 * np.log10(n1_smooth), label='E_phi (Far Field) 0 degrees phi, copolar, with Savitzky-Golay filter', alpha=0.7)
ax1.plot(phi_f, 20 * np.log10(n1), label='E_phi (Far Field) 0 degrees phi, copolar', alpha=0.7)
# ax1.plot(phi_f, 20 * np.log10(E_far_mod[17, :]), label='E_phi (Far Field) 90 degrees phi', alpha=0.7)
# ax1.plot(theta_f, np.log10(E_far_mod[:, 0]), label='E_theta (Far Field)', alpha=0.7)
ax1.set_title('Normalized Far-field Pattern')
ax1.grid()
ax1.legend()
plt.tight_layout()
plt.show()