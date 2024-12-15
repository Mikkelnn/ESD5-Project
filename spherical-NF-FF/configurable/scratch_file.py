from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import *
from modules.pre_process import *
from modules.post_process import *
from modules.output import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
max_error = 0.1          # Maximum error (e.g., ±10%)
deviation_factor = 0.05  # How much the error deviates in each step
initial_error = 1.0      # Starting value
num_points = 100         # Number of points to generate

# Generate the sequence of errors
errors = [initial_error]
for _ in range(1, num_points):
    errors.append(nextError(errors[-1], max_error, deviation_factor))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(errors, marker=None, linestyle='-', color='blue', label='Error values')
plt.title(f'Line Chart of Error Progression, max_error = {max_error}; deviation_factor = {deviation_factor}')
plt.xlabel('Step')
plt.ylabel('Error Value')
plt.grid(True)
plt.legend()
plt.show()


exit()

import numpy as np
import matplotlib.pyplot as plt

# Define the HPBW (Half-Power Beamwidth)
HPBW = 40

# Define angles (theta) in degrees
theta = np.linspace(-90, 90, 1000)  # From -90 to 90 degrees

# Gaussian Function
def gaussian(theta, HPBW):
    return np.exp(-np.log(2) * (2 * theta / HPBW)**2)


# Calculate the functions
gaussian_values = gaussian(theta, HPBW)

# Plot the results
plt.figure(figsize=(10, 6))

# Gaussian Plot
plt.plot(theta, gaussian_values, label='Radiation pattern', color='blue')

# Add labels, legend, and grid
plt.title(f'Approximated probe radiation pattern, HPBW = {HPBW} Degrees')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Amplitude')
plt.axhline(0.5, color='gray', linestyle=':', label='Half-Power (-3 dB)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


exit()

file_path = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV' # FF-data
# file_path = './NF-FF-Data-2/16240-20CBCFF_dir_30_010000.CSV' # NF-data
ffData_loaded, theta_deg, phi_deg, _, _ = load_data_lab_measurements(file_path)


# Define theta and phi ranges for far-field plotting
# get zero in center
phi_deg_center = np.floor(phi_deg - (np.max(phi_deg) / 2))
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)


ffData_error_2D = sum_NF_poles_sqrt(ffData_loaded)
ffData_loaded_2D = ffData_error_2D / np.max(np.abs(ffData_error_2D))

# ffData_loaded_2D = 20 * np.log10(ffData_loaded_2D)

selected_ffData_loaded = select_data_at_angle(ffData_error_2D, phi_deg, 0)

# print(f'First sidelobe val: {selected_ffData_loaded.e_plane_data_original[33]}')
# exit()

def find_first_sidelobe(data):
    # Identify the main lobe (global maximum)
    main_lobe_index = np.argmax(data)
    
    # Search for the first peak after the main lobe
    for i in range(main_lobe_index + 1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            return i, data[i]
    
    return None, None  # Return None if no sidelobe is found

index, value = find_first_sidelobe(selected_ffData_loaded.e_plane_data_original)

print(f'First sidelobe index: {index}; val: {value}')
plot_copolar(selected_ffData_loaded, theta_deg_center, 'Loaded FF copolar')
# plt.savefig(PATH_PREFIX + 'error_transformed_NF_(FF)_copolar.svg', bbox_inches='tight')

show_figures()