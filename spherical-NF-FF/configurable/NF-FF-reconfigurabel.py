from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import spherical_far_field_transform
from modules.output import *
from scipy.signal import savgol_filter  # Savitzky-Golay filter

# The idea of this script is to make it easy to configure different scenarios
# The heavy lifting will be done in multiple sub-files

# Examlpe configurations:
# read_NF_lab_data -> transform -> plot
# read_NF_lab_data -> introduce_error -> transform -> plot
# read_NF_lab_data -> transform -> smooth -> plot
# read_NF_lab_data -> introduce_error -> transform -> smooth -> plot


# 1. Where is the NF data from? (simulated or from a file)

# data from CST simulation:
#file_path = './simoulation CST/Parabolic/ReflectorParabolic5meterEfield_virker3D.txt'
#nfData = load_data_cst(file_path)

# data from lab-measurements:
file_path = './NF-FF-data/SH800_CBC_008000.CSV' # use relative path! i.e. universal :)
nfData = load_data_lab_measurements(file_path)


# simulate data
#nfData = simulate_NF_dipole_array()


# 2. Introduction of errors in the NF
# Comment out if no errors should be present

#amplitude_errors(nfData, standard_deviation=0.1)
#phase_errors(nfData, standard_deviation=0.1)


# Determine theta and phi sizes from the nf_data shape
# Define theta and phi ranges for far-field computation
theta_f = np.linspace(0, np.pi, nfData.shape[0])  # Far-field theta range
phi_f = np.linspace(0, 2 * np.pi, nfData.shape[1])  # Far-field phi range

# 3. Transform data - most likely static...
# This function should ensure data is normalized before transforming!
max_l = 25  # Maximum order of spherical harmonics
ffData = spherical_far_field_transform(nfData, theta_f, phi_f, max_l)


# 4. Smooth transformed data?
# Comment out if no smoothing should be done

# Select data at 0 degrees and smooth it with Savitzky-Golay filter

# plot_heatmap(ffData)
# exit()

#n1 = ffData[ffData.shape[0] // 2, :]
n1 = ffData[0, :]
n1 = np.roll(n1, int(len(n1) / 2)) # roll data to center when plotting

# Apply Savitzky-Golay filter for smoothing
window_size = 11 # Choose an odd window size
poly_order = 2    # Polynomial order for smoothing
ffData_smooth = savgol_filter(n1, window_size, poly_order)


# 5. Output FF - plot or write to file
plot_ff_at(ffData_smooth, n1, theta_f, phi_f)