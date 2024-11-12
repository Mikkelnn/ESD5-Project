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
#file_path = './NF-FF-data/SH800_CBC_006000.CSV' # use relative path! i.e. universal :)
file_path = './NF-FF-Data-2/Flann16240-20_CBC_010000.CSV'
nfData = load_data_lab_measurements(file_path)


# simulate data
#nfData = simulate_NF_dipole_array()


# 2. Introduction of errors in the NF
# Comment out if no errors should be present

#amplitude_errors(nfData, standard_deviation=0.1)
#phase_errors(nfData, standard_deviation=0.1)


# Determine theta and phi sizes from the nf_data shape
# Define theta and phi ranges for far-field computation
theta_f = np.linspace(0, (5/6)*np.pi, nfData.shape[0])  # Far-field theta range
phi_f = np.linspace(0, 2 * np.pi, nfData.shape[1])  # Far-field phi range

# convert to degrees
phi_f_deg = (phi_f * 180 / np.pi)
theta_f_deg = (theta_f * 180 / np.pi)

# get zero in center
phi_f_deg -= (np.max(phi_f_deg) / 2)
theta_f_deg -= (np.max(theta_f_deg) / 2)

phi_f_deg = np.floor(phi_f_deg)
theta_f_deg = np.floor(theta_f_deg)

# 3. Transform data - most likely static...
# This function should ensure data is normalized before transforming!
max_l = 25  # Maximum order of spherical harmonics
ffData = spherical_far_field_transform(nfData, theta_f, phi_f, max_l)

plot_heatmap(ffData, theta_f_deg, phi_f_deg)
#exit()

# 4. Smooth transformed data?
# Comment out if no smoothing should be done

# Select data at 0 degrees and smooth it with Savitzky-Golay filter

n1 = ffData[int(ffData.shape[0] // 2), :]

# Apply Savitzky-Golay filter for smoothing
window_size = 13 # Choose an odd window size
poly_order = 2    # Polynomial order for smoothing
ffData_smooth = savgol_filter(n1, window_size, poly_order)

# 5. Output FF - plot or write to file
hpbw = calculate_hpbw(ffData_smooth, phi_f_deg)
print(f"HPBW: {hpbw} deg")

plot_ff_at(ffData_smooth, n1, phi_f_deg, axisName='phi', title='0 degree theta')


n2 = ffData[:, int(ffData.shape[1] // 2)]

# Apply Savitzky-Golay filter for smoothing
window_size = 13 # Choose an odd window size
poly_order = 2    # Polynomial order for smoothing
ffData_smooth2 = savgol_filter(n2, window_size, poly_order)


# 5. Output FF - plot or write to file
# theta_f_roll = np.roll(theta_f, int(len(theta_f) / 2))
#hpbw2 = calculate_hpbw(ffData_smooth2, theta_f_deg)
#print(f"HPBW: {hpbw2} deg")

plot_ff_at(ffData_smooth2, n2, theta_f_deg, axisName='theta', title="0 degrees phi")

#plot_polar(ffData, theta_f, phi_f)
#plot_heatmap(ffData)

#exit()