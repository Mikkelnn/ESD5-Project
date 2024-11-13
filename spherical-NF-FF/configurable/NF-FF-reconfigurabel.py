from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import spherical_far_field_transform
from modules.output import *
from scipy.signal import savgol_filter  # Savitzky-Golay filter
from collections import namedtuple


def select_data_at_angle(ffData, theta_f_deg, phi_f_deg, theta_select_angle=0, phi_select_angle=0):
  # variabels used for smoothing
  window_size = 13 # Choose an odd window size
  poly_order = 2    # Polynomial order for smoothing

  # deterimne sample number corresponding to plot angles
  theta_index = np.absolute(theta_f_deg - theta_select_angle).argmin()
  theta_plot_angle = theta_f_deg[theta_index]

  phi_index = np.absolute(phi_f_deg - phi_select_angle).argmin()
  phi_plot_angle = phi_f_deg[phi_index]

  # Select data at theta degrees and smooth it with Savitzky-Golay filter
  theta_angle_data_original = ffData[theta_index, :]

  # Select data at phi degrees and smooth it with Savitzky-Golay filter
  phi_angle_data_original = ffData[:, phi_index]

  # Apply Savitzky-Golay filter for smoothing
  theta_angle_data_smooth = savgol_filter(theta_angle_data_original, window_size, poly_order)

  # Apply Savitzky-Golay filter for smoothing    
  phi_angle_data_smooth = savgol_filter(phi_angle_data_original, window_size, poly_order)

  Desc = namedtuple("Desc", ["theta_plot_angle", "theta_angle_data_original", "theta_angle_data_smooth", "phi_plot_angle", "phi_angle_data_original", "phi_angle_data_smooth"])
  return Desc(
      theta_plot_angle,
      theta_angle_data_original,
      theta_angle_data_smooth,

      phi_plot_angle,
      phi_angle_data_original,
      phi_angle_data_smooth
  )

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
nfData, theta_deg, phi_deg = load_data_lab_measurements(file_path)

file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
ffData_loaded, theta_deg_loaded, phi_deg_loaded = load_data_lab_measurements(file_path2)

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
phi_f_deg_center = np.floor(phi_f_deg - (np.max(phi_f_deg) / 2))
theta_f_deg_center = np.floor(theta_f_deg - (np.max(theta_f_deg) / 2))

# 3. Transform data - most likely static...
# This function should ensure data is normalized before transforming!
max_l = 25  # Maximum order of spherical harmonics
ffData = spherical_far_field_transform(nfData, theta_f, phi_f, max_l)

# 4. Select far field at angle and smooth data
data = select_data_at_angle(ffData, theta_f_deg_center, phi_f_deg_center, theta_select_angle=0, phi_select_angle=0)

# 5. Output FF - plot or write to file
#plot_heatmap(ffData, theta_f_deg_center, phi_f_deg_center)

#plot_copolar(data, theta_f_deg_center, phi_f_deg_center)

#plot_polar(data, theta_f, phi_f)

#phi_hpbw = calculate_hpbw(data.theta_angle_data_original, phi_f_deg)
#print(f"Phi HPBW: {phi_hpbw} deg")

#theta_hpbw = calculate_hpbw(data.phi_angle_data_smooth, theta_f_deg)
#print(f"Theta HPBW: {theta_hpbw} deg")


# abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
# calculate the length between the two polarities
ffData_loaded_abs = ((abs(ffData_loaded[:, :, 0])**2 + abs(ffData_loaded[:, :, 1])**2)**0.5)

#ffData_loaded_abs = np.roll(ffData_loaded_abs, int(ffData_loaded_abs.shape[1] // 2), axis=1)
#ffData_loaded_abs = np.roll(ffData_loaded_abs, int(ffData_loaded_abs.shape[0] // 2), axis=0)

data_loaded = select_data_at_angle(ffData_loaded_abs, theta_f_deg_center, phi_f_deg_center, theta_select_angle=0, phi_select_angle=0)
plot_copolar(data_loaded, theta_f_deg_center, phi_f_deg_center)
plot_heatmap(ffData_loaded_abs, theta_f_deg_center, phi_f_deg_center)

# show all figures
show_figures()



