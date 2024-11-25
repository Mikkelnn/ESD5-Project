from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import *
from modules.pre_process import *
from modules.output import *
from scipy.signal import savgol_filter  # Savitzky-Golay filter
from collections import namedtuple
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def select_data_at_angle(ffData, theta_f_deg, phi_f_deg, theta_select_angle=0, phi_select_angle=0):
  # variabels used for smoothing
  window_size = 9 # Choose an odd window size
  poly_order = 2    # Polynomial order for smoothing

  # deterimne sample number corresponding to plot angles
  theta_index = np.absolute(theta_f_deg - theta_select_angle).argmin() - 1
  theta_plot_angle = theta_f_deg[theta_index]

  phi_index = np.absolute(phi_f_deg - phi_select_angle).argmin()
  phi_plot_angle = phi_f_deg[phi_index]

  # Select data at theta degrees and smooth it with Savitzky-Golay filter
  theta_angle_data_original = ffData[theta_index, :]

  # Select data at phi degrees and smooth it with Savitzky-Golay filter
  #get first half:
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

def select_data_at_angle2(ffData, theta_f_deg, phi_f_deg, phi_select_angle=0):
  # variabels used for smoothing
  window_size = 9 # Choose an odd window size
  poly_order = 2    # Polynomial order for smoothing

  # deterimne sample number corresponding to plot angles
  phi_index_0 = np.absolute(phi_f_deg - phi_select_angle).argmin()
  phi_index_180 = np.absolute(phi_f_deg - ((phi_select_angle + 180) % 360)).argmin()

  phi_index_90 = np.absolute(phi_f_deg - ((phi_select_angle + 90) % 360)).argmin()
  phi_index_270 = np.absolute(phi_f_deg - ((phi_select_angle + 270) % 360)).argmin()
  
  h_plane_plot_angle = phi_f_deg[phi_index_0]
  e_plane_plot_angle = phi_f_deg[phi_index_90]

  # Select data
  h_plane_data_original = np.concatenate((np.flip(ffData[:, phi_index_0]), ffData[1:, phi_index_180]))
  e_plane_data_original = np.concatenate((np.flip(ffData[:, phi_index_90]), ffData[1:, phi_index_270]))

  # Apply Savitzky-Golay filter for smoothing
  h_plane_data_smooth = savgol_filter(h_plane_data_original, window_size, poly_order) 
  e_plane_data_smooth = savgol_filter(e_plane_data_original, window_size, poly_order)

  Desc = namedtuple("Desc", ["h_plane_plot_angle", "h_plane_data_original", "h_plane_data_smooth", "e_plane_plot_angle", "e_plane_data_original", "e_plane_data_smooth"])
  return Desc(
      h_plane_plot_angle,
      h_plane_data_original,
      h_plane_data_smooth,

      e_plane_plot_angle,
      e_plane_data_original,
      e_plane_data_smooth
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
file_path = './simoulation CST/Parabolic/0.5sampleNearfield_2.98mParabolic.txt'
nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_cst(file_path)

# file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3mParabolic.txt'
# nfData_2, _, _, _, _ = load_data_cst(file_path)

# file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3.02mParabolic.txt'
# nfData_3, _, _, _, _ = load_data_cst(file_path)

# nfData = combine_data_for_position_error([nfData_1, nfData_2, nfData_3])

# rezise theta and phi axis
new_shape = (int(nfData.shape[0] / 2), int(nfData.shape[1] / 2))
nfData_reduced = get_theta_phi_error_from_fine_set(nfData, new_shape, sample_theta=True, sample_phi=True)

print(phi_deg)


# data from lab-measurements:
#file_path = './NF-FF-data/SH800_CBC_006000.CSV' # use relative path! i.e. universal :)
frequency_Hz = 10e9 # 10GHz
#file_path = './NF-FF-Data-2/16240-20CBCFF_dir_30_010000.CSV'
#nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_lab_measurements(file_path)

# file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
ffData_loaded, theta_deg_loaded, phi_deg_loaded, _, _ = load_data_lab_measurements(file_path2)

# simulate data
#nfData = simulate_NF_dipole_array()


# 2. Introduction of errors in the NF
# Comment out if no errors should be present

#amplitude_errors(nfData, standard_deviation=0.1)
#phase_errors(nfData, standard_deviation=0.1)


# Determine theta and phi sizes from the nf_data shape
# Define theta and phi ranges for far-field computation
#theta_f = np.linspace(0, (5/6)*np.pi, nfData.shape[0])  # Far-field theta range
#phi_f = np.linspace(0, 2 * np.pi, nfData.shape[1])  # Far-field phi range

# convert to degrees
#phi_f_deg = (phi_f * 180 / np.pi)
#theta_f_deg = (theta_f * 180 / np.pi)


theta_deg = np.linspace(0, 180, int(180 / theta_step_deg)+1)

phi_rad =  np.deg2rad(phi_deg)
theta_rad = np.deg2rad(theta_deg)

theta_step_rad = np.deg2rad(theta_step_deg)
phi_step_rad = np.deg2rad(phi_step_deg)

# get zero in center
phi_deg_center = np.floor(phi_deg - (np.max(phi_deg) / 2))
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1) #np.floor(theta_deg - (np.max(theta_deg) / 2))

# pre-process nfData
nfData_sum = HansenPreProcessing(nfData)
#nfData_sum = zero_pad_theta(nfData_sum, theta_deg, theta_step_deg) #Might need to implement this later.

# 3. Transform data - most likely static...
# This function should ensure data is normalized before transforming!
max_l = 10  # Maximum order of spherical harmonics
ffData = spherical_far_field_transform_gigacook(nfData_sum, theta_rad, phi_rad, theta_step_rad, phi_step_rad, frequency_Hz, nf_meas_dist=10e3, N=max_l, M=10) #nf_meas_dist is the distance you want the transform at!
# roll data if needed
#ffData = np.roll(ffData, int(ffData.shape[1] // 2), axis=1)
#ffData = np.roll(ffData, int(ffData.shape[0] // 2), axis=0)

#Flip the array:
ffData = np.flip(ffData, 0)
#Roll 2
ffData = np.roll(ffData, -1, axis=0)

# order by row sum DESC
#row_sums = np.sum(ffData, axis=1)
#sorted_indices_desc = np.argsort(row_sums)[::-1]
#ffData = ffData[sorted_indices_desc]

# 4. Select far field at angle and smooth data
#data = select_data_at_angle(np.abs(ffData[:,:,0]), theta_deg_center, phi_deg_center, theta_select_angle=0, phi_select_angle=0)
#data = select_data_at_angle2(np.abs(ffData[:,:,0]), theta_deg, phi_deg, phi_select_angle=0)

# 5. Output FF - plot or write to file
#plot_heatmap(np.abs(ffData[:,:,0]), theta_deg, phi_deg, 'Transformed NF (FF) heatmap')
#plot_copolar(data, theta_deg_center, phi_deg_center, 'Transformed NF (FF) copolar')
#plot_copolar2(data, theta_deg_center, 'Transformed NF (FF) copolar')
#plot_polar(data, theta_rad, phi_rad, 'Transformed NF (FF) polar')


#theta_deg_center2 = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)
#theta_rad2 = np.linspace(-(5/6)*np.pi, (5/6)*np.pi, len(theta_deg_center2))

#data_loaded = select_data_at_angle2(ffData, theta_deg, phi_deg, phi_select_angle=0)
#plot_heatmap(ffData, theta_deg, phi_deg, 'Transformed NF (FF) heatmap')
#plot_copolar2(data_loaded, theta_deg_center2, 'Transformed NF (FF) copolar')
#plot_polar2(data_loaded, theta_rad2, , 'Transformed NF (FF) polar')

#phi_hpbw = calculate_hpbw(data.theta_angle_data_original, phi_f_deg)
#print(f"Phi HPBW: {phi_hpbw} deg")

#theta_hpbw = calculate_hpbw(data.phi_angle_data_smooth, theta_f_deg)
#print(f"Theta HPBW: {theta_hpbw} deg")


# abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
# calculate the length between the two polarities
ffData_loaded_abs = ((abs(ffData_loaded[:, :, 0])**2 + abs(ffData_loaded[:, :, 1])**2)**0.5)

theta_deg_center2 = np.linspace(-np.max(theta_deg_loaded), np.max(theta_deg_loaded), (len(theta_deg_loaded)*2)-1)
theta_rad2 = np.linspace(-(5/6)*np.pi, (5/6)*np.pi, len(theta_deg_center2))

farfieldData = np.abs(ffData[:,:,0]**2 + ffData[:,:,1]**2)

data_loaded = select_data_at_angle2(ffData_loaded_abs, theta_deg_loaded, phi_deg_loaded, phi_select_angle=0)
data1 = select_data_at_angle2(farfieldData, theta_deg_loaded, phi_deg_loaded, phi_select_angle=0)

#plot_polar2(data_loaded, theta_rad2, 'Loaded FF polar')

#Introduce errors:
nfDataError = np.copy(nfData)
amplitude_errors(nfDataError, 0.05)
phase_errors(nfDataError, 0.05)
#fixed_phase_error(nfDataError, 0.4)

nfData_sum_error = HansenPreProcessing(nfDataError)

ffDataError = spherical_far_field_transform_gigacook(nfData_sum_error, theta_rad, phi_rad, theta_step_rad, phi_step_rad, frequency_Hz, nf_meas_dist=10e3, N=max_l, M=10) #nf_meas_dist is the distance you want the transform at!

ffDataError = np.flip(ffDataError, 0)
ffDataError = np.roll(ffDataError, -1, axis = 0)
farfieldDataError = np.abs(ffDataError[:,:,0]**2 + ffDataError[:,:,1]**2)

dataError = select_data_at_angle2(farfieldDataError, theta_deg_loaded, phi_deg_loaded, phi_select_angle=0)
dataDif = select_data_at_angle2(20*np.log10(abs(farfieldDataError - farfieldData) / farfieldData), theta_deg_loaded, phi_deg_loaded, phi_select_angle=0)

#plot_heatmap(ffData_loaded_abs, theta_deg_loaded, phi_deg_loaded, 'Loaded FF heatmap')
#plot_heatmap(farfieldData, theta_deg_loaded, phi_deg_loaded, 'Transformed FF heatmap')
#plot_heatmap(farfieldDataError, theta_deg_loaded, phi_deg_loaded, 'Transformed FF heatmap with Error')
#plot_heatmap(abs(farfieldDataError - farfieldData) / farfieldData, theta_deg_loaded, phi_deg_loaded, 'Dif error heatmap')

#plot_copolar2(data_loaded, theta_deg_center2, 'Loaded FF copolar')
#plot_copolar2(data1, theta_deg_center2, 'Transformed FF copolar')
#plot_copolar2(dataError, theta_deg_center2, 'Transformed FF copolar with Error')
plot_error_compare(data1, dataError, theta_deg_center2, 'Error compare')
plot_dif(data1, dataError, theta_deg_center2, 'Dif Radiation plot')



# show all figures
show_figures()

