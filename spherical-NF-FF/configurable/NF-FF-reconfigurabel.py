from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import *
from modules.pre_process import *
from modules.post_process import *
from modules.output import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# The idea of this script is to make it easy to configure different scenarios
# The heavy lifting will be done in multiple sub-files

# Examlpe configurations:
# read_NF_lab_data -> transform -> plot
# read_NF_lab_data -> introduce_error -> transform -> plot
# read_NF_lab_data -> transform -> smooth -> plot
# read_NF_lab_data -> introduce_error -> transform -> smooth -> plot

##############################################################################################################
# 1. Where is the NF data from? (simulated or from a file)
##############################################################################################################

# data from CST simulation:
#file_path = './simoulation CST/Parabolic/0.5sampleNearfield_2.98mParabolic.txt'
#nfData_1, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_cst(file_path)

#file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3mParabolic.txt'
#nfData_2, _, _, _, _ = load_data_cst(file_path)

#file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3.02mParabolic.txt'
#nfData_3, _, _, _, _ = load_data_cst(file_path)

#nfData = combine_data_for_position_error([nfData_1, nfData_2, nfData_3])

# rezise theta and phi axis
#new_shape = (int(nfData_1.shape[0] / 2), int(nfData_1.shape[1] / 2))
#nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = get_theta_phi_error_from_fine_set(nfData_1, new_shape, sample_theta=True, sample_phi=True)

# data from lab-measurements:
#file_path = './NF-FF-data/SH800_CBC_006000.CSV' # use relative path! i.e. universal :)
frequency_Hz = 10e9 # 10GHz
file_path = './NF-FF-Data-2/16240-20CBCFF_dir_30_010000.CSV'
nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_lab_measurements(file_path)

# file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
ffData_loaded, theta_deg_loaded, phi_deg_loaded, _, _ = load_data_lab_measurements(file_path2)

# simulate data
#nfData = simulate_NF_dipole_array()

# Determine theta and phi sizes from the nf_data shape
# Define theta and phi ranges for far-field computation
# convert to radians
phi_rad =  np.deg2rad(phi_deg)
theta_rad = np.deg2rad(theta_deg)

theta_step_rad = np.deg2rad(theta_step_deg)
phi_step_rad = np.deg2rad(phi_step_deg)

# get zero in center
phi_deg_center = np.floor(phi_deg - (np.max(phi_deg) / 2))
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)


##############################################################################################################
# 2. Introduction of errors in the NF
# Comment out if no errors should be present
##############################################################################################################

nfDataError = np.copy(nfData)
amplitude_errors(nfDataError, 0.0)
phase_errors(nfDataError, 0.0)
#fixed_phase_error(nfDataError, 0.4)


##############################################################################################################
# 3. Transform data - most likely static...
##############################################################################################################
max_l = 20  # Maximum order of spherical harmonics
M = 10
transform_to_dist_meters = 10e6 # the distance you want the transform to!

# pre-process nfData
nfData = zero_pad_theta(nfData, theta_deg, theta_step_deg)
nfData_sum = HansenPreProcessing(nfData)
nfData_sum_error = HansenPreProcessing(nfDataError)

# transform NF to FF
ffData = spherical_far_field_transform_gigacook(nfData_sum, theta_rad, phi_rad, theta_step_rad, phi_step_rad, frequency_Hz, transform_to_dist_meters, N=max_l, M=M)
ffDataError = spherical_far_field_transform_gigacook(nfData_sum_error, theta_rad, phi_rad, theta_step_rad, phi_step_rad, frequency_Hz, transform_to_dist_meters, N=max_l, M=M)

# post-process FF
farfieldData = sum_NF_poles(ffData)
farfieldDataError = sum_NF_poles(ffDataError)
ffData_loaded = sum_NF_poles_sqrt(ffData_loaded)

# Normalize plots
ffData_loaded = ffData_loaded / np.max(np.abs(ffData_loaded))
farfieldData = farfieldData / np.max(np.abs(farfieldData))
farfieldDataError = farfieldDataError / np.max(np.abs(farfieldData))


##############################################################################################################
# 4. Select far field at angle and smooth data
##############################################################################################################
phi_select_angle = 0 # the angle of witch to represent h-plane plot in degrees

#farfieldData_20log10 = 20 * np.log10(farfieldData)
#farfieldDataError_20log10 = 20 * np.log10(farfieldDataError)
#farfieldDataDiff = 20 * np.log10(abs(farfieldDataError - farfieldData) / farfieldData)

dataLoaded = select_data_at_angle(ffData_loaded, phi_deg_loaded, phi_select_angle)

data1 = select_data_at_angle(farfieldData, phi_deg, phi_select_angle)
dataError = select_data_at_angle(farfieldDataError, phi_deg, phi_select_angle)
#dataDif = select_data_at_angle(farfieldDataDiff, phi_deg, phi_select_angle)


##############################################################################################################
# 5. Output FF - plot or write to file
##############################################################################################################
plot_error_compare(data1, dataLoaded, theta_deg_center, 'Error compare')
plot_dif(data1, dataLoaded, theta_deg_center, 'Dif Radiation plot')
calculate_print_hpbw(data1, theta_deg_center)


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

#plot_polar2(data_loaded, theta_rad2, 'Loaded FF polar')

#plot_heatmap(ffData_loaded_abs, theta_deg_loaded, phi_deg_loaded, 'Loaded FF heatmap')
#plot_heatmap(farfieldData, theta_deg_loaded, phi_deg_loaded, 'Transformed FF heatmap')
#plot_heatmap(farfieldDataError, theta_deg_loaded, phi_deg_loaded, 'Transformed FF heatmap with Error')
#plot_heatmap(abs(farfieldDataError - farfieldData) / farfieldData, theta_deg_loaded, phi_deg_loaded, 'Dif error heatmap')

#plot_copolar2(data_loaded, theta_deg_center2, 'Loaded FF copolar')
#plot_copolar2(data1, theta_deg_center2, 'Transformed FF copolar')
#plot_copolar2(dataError, theta_deg_center2, 'Transformed FF copolar with Error')

# show all figures
show_figures()

