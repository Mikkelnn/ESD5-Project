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
# file_path = './simoulation CST/Parabolic/0.5sampleNearfield_2.98mParabolic.txt'
# nfData_1, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_cst(file_path)

#file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3mParabolic.txt'
#nfData_2, _, _, _, _ = load_data_cst(file_path)

#file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3.02mParabolic.txt'
#nfData_3, _, _, _, _ = load_data_cst(file_path)

#nfData = combine_data_for_position_error([nfData_1, nfData_2, nfData_3])

# rezise theta and phi axis
# new_shape = (int(nfData_1.shape[0] / 2), int(nfData_1.shape[1] / 2))
# nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = get_theta_phi_error_from_fine_set(nfData_1, new_shape, sample_theta=True, sample_phi=True)

# data from lab-measurements:
#file_path = './NF-FF-data/SH800_CBC_006000.CSV' # use relative path! i.e. universal :)
file_path = './NF-FF-Data-2/16240-20CBCFF_dir_30_010000.CSV'
nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_lab_measurements(file_path)


# load no error
ffData_no_error_loaded, _, _, _, _ = load_FF_data_own_output(f'./spherical-NF-FF/testResults/FF_data_no_error.txt')

# file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
# file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
# ffData_loaded, theta_deg_loaded, phi_deg_loaded, _, _ = load_data_lab_measurements(file_path2)

# simulate data
#nfData = simulate_NF_dipole_array()

# zero-pad before converting theta, phi values
nfData, theta_deg2, num_zero_nfData = pad_theta(nfData, theta_step_deg)
# ffData_loaded, theta_deg2, num_zero_ffData = pad_theta(ffData_loaded, theta_step_deg)


# Define theta and phi ranges for far-field plotting
# get zero in center
phi_deg_center = np.floor(phi_deg - (np.max(phi_deg) / 2))
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)

##############################################################################################################
# 2. Introduction of errors in the NF, comment out if no errors should be present
##############################################################################################################



#phaseError = 0.4
# ampError = 0.8
deviationFactor = 0.5
nfDataError = np.copy(nfData)
# appliedError = phase_same_errors_normal(nfDataError, TEST[0])
#appliedError = phase_errors_correlated_theta_same(nfDataError, deviationFactor, TEST[0])
#appliedError = fixed_phase_error(nfDataError, 0.4)

#appliedError = removeXFromEnd(appliedError, int(num_zero_nfData))

##############################################################################################################
# 3. Transform data - most likely static...
##############################################################################################################
frequency_Hz = 10e9 # 10GHz
transform_from_dist_meters = 0.3 # The distance you want the transform from.
transform_to_dist_meters = 10e6 # the distance you want the transform to!

# pre-process nfData
nfData_sum = HansenPreProcessing(nfData)
nfData_sum_error = HansenPreProcessing(nfDataError)

# transform NF to FF
ffData_no_error = spherical_far_field_transform_SNIFT(nfData_sum, frequency_Hz, transform_from_dist_meters, transform_to_dist_meters)
#ffData_error = spherical_far_field_transform_SNIFT(nfData_sum_error, frequency_Hz, transform_from_dist_meters, transform_to_dist_meters)

# post-process FF
ffData_no_error_2D = sum_NF_poles_sqrt(ffData_no_error)
#ffData_error_2D = sum_NF_poles_sqrt(ffData_error)
# ffData_loaded = sum_NF_poles_sqrt(ffData_loaded)

# Normalize plots
# ffData_loaded = ffData_loaded / np.max(np.abs(ffData_loaded))
ffData_no_error_2D = ffData_no_error_2D / np.max(np.abs(ffData_no_error_2D))
#ffData_error_2D = ffData_error_2D / np.max(np.abs(ffData_error_2D))

# Remove original zero padding
# ffData_loaded2 = removeXFromEnd(ffData_loaded, int(num_zero_nfData))
ffData_no_error_2D = removeXFromEnd(ffData_no_error_2D, int(num_zero_nfData))
#ffData_error_2D = removeXFromEnd(ffData_error_2D, int(num_zero_nfData))

##############################################################################################################
# 4. Select far field at angle and smooth data
##############################################################################################################
phi_select_angle = 0 # the angle of witch to represent h-plane plot in degrees

ffData_no_error_2D = 20 * np.log10(ffData_no_error_2D)
#ffData_error_2D = 20 * np.log10(ffData_error_2D)

# dataLoaded = select_data_at_angle(ffData_loaded2, phi_deg_loaded, phi_select_angle)

selected_ffData_no_error = select_data_at_angle(ffData_no_error_2D, phi_deg, phi_select_angle)
#selected_ffData_error = select_data_at_angle(ffData_error_2D, phi_deg, phi_select_angle)
#dataDif = select_data_at_angle(farfieldDataDiff, phi_deg, phi_select_angle)

plot_ffData_no_error = select_data_at_angle(ffData_no_error_loaded, phi_deg, phi_select_angle)

##############################################################################################################
# 5. Output FF - plot or write to file
##############################################################################################################
plot_copolar(selected_ffData_no_error, theta_deg_center, '(new) Transformed NF-FF copolar')
plot_copolar(plot_ffData_no_error, theta_deg_center, '(old) Transformed NF-FF copolar')

#plt.savefig(PATH_PREFIX + 'error_transformed_NF_(FF)_copolar.svg', bbox_inches='tight')

# show all figures
show_figures()

