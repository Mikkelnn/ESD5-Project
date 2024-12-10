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

# file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
ffData_loaded, theta_deg_loaded, phi_deg_loaded, _, _ = load_data_lab_measurements(file_path2)

# simulate data
#nfData = simulate_NF_dipole_array()

# zero-pad before converting theta, phi values
nfData, theta_deg2, num_zero_nfData = pad_theta(nfData, theta_step_deg)
ffData_loaded, theta_deg2, num_zero_ffData = pad_theta(ffData_loaded, theta_step_deg)


# Define theta and phi ranges for far-field plotting
# get zero in center
phi_deg_center = np.floor(phi_deg - (np.max(phi_deg) / 2))
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)

##############################################################################################################
# 2. Introduction of errors in the NF, comment out if no errors should be present
##############################################################################################################
TEST_NAME = 'interference_both_pol_same_error' # used to determine folder to output files
PATH_PREFIX = f'./spherical-NF-FF/testResults/{TEST_NAME}/'
# ensure folder exist
from pathlib import Path
Path(PATH_PREFIX).mkdir(parents=False, exist_ok=True)

#phaseError = 0.4
ampError = 0.8
deviationFactor = 0.8
nfDataError = np.copy(nfData)
#appliedError = amplitude_same_errors_uniform(nfDataError,0.8)
appliedError = amplitude_errors_correlated_rev(nfDataError, deviationFactor, ampError)
#appliedError = phase_errors_correlated_rev(nfDataError, deviationFactor, phaseError)
#appliedError = fixed_phase_error(nfDataError, 0.4)

appliedError = removeXFromEnd(appliedError, int(num_zero_nfData))

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
ffData_error = spherical_far_field_transform_SNIFT(nfData_sum_error, frequency_Hz, transform_from_dist_meters, transform_to_dist_meters)

# post-process FF
ffData_no_error_2D = sum_NF_poles_sqrt(ffData_no_error)
ffData_error_2D = sum_NF_poles_sqrt(ffData_error)
# ffData_loaded = sum_NF_poles_sqrt(ffData_loaded)

# Normalize plots
# ffData_loaded = ffData_loaded / np.max(np.abs(ffData_loaded))
ffData_no_error_2D = ffData_no_error_2D / np.max(np.abs(ffData_no_error_2D))
ffData_error_2D = ffData_error_2D / np.max(np.abs(ffData_error_2D))

# Remove original zero padding
# ffData_loaded2 = removeXFromEnd(ffData_loaded, int(num_zero_nfData))
ffData_no_error_2D = removeXFromEnd(ffData_no_error_2D, int(num_zero_nfData))
ffData_error_2D = removeXFromEnd(ffData_error_2D, int(num_zero_nfData))

##############################################################################################################
# 4. Select far field at angle and smooth data
##############################################################################################################
phi_select_angle = 0 # the angle of witch to represent h-plane plot in degrees

ffData_no_error_2D = 20 * np.log10(ffData_no_error_2D)
ffData_error_2D = 20 * np.log10(ffData_error_2D)

# dataLoaded = select_data_at_angle(ffData_loaded2, phi_deg_loaded, phi_select_angle)

selected_ffData_no_error = select_data_at_angle(ffData_no_error_2D, phi_deg, phi_select_angle)
selected_ffData_error = select_data_at_angle(ffData_error_2D, phi_deg, phi_select_angle)
#dataDif = select_data_at_angle(farfieldDataDiff, phi_deg, phi_select_angle)

##############################################################################################################
# 5. Output FF - plot or write to file
##############################################################################################################
plot_copolar(selected_ffData_error, theta_deg_center, 'Transformed NF-FF copolar (with errors)')
plt.savefig(PATH_PREFIX + 'error_transformed_NF_(FF)_copolar.svg', bbox_inches='tight')

plot_polar(selected_ffData_error, theta_deg_center, 'Transformed NF-FF polar (with errors)')
plt.savefig(PATH_PREFIX + 'error_transformed_NF_(FF)_polar.svg', bbox_inches='tight')

plot_heatmap(ffData_error_2D, theta_deg, phi_deg, 'Transformed NF-FF heatmap (with errors)')
plt.savefig(PATH_PREFIX + 'error_transformed_NF_FF_heatmap.svg', bbox_inches='tight')
save_data_txt(ffData_error_2D, theta_deg, phi_deg, PATH_PREFIX + 'error_transformed_NF_FF_heatmap.txt', 'Theta Phi E_field')

# applied errors
plot_heatmap(appliedError[:,:,0], theta_deg, phi_deg, 'Applied NF error heatmap of polarity 0')
plt.savefig(PATH_PREFIX + 'applied_NF_error_heatmap_pol_0.svg', bbox_inches='tight')
plot_heatmap(appliedError[:,:,1], theta_deg, phi_deg, 'Applied NF error heatmap of polarity 1')
plt.savefig(PATH_PREFIX + 'applied_NF_error_heatmap_pol_1.svg', bbox_inches='tight')
save_data_txt(appliedError, theta_deg, phi_deg, PATH_PREFIX + 'aplied_error.txt', 'Theta Phi E_Pol_0 E_Pol_1')

# compare/dif plots
plot_error_compare(selected_ffData_no_error, selected_ffData_error, theta_deg_center, f'Error compare amplitude correlation reverse dev({deviationFactor}) amp({ampError})')
plt.savefig(PATH_PREFIX + 'compare_amplitude_correlation_reverse.svg', bbox_inches='tight')

plot_dif(selected_ffData_no_error, selected_ffData_error, theta_deg_center, f'Dif Radiation amplitude correlation reverse dev({deviationFactor}) amp({ampError})')
plt.savefig(PATH_PREFIX + 'dif_amplitude_correlation_reverse.svg', bbox_inches='tight')

diffData = ffData_no_error_2D - ffData_error_2D
plot_heatmap(diffData, theta_deg, phi_deg, 'Diff no error and error heatmap')
plt.savefig(PATH_PREFIX + 'diff_no_error_and_error_heatmap.svg', bbox_inches='tight')
save_data_txt(diffData, theta_deg, phi_deg, PATH_PREFIX + 'diff_no_error_and_error.txt', 'Theta Phi E_Pol_0 E_Pol_1')

plot_heatmap(ffData_error_2D, theta_deg, phi_deg, 'Transformed NF-FF heatmap (no errors)')
plt.savefig(PATH_PREFIX + 'no_error_transformed_NF_FF_heatmap.svg', bbox_inches='tight')

### save metrics data in txt (HPBW, mean, max)
metricsTxt = ''

#HPBW
metricsTxt += f'NF transformed data (FF) no errors:\n{calculate_print_hpbw(selected_ffData_no_error, theta_deg_center)}\n'
metricsTxt += f'\nNF transformed data (FF) with errors:\n{calculate_print_hpbw(selected_ffData_error, theta_deg_center)}\n'

# mean, max (selected data)
metricsTxt += '\nMean and Max errors between selected data with errors and selected data no errors\n'
metricsTxt += f"Max error e-plane: {calculate_max_indexed_error(selected_ffData_no_error.e_plane_data_original, selected_ffData_error.e_plane_data_original)}\n" 
metricsTxt += f"Max error h-plane: {calculate_max_indexed_error(selected_ffData_no_error.h_plane_data_original, selected_ffData_error.h_plane_data_original)}\n"
metricsTxt += f"Mean error e-plane: {calculate_mean_indexed_error(selected_ffData_no_error.e_plane_data_original, selected_ffData_error.e_plane_data_original)}\n"
metricsTxt += f"Mean error h-plane: {calculate_mean_indexed_error(selected_ffData_no_error.h_plane_data_original, selected_ffData_error.h_plane_data_original)}\n"

# mean, max (all data)
metricsTxt += '\nMean and Max errors between data with errors and data no errors (2D array)\n'
metricsTxt += f"Max absolute error (all data): {calculate_max_indexed_error(ffData_no_error_2D, ffData_error_2D)}\n" 
metricsTxt += f"Mean error (all data): {calculate_mean_indexed_error(ffData_no_error_2D, ffData_error_2D)}\n"

write_file(metricsTxt, PATH_PREFIX + 'metrics.txt')

# show all figures
show_figures()

