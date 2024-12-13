from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import *
from modules.pre_process import *
from modules.post_process import *
from modules.output import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# file_path = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
file_path = './NF-FF-Data-2/16240-20CBCFF_dir_30_010000.CSV'
ffData_loaded, theta_deg, phi_deg, _, _ = load_data_lab_measurements(file_path)


# Define theta and phi ranges for far-field plotting
# get zero in center
phi_deg_center = np.floor(phi_deg - (np.max(phi_deg) / 2))
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)


ffData_error_2D = sum_NF_poles_sqrt(ffData_loaded)
ffData_loaded_2D = ffData_error_2D / np.max(np.abs(ffData_error_2D))
selected_ffData_loaded = select_data_at_angle(ffData_error_2D, phi_deg, 0)

print(f'First sidelobe val: {selected_ffData_loaded.e_plane_data_original[33]}')

exit()

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