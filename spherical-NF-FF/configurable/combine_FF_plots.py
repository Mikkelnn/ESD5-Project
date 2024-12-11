import glob
import numpy as np
import matplotlib.pyplot as plt
from modules.loadData import load_FF_data_own_output
from modules.post_process import select_data_at_angle

def plot_error_compare(noError, dataErrorArr, errorTxt, theta_f_deg, figure_title):
    h_plane_plot_angle = noError.h_plane_plot_angle
    e_plane_plot_angle = noError.e_plane_plot_angle

    # Plot the far-field patterns
    fig = plt.figure(figure_title, figsize=(8, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], width_ratios=[1])
    ###
    # H-plane
    ###
    ax1 = fig.add_subplot(grid[0, 0])    
    ax1.set_title(f'{figure_title} H-plane Phi = {h_plane_plot_angle}')
    
    #ax1.plot(theta_f_deg, noError.h_plane_data_smooth , label=f'Radiation without errors', alpha=0.7)    
    ax1.plot(theta_f_deg, noError.h_plane_data_original, label=f'Radiation without errors', alpha=0.7)
    for idx, data in enumerate(dataErrorArr):
        # ax1.plot(theta_f_deg, data.h_plane_data_smooth, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)    
        ax1.plot(theta_f_deg, data.h_plane_data_original, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)

    ax1.set_xlabel('Theta')
    ax1.grid()
    ax1.legend()
    ###
    # E-plane
    ###
    ax2 = fig.add_subplot(grid[1, 0])    
    ax2.set_title(f'{figure_title} E-plane Phi = {e_plane_plot_angle}')

    #ax2.plot(theta_f_deg, noError.e_plane_data_smooth , label=f'Radiation without errors', alpha=0.7)    
    ax2.plot(theta_f_deg, noError.e_plane_data_original, label=f'Radiation without errors', alpha=0.7)
    for idx, data in enumerate(dataErrorArr):
        # ax2.plot(theta_f_deg, data.e_plane_data_smooth, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)   
        ax2.plot(theta_f_deg, data.e_plane_data_original, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)
    
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()

    


TEST_NAME = 'position_both_pol_same_error_correlated_theta' # used to determine folder to output files
PATH_PREFIX = f'./spherical-NF-FF/testResults/{TEST_NAME}/'
FILE_PATH_SEARCH = f'{PATH_PREFIX}*/error_transformed_NF_FF_heatmap.txt'

phi_select_angle = 0
errorTxt = ['1mm', '5mm', '10mm', '20mm', '30mm', '50mm']
plotData = []

# load no error
ffData_no_error, theta_deg, phi_deg, _, _ = load_FF_data_own_output(f'./spherical-NF-FF/testResults/FF_data_no_error.txt')
plot_ffData_no_error = select_data_at_angle(ffData_no_error, phi_deg, phi_select_angle)
theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)

# load with errors, anmd select
matching_files = glob.glob(FILE_PATH_SEARCH)
for file_path in matching_files:
    ffData, _, _, _, _ = load_FF_data_own_output(file_path)
    data = select_data_at_angle(ffData, phi_deg, phi_select_angle)
    plotData.append(data)

plot_error_compare(plot_ffData_no_error, plotData, errorTxt, theta_deg_center, f'Radiation comparison of transform w/o error')
plt.savefig(PATH_PREFIX + 'compare_all.svg', bbox_inches='tight')

plt.tight_layout()
plt.show()