import matplotlib as plt
from modules.loadData import load_FF_data_own_output
from modules.post_process import select_data_at_angle

def plot_error_compare(noError, dataErrorArr, errorTxt, theta_f_deg, figure_title):
    h_plane_plot_angle = noError.h_plane_plot_angle
    e_plane_plot_angle = noError.e_plane_plot_angle

    # Plot the far-field patterns
    fig = plt.figure(figure_title, figsize=(8, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], width_ratios=[1])

    ax1 = fig.add_subplot(grid[0, 0])    
    ax1.set_title(f'{figure_title} H-plane Phi = {h_plane_plot_angle}')
    ax1.set_xlabel('Theta')
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(grid[1, 0])    
    ax2.set_title(f'{figure_title} E-plane Phi = {e_plane_plot_angle}')
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()

    #ax1.plot(theta_f_deg, noError.h_plane_data_smooth , label=f'Radiation without errors', alpha=0.7)    
    ax1.plot(theta_f_deg, noError.h_plane_data_original, label=f'Radiation without errors', alpha=0.7)
    
    #ax2.plot(theta_f_deg, noError.e_plane_data_smooth , label=f'Radiation without errors', alpha=0.7)    
    ax2.plot(theta_f_deg, noError.e_plane_data_original, label=f'Radiation without errors', alpha=0.7)

    for data, idx in enumerate(dataErrorArr):
        ax1.plot(theta_f_deg, data.h_plane_data_smooth, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)    
        ax1.plot(theta_f_deg, data.h_plane_data_original, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)

        ax2.plot(theta_f_deg, data.e_plane_data_smooth, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)   
        ax2.plot(theta_f_deg, data.e_plane_data_original, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)


TEST_NAME = 'position_both_pol_same_error_correlated_theta' # used to determine folder to output files
PATH_PREFIX = f'./spherical-NF-FF/testResults/{TEST_NAME}/*/error_transformed_NF_FF_heatmap.txt'

ffData_no_error, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_FF_data_own_output(file_path)

ffData, _, _, _, _ = load_FF_data_own_output(file_path)


# 'error_transformed_NF_FF_heatmap'


plt.tight_layout()
plt.show()