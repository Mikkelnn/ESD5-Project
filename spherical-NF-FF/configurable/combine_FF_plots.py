import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from modules.loadData import load_FF_data_own_output
from modules.post_process import select_data_at_angle
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    
    # ax1.plot(theta_f_deg, noError.h_plane_data_smooth, label=f'Radiation without errors', alpha=0.7)
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

    # ax2.plot(theta_f_deg, noError.e_plane_data_smooth , label=f'Radiation without errors', alpha=0.7)    
    ax2.plot(theta_f_deg, noError.e_plane_data_original, label=f'Radiation without errors', alpha=0.7)
    for idx, data in enumerate(dataErrorArr):
        # ax2.plot(theta_f_deg, data.e_plane_data_smooth, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)   
        ax2.plot(theta_f_deg, data.e_plane_data_original, label=f'Radiation with errors ({errorTxt[idx]})', alpha=0.7)
    
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()

def plot_error_compare_grouped(noError, dataErrorArr, errorTxt, theta_f_deg, figure_title, errorType='Error', errorTitle='error'):
    h_plane_plot_angle = noError.h_plane_plot_angle
    e_plane_plot_angle = noError.e_plane_plot_angle

    # Split dataErrorArr and errorTxt into two groups
    mid_point = len(dataErrorArr) // 2
    dataErrorArr_group1 = dataErrorArr[:mid_point]
    dataErrorArr_group2 = dataErrorArr[mid_point:]
    errorTxt_group1 = errorTxt[:mid_point]
    errorTxt_group2 = errorTxt[mid_point:]

    # Set up the figure for grouped subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Add shared titles for rows
    fig.suptitle(figure_title, fontsize=16)

    ###
    # H-plane Group 1 Plot
    ###
    ax_h1 = axes[0, 0]  # Top-left for H-plane Group 1
    ax_h1.set_title(f'H-plane | Phi = {h_plane_plot_angle}')

    # Plot "no error" data
    ax_h1.plot(theta_f_deg, noError.h_plane_data_original, label=f'No {errorTitle}', alpha=0.7, linewidth=2)

    # Plot data with errors (Group 1)
    for idx, data in enumerate(dataErrorArr_group1):
        ax_h1.plot(theta_f_deg, data.h_plane_data_original, 
                   label=f'{errorType}: {errorTxt_group1[idx]}', alpha=0.7)

    ax_h1.set_xlabel('Theta')
    ax_h1.set_ylabel('Amplitude')
    ax_h1.grid()
    ax_h1.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

    ###
    # H-plane Group 2 Plot
    ###
    ax_h2 = axes[0, 1]  # Top-right for H-plane Group 2
    ax_h2.set_title(f'H-plane | Phi = {h_plane_plot_angle}')

    # Plot "no error" data
    ax_h2.plot(theta_f_deg, noError.h_plane_data_original, label=f'No {errorTitle}', alpha=0.7, linewidth=2)

    # Plot data with errors (Group 2)
    for idx, data in enumerate(dataErrorArr_group2):
        ax_h2.plot(theta_f_deg, data.h_plane_data_original, 
                   label=f'{errorType}: {errorTxt_group2[idx]}', alpha=0.7)

    ax_h2.set_xlabel('Theta')
    ax_h2.set_ylabel('Amplitude')
    ax_h2.grid()
    ax_h2.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

    ###
    # E-plane Group 1 Plot
    ###
    ax_e1 = axes[1, 0]  # Bottom-left for E-plane Group 1
    ax_e1.set_title(f'E-plane | Phi = {e_plane_plot_angle}')

    # Plot "no error" data
    ax_e1.plot(theta_f_deg, noError.e_plane_data_original, label=f'No {errorTitle}', alpha=0.7, linewidth=2)

    # Plot data with errors (Group 1)
    for idx, data in enumerate(dataErrorArr_group1):
        ax_e1.plot(theta_f_deg, data.e_plane_data_original, 
                   label=f'{errorType}: {errorTxt_group1[idx]}', alpha=0.7)

    ax_e1.set_xlabel('Theta')
    ax_e1.set_ylabel('Amplitude')
    ax_e1.grid()
    ax_e1.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

    ###
    # E-plane Group 2 Plot
    ###
    ax_e2 = axes[1, 1]  # Bottom-right for E-plane Group 2
    ax_e2.set_title(f'E-plane | Phi = {e_plane_plot_angle}')

    # Plot "no error" data
    ax_e2.plot(theta_f_deg, noError.e_plane_data_original, label=f'No {errorTitle}', alpha=0.7, linewidth=2)

    # Plot data with errors (Group 2)
    for idx, data in enumerate(dataErrorArr_group2):
        ax_e2.plot(theta_f_deg, data.e_plane_data_original, 
                   label=f'{errorType}: {errorTxt_group2[idx]}', alpha=0.7)

    ax_e2.set_xlabel('Theta')
    ax_e2.set_ylabel('Amplitude')
    ax_e2.grid()
    ax_e2.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

    # Adjust layout for better visualization
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

def generateSaveCompareImage(PATH_PREFIX, reverseOrder, phi_select_angle, compareToPath, titleSuffix='error', legendType='Error'):
    FILE_PATH_SEARCH = f'{PATH_PREFIX}*/error_transformed_NF_FF_heatmap.txt'

    errorTxt = []
    plotData = []

    # load no error
    ffData_no_error, theta_deg, phi_deg, _, _ = load_FF_data_own_output(compareToPath)
    plot_ffData_no_error = select_data_at_angle(ffData_no_error, phi_deg, phi_select_angle)
    theta_deg_center = np.linspace(-np.max(theta_deg), np.max(theta_deg), (len(theta_deg)*2)-1)

    # Extract the numeric part of the folder name and sort paths
    def extract_numeric_key(path):
        match = re.search(r'[\\/]+([\dE+-]+)(mm|dB)?[\\/]', path)  # Find a number followed by "mm" in the path
        return float(match.group(1)) if match else float('inf')  # Default to 'inf' if no match is found

    # Find and sort all matching file paths
    matching_files = sorted(glob.glob(FILE_PATH_SEARCH), key=extract_numeric_key, reverse=reverseOrder)

    for file_path in matching_files:
        # errorTxt.append(extract_numeric_key(file_path))
        errorTxt.append(file_path.removeprefix(PATH_PREFIX).split('/')[0])
        ffData, _, _, _, _ = load_FF_data_own_output(file_path)
        data = select_data_at_angle(ffData, phi_deg, phi_select_angle)
        plotData.append(data)

    plot_error_compare_grouped(plot_ffData_no_error, plotData, errorTxt, theta_deg_center, f'Radiation comparison of transform w/o {titleSuffix}', legendType, titleSuffix)
    plt.savefig(PATH_PREFIX + 'compare_all_original.svg', bbox_inches='tight')

    # plt.tight_layout()
    # plt.show()


def generateCompareImageFromTestDescriptors(rootPath, descriptors, phi_select_angle=0, compareToPath=f'./spherical-NF-FF/testResults/FF_data_no_error.txt', showProgress=True):
    for descriptor in tqdm(descriptors, disable=(not showProgress)):
        generateSaveCompareImage(f'{rootPath}/{descriptor.testName}', descriptor.reverseTableRowOrder, phi_select_angle, compareToPath, titleSuffix=descriptor.titleSuffix, legendType=descriptor.legendType)
