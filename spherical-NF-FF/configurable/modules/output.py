import numpy as np
import matplotlib.pyplot as plt

def plot_copolar(data, theta_f_deg, phi_f_deg, figure_title):

    theta_plot_angle = data.theta_plot_angle
    theta_angle_data_log_ori = 20 * np.log10(data.theta_angle_data_original)
    theta_angle_data_log_smo = 20 * np.log10(data.theta_angle_data_smooth)

    phi_plot_angle = data.phi_plot_angle
    phi_angle_data_log_ori = 20 * np.log10(data.phi_angle_data_original)
    phi_angle_data_log_smo = 20 * np.log10(data.phi_angle_data_smooth)
    
    # Plot the far-field patterns
    fig = plt.figure(figure_title, figsize=(8, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], width_ratios=[1])
    
    # Plot phi = 0 i.e theta values
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(phi_f_deg, theta_angle_data_log_smo , label=f'E_phi (Far Field) {theta_plot_angle} degree theta, copolar, with Savitzky-Golay filter', alpha=0.7)
    ax1.plot(phi_f_deg, theta_angle_data_log_ori, label=f'E_phi (Far Field) {theta_plot_angle} degree theta, copolar', alpha=0.7)    
    ax1.set_title(f'Normalized Far-field Pattern Theta = {theta_plot_angle}')
    ax1.set_xlabel('Phi')
    ax1.grid()
    ax1.legend()
    
    # Plot theta = 0 i.e phi values
    ax2 = fig.add_subplot(grid[1, 0])
    ax2.plot(theta_f_deg, phi_angle_data_log_smo , label=f'E_theta (Far Field) {phi_plot_angle} degree phi, copolar, with Savitzky-Golay filter', alpha=0.7)
    ax2.plot(theta_f_deg, phi_angle_data_log_ori, label=f'E_theta (Far Field) {phi_plot_angle} degree phi, copolar', alpha=0.7)    
    ax2.set_title(f'Normalized Far-field Pattern Phi = {phi_plot_angle}')
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()

def plot_dif(data, data2, theta_f_deg, figure_title):
    h_plane_plot_angle = data.h_plane_plot_angle
    h_plane_data_original = data.h_plane_data_original # 20 * np.log10(data.h_plane_data_original)
    h_plane_data_smooth = data.h_plane_data_smooth #20 * np.log10(data.h_plane_data_smooth)

    e_plane_plot_angle = data.e_plane_plot_angle
    e_plane_data_original = data.e_plane_data_original #20 * np.log10(data.e_plane_data_original)
    e_plane_data_smooth = data.e_plane_data_smooth #20 * np.log10(data.e_plane_data_smooth)

    h_plane_plot_angle2 = data2.h_plane_plot_angle
    h_plane_data_original2 = data2.h_plane_data_original # 20 * np.log10(data.h_plane_data_original)
    h_plane_data_smooth2 = data2.h_plane_data_smooth #20 * np.log10(data.h_plane_data_smooth)

    e_plane_plot_angle2 = data2.e_plane_plot_angle
    e_plane_data_original2 = data2.e_plane_data_original #20 * np.log10(data.e_plane_data_original)
    e_plane_data_smooth2 = data2.e_plane_data_smooth #20 * np.log10(data.e_plane_data_smooth)

    # Plot the far-field patterns
    fig = plt.figure(figure_title, figsize=(8, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], width_ratios=[1])

    ax1 = fig.add_subplot(grid[0, 0])
    #ax1.plot(theta_f_deg, abs(h_plane_data_smooth - h_plane_data_smooth2) / np.max(h_plane_data_smooth) , label=f'Dif plot', alpha=0.7)  
    ax1.plot(theta_f_deg, abs(h_plane_data_original - h_plane_data_original2) / np.max(h_plane_data_original) , label=f'Dif plot - original', alpha=0.7)  
    ax1.set_title(f'Far-field Pattern H-plane Phi = {h_plane_plot_angle}')
    ax1.set_xlabel('Theta')
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(grid[1, 0])
    #ax2.plot(theta_f_deg, abs(e_plane_data_smooth - e_plane_data_smooth2) / np.max(e_plane_data_smooth) , label=f'Dif plot', alpha=0.7)  
    ax2.plot(theta_f_deg, abs(e_plane_data_original - e_plane_data_original2) / np.max(e_plane_data_original) , label=f'Dif plot', alpha=0.7)  
    ax2.set_title(f'Far-field Pattern E-plane Phi = {e_plane_plot_angle}')
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()

def plot_error_compare(data, data2, theta_f_deg, figure_title):
    h_plane_plot_angle = data.h_plane_plot_angle
    h_plane_data_original = data.h_plane_data_original # 20 * np.log10(data.h_plane_data_original)
    h_plane_data_smooth = data.h_plane_data_smooth #20 * np.log10(data.h_plane_data_smooth)

    e_plane_plot_angle = data.e_plane_plot_angle
    e_plane_data_original = data.e_plane_data_original #20 * np.log10(data.e_plane_data_original)
    e_plane_data_smooth = data.e_plane_data_smooth #20 * np.log10(data.e_plane_data_smooth)

    h_plane_plot_angle2 = data2.h_plane_plot_angle
    h_plane_data_original2 = data2.h_plane_data_original # 20 * np.log10(data.h_plane_data_original)
    h_plane_data_smooth2 = data2.h_plane_data_smooth #20 * np.log10(data.h_plane_data_smooth)

    e_plane_plot_angle2 = data2.e_plane_plot_angle
    e_plane_data_original2 = data2.e_plane_data_original #20 * np.log10(data.e_plane_data_original)
    e_plane_data_smooth2 = data2.e_plane_data_smooth #20 * np.log10(data.e_plane_data_smooth)

    # Plot the far-field patterns
    fig = plt.figure(figure_title, figsize=(8, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], width_ratios=[1])

    ax1 = fig.add_subplot(grid[0, 0])
    #ax1.plot(theta_f_deg, h_plane_data_smooth , label=f'Radiation plot without errors', alpha=0.7)
    #ax1.plot(theta_f_deg, h_plane_data_smooth2, label=f'Radiation plot with errors', alpha=0.7)    
    ax1.plot(theta_f_deg, h_plane_data_original , label=f'Radiation plot without errors - original', alpha=0.7)
    ax1.plot(theta_f_deg, h_plane_data_original2, label=f'Radiation plot with errors - original', alpha=0.7)    
    ax1.set_title(f'Far-field Pattern H-plane Phi = {h_plane_plot_angle}')
    ax1.set_xlabel('Theta')
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(grid[1, 0])
    #ax2.plot(theta_f_deg, e_plane_data_smooth , label=f'Radiation plot without errors', alpha=0.7)
    #ax2.plot(theta_f_deg, e_plane_data_smooth2, label=f'Radiation plot with errors', alpha=0.7)   
    ax2.plot(theta_f_deg, e_plane_data_original , label=f'Radiation plot without errors - original', alpha=0.7)
    ax2.plot(theta_f_deg, e_plane_data_original2, label=f'Radiation plot with errors - original', alpha=0.7)    
    ax2.set_title(f'Far-field Pattern E-plane Phi = {e_plane_plot_angle}')
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()

def plot_copolar2(data, theta_f_deg, figure_title):
    h_plane_plot_angle = data.h_plane_plot_angle
    h_plane_data_original = data.h_plane_data_original # 20 * np.log10(data.h_plane_data_original)
    h_plane_data_smooth = data.h_plane_data_smooth #20 * np.log10(data.h_plane_data_smooth)

    e_plane_plot_angle = data.e_plane_plot_angle
    e_plane_data_original = data.e_plane_data_original #20 * np.log10(data.e_plane_data_original)
    e_plane_data_smooth = data.e_plane_data_smooth #20 * np.log10(data.e_plane_data_smooth)

    # Plot the far-field patterns
    fig = plt.figure(figure_title, figsize=(8, 10))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], width_ratios=[1])

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(theta_f_deg, h_plane_data_smooth , label=f'smoothed with Savitzky-Golay filter', alpha=0.7)
    ax1.plot(theta_f_deg, h_plane_data_original, label=f'no smoothing', alpha=0.7)    
    ax1.set_title(f'Far-field Pattern H-plane Phi = {h_plane_plot_angle}')
    ax1.set_xlabel('Theta')
    ax1.grid()
    ax1.legend()

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.plot(theta_f_deg, e_plane_data_smooth , label=f'smoothed with Savitzky-Golay filter', alpha=0.7)
    ax2.plot(theta_f_deg, e_plane_data_original, label=f'no smoothing', alpha=0.7)    
    ax2.set_title(f'Far-field Pattern E-plane Phi = {e_plane_plot_angle}')
    ax2.set_xlabel('Theta')
    ax2.grid()
    ax2.legend()


def plot_heatmap(ffData, theta_f_deg, phi_f_deg, figure_title):

    # Heatmap (Bottom, centered across both columns)
    plt.figure(figure_title)
    ax3 = plt.subplot(1, 1, 1)
    cax = ax3.imshow(ffData, cmap='hot', aspect='auto')
    ax3.set_title('Far-Field Radiation Pattern Heatmap')

    plt.colorbar(cax, ax=ax3, label='Far-field amplitude (normalized)')

    ax3.set_xlabel('Phi')
    ax3.set_ylabel('Theta °')
    
    # Set x-ticks and y-ticks with a reduced number of labels
    xtick_step = max(1, len(phi_f_deg) // 13)  # Show every nth phi label
    ytick_step = max(1, len(theta_f_deg) // 10)  # Show every nth theta label

    ax3.set_xticks(np.arange(0, len(phi_f_deg), xtick_step))
    ax3.set_xticklabels(phi_f_deg[::xtick_step])

    ax3.set_yticks(np.arange(0, len(theta_f_deg), ytick_step))
    ax3.set_yticklabels(theta_f_deg[::ytick_step])
    

def plot_polar(data, theta_f, phi_f, figure_title):
    # select data and roll back, roll ensures center of main lobe is at 0 deg
    theta_plot_angle = data.theta_plot_angle
    h_plane_magnitude = 20 * np.log10(data.theta_angle_data_smooth)
    h_plane_magnitude = np.roll(h_plane_magnitude, len(h_plane_magnitude) // 2)

    phi_plot_angle = data.phi_plot_angle
    e_plane_magnitude = 20 * np.log10(data.phi_angle_data_smooth)
    e_plane_magnitude = np.roll(e_plane_magnitude, len(e_plane_magnitude) // 2)
    
    # Create the figure and the gridspec
    fig = plt.figure(figure_title)
    grid = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[1, 1])

    # E-plane polar plot
    ax1 = fig.add_subplot(grid[0, 0], projection='polar')
    ax1.plot(theta_f, e_plane_magnitude)
    ax1.set_title(f'E-Plane (Phi = {phi_plot_angle})')

    # H-plane polar plot
    ax2 = fig.add_subplot(grid[0, 1], projection='polar')
    ax2.plot(phi_f, h_plane_magnitude)
    ax2.set_title(f'H-Plane (Theta = {theta_plot_angle})')


def plot_polar2(data, theta_f, figure_title):
    # select data and roll back, roll ensures center of main lobe is at 0 deg
    h_plane_plot_angle = data.h_plane_plot_angle
    h_plane_magnitude = 20 * np.log10(data.h_plane_data_smooth)

    e_plane_plot_angle = data.e_plane_plot_angle
    e_plane_magnitude = 20 * np.log10(data.e_plane_data_smooth)
    
    # Create the figure and the gridspec
    fig = plt.figure(figure_title)
    grid = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[1, 1])

    # E-plane polar plot
    ax1 = fig.add_subplot(grid[0, 0], projection='polar')
    ax1.plot(theta_f, e_plane_magnitude)
    ax1.set_title(f'E-Plane (Phi = {e_plane_plot_angle})')

    # H-plane polar plot
    ax2 = fig.add_subplot(grid[0, 1], projection='polar')
    ax2.plot(theta_f, h_plane_magnitude)
    ax2.set_title(f'H-Plane (Phi = {h_plane_plot_angle})')


def show_figures():
    plt.tight_layout()
    plt.show()


def calculate_hpbw(data, angles):
    """
    Calculate the Half-Power Beamwidth (HPBW) of a signal.

    Parameters:
    - data: Array of far-field data points (e.g., power or intensity values).
    - angles: Array of corresponding angles in degrees.

    Returns:
    - hpbw: The calculated HPBW in degrees.
    """

    # Ensure data and angles have the same length
    if len(data) != len(angles):
        raise ValueError("Data and angles arrays must have the same length")

    # Find the index of the maximum value in the data array
    max_index = np.argmax(data)
    max_value = data[max_index]

    # Calculate the half-power level (-3 dB point)
    half_power_level = max_value / 2.0

    # Find the indices where data crosses the half-power level on both sides of max
    left_index = np.where(data[:max_index] <= half_power_level)[0]
    right_index = np.where(data[max_index:] <= half_power_level)[0] + max_index

    if len(left_index) == 0 or len(right_index) == 0:
        raise ValueError("Cannot find -3 dB points on both sides of the main lobe")

    # Get the angle values for the -3 dB points
    left_angle = angles[left_index[-1]]
    right_angle = angles[right_index[0]]

    # Calculate the HPBW
    hpbw = np.abs(right_angle - left_angle)

    if (hpbw > 180):
        hpbw = 360 - hpbw
    
    return np.round(hpbw, 2)


def calculate_hpbw_linear_approx(data, angles):
    """
    Calculate the Half-Power Beamwidth (HPBW) of a signal using linear interpolation for precision.

    Parameters:
    - data: Array of far-field data points (e.g., power or intensity values).
    - angles: Array of corresponding angles in degrees.

    Returns:
    - hpbw: The calculated HPBW in degrees.
    """

    # Ensure data and angles have the same length
    if len(data) != len(angles):
        raise ValueError("Data and angles arrays must have the same length")

    # Find the index of the maximum value in the data array
    max_index = np.argmax(data)
    max_value = data[max_index]

    # Calculate the half-power level (-3 dB point)
    half_power_level = max_value / 2.0

    # Helper function for linear interpolation
    def interpolate(x1, y1, x2, y2, target_y):
        """Linear interpolation to find x for a given y."""
        return x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)

    # Find the crossing point on the left
    left_index = np.where(data[:max_index] > half_power_level)[0][-1]  # Last point above
    left_angle = interpolate(
        angles[left_index], data[left_index],
        angles[left_index + 1], data[left_index + 1],
        half_power_level
    )

    # Find the crossing point on the right
    right_index = max_index + np.where(data[max_index:] > half_power_level)[0][0]  # First point above
    right_angle = interpolate(
        angles[right_index - 1], data[right_index - 1],
        angles[right_index], data[right_index],
        half_power_level
    )

    # Calculate the HPBW
    hpbw = np.abs(right_angle - left_angle)

    # Handle wrap-around for angles
    if hpbw > 180:
        hpbw = 360 - hpbw

    return np.round(hpbw, 2)

def calculate_print_hpbw(data, theta_deg_center):
    h_plane_hpbw_smooth = calculate_hpbw(data.h_plane_data_smooth, theta_deg_center)
    h_plane_hpbw_original = calculate_hpbw(data.h_plane_data_original, theta_deg_center)
    e_plane_hpbw_smooth = calculate_hpbw(data.e_plane_data_smooth, theta_deg_center)
    e_plane_hpbw_original = calculate_hpbw(data.e_plane_data_original, theta_deg_center)

    #print(f"H-plane (smoothed) HPBW: {h_plane_hpbw_smooth}deg; H-plane (original) HPBW: {h_plane_hpbw_original} deg")
    #print(f"E-plane (smoothed) HPBW: {e_plane_hpbw_smooth} deg; E-plane (original) HPBW: {e_plane_hpbw_original} deg")

    #print(f"H-plane (smoothed) HPBW: {h_plane_hpbw_smooth}deg")
    #print(f"E-plane (smoothed) HPBW: {e_plane_hpbw_smooth} deg")

    print(f"H-plane (original) HPBW: {h_plane_hpbw_original} deg")
    print(f"E-plane (original) HPBW: {e_plane_hpbw_original} deg")

def calculate_mean_indexed_error(data1, data2):
    # Compute the absolute differences at each index
    differences = np.abs(data1 - data2)

    # Return the mean value
    return np.mean(differences)


def calculate_max_indexed_error(data1, data2):
    # Compute the absolute differences at each index
    differences = np.abs(data1 - data2)
    
    # Return the maximum difference
    return np.max(differences)