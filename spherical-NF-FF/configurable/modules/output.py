import numpy as np
import matplotlib.pyplot as plt
import random

def seed():
    return random.randint(1, 1000)

def plot_copolar(data, theta_f_deg, phi_f_deg):

    theta_plot_angle = data.theta_plot_angle
    theta_angle_data_log_ori = 20 * np.log10(data.theta_angle_data_original)
    theta_angle_data_log_smo = 20 * np.log10(data.theta_angle_data_smooth)

    phi_plot_angle = data.phi_plot_angle
    phi_angle_data_log_ori = 20 * np.log10(data.phi_angle_data_original)
    phi_angle_data_log_smo = 20 * np.log10(data.phi_angle_data_smooth)
    
    # Plot the far-field patterns
    fig = plt.figure(seed(), figsize=(8, 10))
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
    

def plot_heatmap(ffData, theta_f_deg, phi_f_deg):

    # Heatmap (Bottom, centered across both columns)
    plt.figure(seed())
    ax3 = plt.subplot(1, 1, 1)
    cax = ax3.imshow(ffData, cmap='hot', aspect='auto')
    ax3.set_title('Far-Field Radiation Pattern Heatmap')

    plt.colorbar(cax, ax=ax3, label='Far-field amplitude (normalized)')

    ax3.set_xlabel('Phi degree')
    ax3.set_ylabel('Theta degree')
    
    ax3.set_xticks(np.arange(len(phi_f_deg)), labels=phi_f_deg)
    ax3.set_yticks(np.arange(len(theta_f_deg)), labels=theta_f_deg)



def plot_polar(data, theta_f, phi_f):

    theta_plot_angle = data.theta_plot_angle
    h_plane_magnitude = 20 * np.log10(data.theta_angle_data_smooth)
    h_plane_magnitude = np.roll(h_plane_magnitude, len(h_plane_magnitude) // 2)

    phi_plot_angle = data.phi_plot_angle
    e_plane_magnitude = 20 * np.log10(data.phi_angle_data_smooth)
    e_plane_magnitude = np.roll(e_plane_magnitude, len(e_plane_magnitude) // 2)
    
    # Create the figure and the gridspec
    fig = plt.figure(seed())
    grid = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[1, 1])

    # E-plane polar plot
    ax1 = fig.add_subplot(grid[0, 0], projection='polar')
    ax1.plot(theta_f, e_plane_magnitude)
    ax1.set_title(f'E-Plane (Phi = {phi_plot_angle})')

    # H-plane polar plot
    ax2 = fig.add_subplot(grid[0, 1], projection='polar')
    ax2.plot(phi_f, h_plane_magnitude)
    ax2.set_title(f'H-Plane (Theta = {theta_plot_angle})')



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
