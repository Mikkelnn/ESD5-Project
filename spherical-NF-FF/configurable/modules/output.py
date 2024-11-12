import numpy as np
import matplotlib.pyplot as plt

def plot_ff_at(smoothed_ff, original_ff, angles_f, axisName, title='0 degrees phi'):
   
    log_smo_data = 20 * np.log10(smoothed_ff)
    log_ori_data = 20 * np.log10(original_ff)
    
    # Plot the far-field patterns
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(angles_f, log_smo_data , label=f'E_phi (Far Field) {title}, copolar, with Savitzky-Golay filter', alpha=0.7)
    ax1.plot(angles_f, log_ori_data, label=f'E_phi (Far Field) {title}, copolar', alpha=0.7)

    ax1.set_title('Normalized Far-field Pattern')
    ax1.set_xlabel(axisName)
    ax1.grid()
    ax1.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(fData, theta_f_deg, phi_f_deg):

    # Heatmap (Bottom, centered across both columns)
    ax3 = plt.subplot(1, 1, 1)
    cax = ax3.imshow(fData, cmap='hot', aspect='auto')
    ax3.set_title('Far-Field Radiation Pattern Heatmap')

    plt.colorbar(cax, ax=ax3, label='Far-field amplitude (normalized)')

    ax3.set_xlabel('Phi degree')
    ax3.set_ylabel('Theta degree')
    
    ax3.set_xticks(np.arange(len(phi_f_deg)), labels=phi_f_deg)
    #ax3.set_yticks(len(theta_f_deg), theta_f_deg)

    plt.tight_layout()
    plt.show()


def plot_polar(ffData, theta, phi):
    
    e_plane_magnitude = 20 * np.log10(ffData[:, ffData.shape[1] // 2])  # Cut at phi = 0
    #e_plane_magnitude_limited = e_plane_magnitude[valid_indices]  # Limit to valid angular range
    
    # H-plane: when theta = 0 (elevation is constant)
    h_plane_magnitude = 20 * np.log10(ffData[ffData.shape[0] // 2 , :])  # Cut at theta = 0
    #h_plane_magnitude_limited = h_plane_magnitude[valid_indices]  # Limit to valid angular range
    
    # Create the figure and the gridspec
    fig = plt.figure(figsize=(10, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], width_ratios=[1, 1])

    # E-plane polar plot (Top Left)
    ax1 = fig.add_subplot(grid[0, 0], projection='polar')
    ax1.plot(theta, e_plane_magnitude)
    ax1.set_title('E-Plane (Limited Angle Range)')

    # H-plane polar plot (Top Right)
    ax2 = fig.add_subplot(grid[0, 1], projection='polar')
    ax2.plot(phi, h_plane_magnitude)
    ax2.set_title('H-Plane (Limited Angle Range)')

    plt.tight_layout()
    plt.show()



import numpy as np

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
    
    return hpbw
