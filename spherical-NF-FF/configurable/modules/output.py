import numpy as np
import matplotlib.pyplot as plt

def plot_ff_at(smoothed_ff, original_ff, theta_f, phi_f):

    shifted_phi = (phi_f * 180 / np.pi) #np.roll(, int(len(phi_f) / 2))
    # Plot the far-field patterns
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(shifted_phi, 20 * np.log10(smoothed_ff), label='E_phi (Far Field) 0 degrees phi, copolar, with Savitzky-Golay filter', alpha=0.7)
    ax1.plot(shifted_phi, 20 * np.log10(original_ff), label='E_phi (Far Field) 0 degrees phi, copolar', alpha=0.7)
    # ax1.plot(phi_f, 20 * np.log10(E_far_mod[17, :]), label='E_phi (Far Field) 90 degrees phi', alpha=0.7)
    # ax1.plot(theta_f, np.log10(E_far_mod[:, 0]), label='E_theta (Far Field)', alpha=0.7)
    ax1.set_title('Normalized Far-field Pattern')
    ax1.grid()
    ax1.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(ff):
    # change to dbi
    # far_field_dbi = 20 * np.log10(ff)

    # Heatmap (Bottom, centered across both columns)
    #ax3 = fig.add_subplot(grid[1, :])
    ax3 = plt.subplot(1, 1, 1)
    cax = ax3.imshow(ff, cmap='hot', aspect='auto') #extent=[-1, 1, -1, 1],
    #fig.colorbar(cax, ax=ax3, label='Far-field amplitude (normalized)')
    ax3.set_title('Far-Field Radiation Pattern Heatmap')
    #ax3.set_xlabel('K_Y (1/m)')
    #ax3.set_ylabel('K_Z (1/m)')
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
