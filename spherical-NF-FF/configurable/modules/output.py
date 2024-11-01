import numpy as np
import matplotlib as plt

def plot_ff_at(smoothed_ff, original_ff):
    # Plot the far-field patterns
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(phi_f, 20 * np.log10(smoothed_ff), label='E_phi (Far Field) 0 degrees phi, copolar, with Savitzky-Golay filter', alpha=0.7)
    ax1.plot(phi_f, 20 * np.log10(original_ff), label='E_phi (Far Field) 0 degrees phi, copolar', alpha=0.7)
    # ax1.plot(phi_f, 20 * np.log10(E_far_mod[17, :]), label='E_phi (Far Field) 90 degrees phi', alpha=0.7)
    # ax1.plot(theta_f, np.log10(E_far_mod[:, 0]), label='E_theta (Far Field)', alpha=0.7)
    ax1.set_title('Normalized Far-field Pattern')
    ax1.grid()
    ax1.legend()
    plt.tight_layout()
    plt.show()