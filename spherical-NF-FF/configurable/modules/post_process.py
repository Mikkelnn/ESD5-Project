import numpy as np
from scipy.signal import savgol_filter  # Savitzky-Golay filter
from collections import namedtuple

def sum_NF_poles(nfData):
    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    return abs(nfData[:, :, 0]) + abs(nfData[:, :, 1])

def sum_NF_poles_sqrt(nfData):
    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    return (abs(nfData[:, :, 0])**2 + abs(nfData[:, :, 1])**2)**0.5


def select_data_at_angle_old(ffData, theta_f_deg, phi_f_deg, theta_select_angle=0, phi_select_angle=0):
  # variabels used for smoothing
  window_size = 9 # Choose an odd window size
  poly_order = 2    # Polynomial order for smoothing

  # deterimne sample number corresponding to plot angles
  theta_index = np.absolute(theta_f_deg - theta_select_angle).argmin() - 1
  theta_plot_angle = theta_f_deg[theta_index]

  phi_index = np.absolute(phi_f_deg - phi_select_angle).argmin()
  phi_plot_angle = phi_f_deg[phi_index]

  # Select data at theta degrees and smooth it with Savitzky-Golay filter
  theta_angle_data_original = ffData[theta_index, :]

  # Select data at phi degrees and smooth it with Savitzky-Golay filter
  #get first half:
  phi_angle_data_original = ffData[:, phi_index]

  # Apply Savitzky-Golay filter for smoothing
  theta_angle_data_smooth = savgol_filter(theta_angle_data_original, window_size, poly_order)

  # Apply Savitzky-Golay filter for smoothing    
  phi_angle_data_smooth = savgol_filter(phi_angle_data_original, window_size, poly_order)

  Desc = namedtuple("Desc", ["theta_plot_angle", "theta_angle_data_original", "theta_angle_data_smooth", "phi_plot_angle", "phi_angle_data_original", "phi_angle_data_smooth"])
  return Desc(
      theta_plot_angle,
      theta_angle_data_original,
      theta_angle_data_smooth,

      phi_plot_angle,
      phi_angle_data_original,
      phi_angle_data_smooth
  )

def select_data_at_angle(ffData, phi_f_deg, phi_select_angle=0):
  # variabels used for smoothing
  window_size = 5 # Choose an odd window size
  poly_order = 2    # Polynomial order for smoothing

  # deterimne sample number corresponding to plot angles
  phi_index_0 = np.absolute(phi_f_deg - phi_select_angle).argmin()
  phi_index_180 = np.absolute(phi_f_deg - ((phi_select_angle + 180) % 360)).argmin()

  phi_index_90 = np.absolute(phi_f_deg - ((phi_select_angle + 90) % 360)).argmin()
  phi_index_270 = np.absolute(phi_f_deg - ((phi_select_angle + 270) % 360)).argmin()
  
  h_plane_plot_angle = phi_f_deg[phi_index_0]
  e_plane_plot_angle = phi_f_deg[phi_index_90]

  # Select data
  h_plane_data_original = np.concatenate((np.flip(ffData[:, phi_index_0]), ffData[1:, phi_index_180]))
  e_plane_data_original = np.concatenate((np.flip(ffData[:, phi_index_90]), ffData[1:, phi_index_270]))

  # Apply Savitzky-Golay filter for smoothing
  h_plane_data_smooth = savgol_filter(h_plane_data_original, window_size, poly_order) 
  e_plane_data_smooth = savgol_filter(e_plane_data_original, window_size, poly_order)

  Desc = namedtuple("Desc", ["h_plane_plot_angle", "h_plane_data_original", "h_plane_data_smooth", "e_plane_plot_angle", "e_plane_data_original", "e_plane_data_smooth"])
  return Desc(
      h_plane_plot_angle,
      h_plane_data_original,
      h_plane_data_smooth,

      e_plane_plot_angle,
      e_plane_data_original,
      e_plane_data_smooth
  )
