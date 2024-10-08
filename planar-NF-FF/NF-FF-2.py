import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importing tqdm for progress bar
import os  # For checking file existence

def simulate_near_field_dipole(antenna_size, wavelength, plane_size, z_distance, num_points=500):
    k = 2 * np.pi / wavelength  # Wavenumber

    # Define a 2D grid for the measurement plane (in y-z plane at x = z_distance)
    y = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    z = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    Y, Z = np.meshgrid(y, z)

    # Coordinates of the dipole's center
    dipole_top = np.array([0, 0, antenna_size / 2])
    dipole_bottom = np.array([0, 0, -(antenna_size / 2)])

    # Calculate distance R from dipole to each point on the measurement plane
    R_top = np.sqrt((z_distance - dipole_top[0])**2 + (Y - dipole_top[1])**2 + (Z - dipole_top[2])**2)
    R_bottom = np.sqrt((z_distance - dipole_bottom[0])**2 + (Y - dipole_bottom[1])**2 + (Z - dipole_bottom[2])**2)

    # Electric field components (simplified near-field model)
    E_top = np.sin(np.arctan2(np.sqrt(Y**2 + Z**2), z_distance)) / R_top * np.exp(-1j * k * R_top)
    E_bottom = np.sin(np.arctan2(np.sqrt(Y**2 + Z**2), z_distance)) / R_bottom * np.exp(-1j * k * R_bottom)

    # Total near-field data (superposition of the fields from the top and bottom of the dipole)
    near_field = E_top + E_bottom
    
    return near_field, Y, Z

def horn_near_field_precise(wavelength, aperture_width, aperture_height, distance, plane_width, plane_height, num_points_w, num_points_h, num_aperture_points=200):
    # Derived parameters
    k = 2 * np.pi / wavelength  # Wavenumber

    # Create grid for the observation plane
    x_plane = np.linspace(-plane_width / 2, plane_width / 2, num_points_w)
    y_plane = np.linspace(-plane_height / 2, plane_height / 2, num_points_h)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    
    # Create grid for the aperture (higher resolution for precision)
    x_aperture = np.linspace(-aperture_width / 2, aperture_width / 2, num_aperture_points)
    y_aperture = np.linspace(-aperture_height / 2, aperture_height / 2, num_aperture_points)
    X_aperture, Y_aperture = np.meshgrid(x_aperture, y_aperture)

    # Initialize field on the observation plane
    E_field_plane = np.zeros((num_points_w, num_points_h), dtype=complex)

    # Define an aperture field distribution (e.g., a cosine distribution for TE10 mode)
    # You can modify this to use a Gaussian or other distribution if needed
    E_aperture = np.cos(np.pi * X_aperture / aperture_width) * np.cos(np.pi * Y_aperture / aperture_height)
    
    # Loop over points on the aperture to calculate contribution at each observation point
    for i in tqdm(range(num_aperture_points), desc="Simulating Aperture Points"):
        for j in range(num_aperture_points):
            # Position of current aperture point
            x_a = X_aperture[i, j]
            y_a = Y_aperture[i, j]
            
            # Distance from aperture point to each point on the observation plane
            R = np.sqrt((X_plane - x_a)**2 + (Y_plane - y_a)**2 + distance**2)
            
            # Fresnel Diffraction Approximation: Contribution from this aperture point
            E_field_plane += E_aperture[i, j] * np.exp(-1j * k * R) / R * (1j / wavelength) * np.exp(-1j * k * distance)

    # Normalize by the number of aperture points to ensure reasonable magnitude
    E_field_plane /= num_aperture_points**2

    return E_field_plane


def nf_ff_transform(near_field, wavelength, plane_size):
    num_points = near_field.shape[0]
    k = 2 * np.pi / wavelength  # Wavenumber
    
    # Spatial frequency grid in the plane (for Fourier transform)
    delta_yz = plane_size / num_points  # Spatial resolution in the y and z directions
    ky = np.fft.fftfreq(num_points, d=delta_yz) * 2 * np.pi
    kz = np.fft.fftfreq(num_points, d=delta_yz) * 2 * np.pi
    KY, KZ = np.meshgrid(ky, kz)
    
    # Perform 2D Fourier transform (with shift to center frequencies)
    far_field_2D = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(near_field)))
    
    # Calculate corresponding far-field angles based on wavenumber
    #theta_far = np.arctan2(np.sqrt(KY**2 + KZ**2), k)  # Far-field angle (theta)

    # Normalize the far-field pattern
    #far_field_pattern = np.abs(far_field_2D) #/ np.max(np.abs(far_field_2D))

    # Limit the far-field data to the valid angular range (based on antenna size and plane)
    inner_angle = np.arctan2((plane_size / 2), z_distance)  # Maximum valid angle from the plane
    angle_min = np.pi/2 - inner_angle
    angle_max = np.pi/2 + inner_angle

    # Define angle ranges
    angle_range = np.linspace(angle_min, angle_max, num_points)  # Elevation angle range
    
    # radius distance to each point in sample grid
    r_vals = np.sqrt(KY**2 + KZ**2 + z_distance**2)

    for i in range(far_field_2D.shape[0]):
        for j in range(far_field_2D.shape[1]):
            theta_pos = angle_range[j]
            r = r_vals[i][j]
            far_field_2D[i][j] = (1j * ((k * np.exp(-1j * k * r)) / (2 * np.pi * r)) * np.cos(theta_pos) * far_field_2D[i][j])

    #far_field_pattern = (1j((k * np.exp(-1j * k * theta_far)) / (2 * np.pi * theta_far)) * np.cos(theta_far) * far_field_pattern)    

    far_field_pattern = np.abs(far_field_2D) / np.max(np.abs(far_field_2D))

    #print(f"FF-trans: {far_field_pattern}")

    return far_field_pattern, angle_range, ky, kz

def plot_field_heatmap(field_pattern):
    """
    Plot the field pattern as a 2D heat map in the translated plane.
    
    Parameters:
    - far_field_pattern: The field amplitude data (2D array).
    - theta_far: The field angles in the elevation plane (in radians).
    - ky: Spatial frequency corresponding to the y direction.
    - kz: Spatial frequency corresponding to the z direction.
    """

    # Normalize the field data for heatmap
    field_normalized = np.abs(field_pattern) / np.max(np.abs(field_pattern))

    # Plot the heatmap using imshow
    plt.figure(figsize=(8, 6))
    plt.imshow(field_normalized, extent=[-1, 1, -1, 1], cmap='hot', aspect='auto')

    plt.colorbar(label='Near-field amplitude (normalized)')
    plt.title('Near-Field Radiation Pattern Heatmap')
    plt.xlabel('K_Y (1/m)')
    plt.ylabel('K_Z (1/m)')
    
    plt.show()

def plot_far_field(near_field, far_field_amplitude, angles, num_points):
    """
    Plot the far-field radiation pattern in 2D (E-plane).
    
    Parameters:
    - far_field_amplitude: The far-field amplitude data.
    - theta_far: The corresponding angles for the far-field pattern.
    """

    e_plane_magnitude = np.log(far_field_amplitude[:, num_points // 2])  # Cut at phi = 0
    #e_plane_magnitude_limited = e_plane_magnitude[valid_indices]  # Limit to valid angular range
    
    # H-plane: when theta = 0 (elevation is constant)
    h_plane_magnitude = np.log(far_field_amplitude[num_points // 2 , :])  # Cut at theta = 0
    #h_plane_magnitude_limited = h_plane_magnitude[valid_indices]  # Limit to valid angular range
    
    # Create the figure and the gridspec
    fig = plt.figure(figsize=(10, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], width_ratios=[1, 1])

    # E-plane polar plot (Top Left)
    ax1 = fig.add_subplot(grid[0, 0], projection='polar')
    ax1.plot(angles, e_plane_magnitude)
    ax1.set_title('E-Plane (Limited Angle Range)')

    # H-plane polar plot (Top Right)
    ax2 = fig.add_subplot(grid[0, 1], projection='polar')
    ax2.plot(angles, h_plane_magnitude)
    ax2.set_title('H-Plane (Limited Angle Range)')

    # Heatmap (Bottom, centered across both columns)
    ax3 = fig.add_subplot(grid[1, :])
    cax = ax3.imshow(far_field_amplitude, extent=[-1, 1, -1, 1], cmap='hot', aspect='auto')
    fig.colorbar(cax, ax=ax3, label='Far-field amplitude (normalized)')
    ax3.set_title('Far-Field Radiation Pattern Heatmap')
    ax3.set_xlabel('K_Y (1/m)')
    ax3.set_ylabel('K_Z (1/m)')


    near_field_amplitude = np.abs(near_field) / np.max(np.abs(near_field))
    # Fourth plot (Third Row, centered across both columns)
    ax4 = fig.add_subplot(grid[2, :])
    cax2 = ax4.imshow(near_field_amplitude, extent=[-1, 1, -1, 1], cmap='hot', aspect='auto')
    fig.colorbar(cax2, ax=ax4, label='Near-field amplitude (normalized)')
    ax4.set_title('Near-Field Radiation Pattern Heatmap')
    ax4.set_xlabel('K_Y (1/m)')
    ax4.set_ylabel('K_Z (1/m)')

    # Adjust layout for better spacing
    plt.tight_layout()

    plt.show()


def save_simulation_data(data, filename):
    """Save the simulation data to a .npy file."""
    np.save(filename, data)

def load_simulation_data(filename):
    """Load the simulation data from a .npy file."""
    return np.load(filename, allow_pickle=True)


# Constants
c = 3e8  # Speed of light in vacuum (m/s)
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
eps_0 = 1 / (mu_0 * c**2)  # Permittivity of free space (F/m)


# Example usage
antenna_size = 5   # Largest dimension of the antenna in meters

#wavelength = c / frequency  # Wavelength (meters)
wavelength = 0.03    # Wavelength in meters
plane_size = 0.6    # Size of the measurement plane in meters
num_points = 200   # Number of point along each axis
z_distance = 0.5   # Distance from the antenna to the measurement plane

# Example of running the function:
aperture_width = 0.1  # 10 cm
aperture_height = 0.1  # 10 cm
num_aperture_points = 200  # High-resolution aperture sampling

# Step 1: Simulate the near-field data
#near_field, Y, Z = simulate_near_field_dipole(antenna_size, wavelength, plane_size, z_distance, num_points)
#near_field = horn_near_field_precise(wavelength, aperture_width, aperture_height, z_distance, plane_size, plane_size, num_points, num_points, num_aperture_points)

# Filename to save/load simulation data
filename = "./near_field_simulation_data.npy"

# Check if the simulation data file exists
if os.path.exists(filename):
    user_input = input("Simulation data found. Do you want to use the saved data? (yes/no): ").strip().lower()
    if user_input in ['yes', 'y']:
        near_field = load_simulation_data(filename)
        print("Loaded saved simulation data.")
else:
    near_field = horn_near_field_precise(wavelength, aperture_width, aperture_height, z_distance, plane_size, plane_size, num_points, num_points, num_aperture_points)
    save_simulation_data(near_field, filename)
    print("New simulation completed and data saved.")


#plot_field_heatmap(near_field)
# Step 2: Perform the near-field to far-field transformation
far_field_pattern, theta_far, ky, kz = nf_ff_transform(near_field, wavelength, plane_size)

#print(f"Angle_min: {np.min(theta_far)} Angle_max: {np.max(theta_far)}")

# Step 3: Plot the far-field radiation pattern heatmap
#plot_far_field_heatmap(far_field_pattern, theta_far, ky, kz)
plot_far_field(near_field, far_field_pattern, theta_far, num_points)