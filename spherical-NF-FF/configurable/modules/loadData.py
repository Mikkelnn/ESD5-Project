import numpy as np
import pandas as pd
import re

def load_header(file_path, num_lines=13):
    with open(file_path, 'r') as file:
        header_lines = [file.readline().strip() for _ in range(num_lines)]
    return '\n'.join(header_lines)

def parse_csv_header(header_text):
    # Define regex patterns for each parameter
    theta_pattern = r"Step axis\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    phi_pattern = r"Scan axis\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    
    # Extract values for Theta (Step axis)
    theta_match = re.search(theta_pattern, header_text)
    if theta_match:
        theta_start, theta_end, theta_step_size = map(float, theta_match.groups())
        theta_values = np.arange(theta_start, theta_end + theta_step_size, theta_step_size)  # Array of theta values
    else:
        raise ValueError("Theta (Step axis) information not found in header.")

    # Extract values for Phi (Scan axis)
    phi_match = re.search(phi_pattern, header_text)
    if phi_match:
        phi_start, phi_end, phi_step_size = map(float, phi_match.groups())
        phi_values = np.arange(phi_start, phi_end + phi_step_size, phi_step_size)  # Array of phi values
    else:
        raise ValueError("Phi (Scan axis) information not found in header.")
    
    # Return the parsed data as a dictionary
    return {
        "theta_start": theta_start,
        "theta_end": theta_end,
        "theta_stepSize": theta_step_size,
        "theta_values": theta_values,
        "phi_start": phi_start,
        "phi_end": phi_end,
        "phi_stepSize": phi_step_size,
        "phi_values": phi_values
    }

# Function to load the data and return a three-dimensional array of complex numbers
def load_data_lab_measurements(file_path):
    # Load the data, skipping the non-data rows
    header_text = load_header(file_path, num_lines=13)
    nf_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=13, header=None)
    
    # determine the theta, phi angles and step sizes from the header data
    headerData = parse_csv_header(header_text)

    # Calculate theta and phi sizes
    theta_size = len(headerData["theta_values"]) #int(np.sqrt(nf_data.shape[0] / (4 * 2.5)))
    phi_size = len(headerData["phi_values"]) #int(theta_size * 2 * 2.5)
   
    # Initialize the total electric fields in the near field as a single array
    complex_field_data = np.zeros((theta_size, phi_size, 2), dtype=complex)

    # Fill in the electric field arrays with complex numbers
    k = 0
    for i in range(theta_size):
        for j in range(phi_size):
            # E_theta component
            complex_field_data[i, j, 0] += nf_data.iloc[k, 3] + 1j * nf_data.iloc[k, 4]
            # E_phi component
            complex_field_data[i, j, 1] += nf_data.iloc[k + int(nf_data.shape[0] / 2), 3] + 1j * nf_data.iloc[k + int(nf_data.shape[0] / 2), 4]
            k += 1

    return (complex_field_data, headerData["theta_values"], headerData["phi_values"], headerData["theta_stepSize"], headerData["phi_stepSize"])


def load_data_cst(file_path):
    # Load the data, skipping the row with dashes (assumed to be the second row)
    nf_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, header=None)

    # Calculate theta and phi sizes
    theta_values = sorted(list(set(nf_data.iloc[:, 0]))) # distinct (deduplicate) theta values
    theta_size = len(theta_values)
    theta_stepSize = (np.max(theta_values) - np.min(theta_values)) / (theta_size - 1)

    phi_values = sorted(list(set(nf_data.iloc[:, 1]))) # distinct (deduplicate) phi values
    phi_size = len(phi_values)
    phi_stepSize = (np.max(phi_values) - np.min(phi_values)) / (phi_size - 1)

    # Initialize the total electric fields in the near field
    complex_field_data = np.zeros((theta_size, phi_size, 2), dtype=np.complex_)

    # Fill in the electric field arrays with complex numbers
    k = 0
    for j in range(phi_size):
        for i in range(theta_size):
            # Compute E_theta and E_phi as complex numbers
            e_theta_magnitude = nf_data.iloc[k, 3]
            e_theta_phase = np.deg2rad(nf_data.iloc[k, 4]) # Convert degrees to radians
            e_phi_magnitude = nf_data.iloc[k, 5]
            e_phi_phase = np.deg2rad(nf_data.iloc[k, 6]) # Convert degrees to radians
            
            # Construct the complex electric field components
            complex_field_data[i, j, 0] = e_theta_magnitude + 1j * np.sin(e_theta_phase)
            complex_field_data[i, j, 1] = e_phi_magnitude + 1j * np.sin(e_phi_phase)
            k += 1

    return (complex_field_data, theta_values, phi_values, theta_stepSize, phi_stepSize)


def load_FF_data_own_output(file_path):
    # Load the data, skipping the row with dashes (assumed to be the second row)
    nf_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)

    # Calculate theta and phi sizes
    theta_values = np.array(sorted(list(set(nf_data.iloc[:, 0])))) # distinct (deduplicate) theta values
    theta_size = len(theta_values)
    theta_stepSize = (np.max(theta_values) - np.min(theta_values)) / (theta_size - 1)

    phi_values = np.array(sorted(list(set(nf_data.iloc[:, 1])))) # distinct (deduplicate) phi values
    phi_size = len(phi_values)
    phi_stepSize = (np.max(phi_values) - np.min(phi_values)) / (phi_size - 1)

    # Initialize the total electric fields in the near field
    ff_data = np.zeros((theta_size, phi_size), dtype=float)

    # Fill in the electric field arrays with complex numbers
    k = 0
    for i in range(theta_size):
        for j in range(phi_size):
            ff_data[i, j] = nf_data.iloc[k, 2]
            k += 1

    return (ff_data, theta_values, phi_values, theta_stepSize, phi_stepSize)