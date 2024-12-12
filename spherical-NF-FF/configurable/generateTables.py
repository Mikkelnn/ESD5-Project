import re
import glob

# Define a function to parse the file and extract relevant metrics
def parse_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # Extract parameters
    test_params_match = re.search(r'TEST_PARAMS: (\S+) ([\d\.]+)(?=; DeviationFactor: ([\d\.]+))?', data)
    if not test_params_match:
        raise ValueError("TEST_PARAMS not found")
    param_name = test_params_match.group(1)
    param_percent = float(test_params_match.group(2)) * 100
    deviation_factor = float(test_params_match.groups(0)[2])

    # Extract errors
    max_error_e_plane = float(re.search(r'Max error e-plane: ([\d.]+)', data).group(1))
    max_error_h_plane = float(re.search(r'Max error h-plane: ([\d.]+)', data).group(1))
    mean_error_e_plane = float(re.search(r'Mean error e-plane: ([\d.]+)', data).group(1))
    mean_error_h_plane = float(re.search(r'Mean error h-plane: ([\d.]+)', data).group(1))

    # Extract absolute errors
    max_absolute_error = float(re.search(r'Max absolute error \(all data\): ([\d.]+)', data).group(1))
    mean_absolute_error = float(re.search(r'Mean error \(all data\): ([\d.]+)', data).group(1))

    # Extract HPBW values after the relevant section
    errors_section = re.search(r'NF transformed data \(FF\) with errors:(.*?)(?:\n\n|$)', data, re.S)
    if not errors_section:
        raise ValueError("HPBW data for errors not found")
    errors_data = errors_section.group(1)

    h_smooth = float(re.search(r'H-plane \(smoothed\) HPBW: ([\d.]+)', errors_data).group(1))
    h_orig = float(re.search(r'H-plane \(original\) HPBW: ([\d.]+)', errors_data).group(1))
    e_smooth = float(re.search(r'E-plane \(smoothed\) HPBW: ([\d.]+)', errors_data).group(1))
    e_orig = float(re.search(r'E-plane \(original\) HPBW: ([\d.]+)', errors_data).group(1))

    return {
        'param_name': param_name,
        'param_percent': param_percent,
        'deviation_factor': deviation_factor,
        'max_error_e_plane': max_error_e_plane,
        'max_error_h_plane': max_error_h_plane,
        'mean_error_e_plane': mean_error_e_plane,
        'mean_error_h_plane': mean_error_h_plane,
        'max_absolute_error': max_absolute_error,
        'mean_absolute_error': mean_absolute_error,
        'h_smooth': h_smooth,
        'h_orig': h_orig,
        'e_smooth': e_smooth,
        'e_orig': e_orig
    }

# Generate LaTeX row for the extracted data
def generate_latex_row(data):
    return (f"{data['param_name']} {data['param_percent']:.1f}\\%   & "
            f"{data['max_error_e_plane']:.2f}  & {data['mean_error_e_plane']:.2f} & "
            f"{data['max_error_h_plane']:.2f}  & {data['mean_error_h_plane']:.2f} & "
            f"{data['max_absolute_error']:.2f}  & {data['mean_absolute_error']:.2f} & "
            f"{data['e_orig']:.2f} & {data['h_orig']:.2f} \\\\")


TEST_NAME = 'position_both_pol_same_error_correlated_theta' # used to determine folder to output files
PATH_PREFIX = f'./spherical-NF-FF/testResults/{TEST_NAME}/'
FILE_PATH_SEARCH = f'{PATH_PREFIX}*/metrics.txt'

# Extract the numeric part of the folder name and sort paths
def extract_numeric_key(path):
    match = re.search(r'/(\d+)mm/', path)  # Find a number followed by "mm" in the path
    return int(match.group(1)) if match else float('inf')  # Default to 'inf' if no match is found

# Find and sort all matching file paths
matching_files = sorted(glob.glob(FILE_PATH_SEARCH), key=extract_numeric_key)

# matching_files = sort(matching_files)
for file_path in matching_files:
    parsed_data = parse_file(file_path)
    latex_row = generate_latex_row(parsed_data)
    print(latex_row)
