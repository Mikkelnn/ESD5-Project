import re
import glob
import numpy as np
from tqdm import tqdm
from modules.output import write_file 
import pandas as pd
from pathlib import Path

# Define a function to parse the file and extract relevant metrics
def parse_file(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # Extract parameters
    test_params_match = re.search(r'TEST_PARAMS: (\S+) ([\d\.]+)(?=; DeviationFactor: ([\d\.]+))?', data)
    if not test_params_match:
        raise ValueError("TEST_PARAMS not found")
    param_name = test_params_match.group(1)
    param_percent = float(test_params_match.group(2))
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

    firstSidelobeError = float(re.search(r'First sidelobe error: ([\d.]+)', data).group(1))

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
        'e_orig': e_orig,
        'first_sidelobe_error': firstSidelobeError
    }

# Generate LaTeX row for the extracted data
def generate_latex_row(data):
    return (f"{data["param_name"]} & "
            f"{data["std_mean_maxerror_e"]:.2f}  & {data["std_mean_meanerror_e"]:.2f} & "
            f"{data["std_mean_maxerror_h"]:.2f}  & {data["std_mean_meanerror_h"]:.2f} & "
            f"{data["std_max_absError"]:.2f}  & {data["std_mean_absError"]:.2f} & "
            f"{data["std_mean_firstSidelobeError"]:.2f} & {data["std_max_firstSidelobeError"]:.2f} \\\\")


# Extract the numeric part of the folder name and sort paths
def extract_numeric_key(path):
    match = re.search(r'[\\\/]+([\dE+-]+)(mm|dB)?[\\\/]', path)  # Find a number followed by "mm" in the path
    return float(match.group(1)) if match else float('inf')  # Default to 'inf' if no match is found

def generateSaveTable(filePath, reverseRowOrder=False):
    FILE_PATH_SEARCH = f'{filePath}/*/metrics.txt'
    # Find and sort all matching file paths
    matching_files = sorted(glob.glob(FILE_PATH_SEARCH), key=extract_numeric_key, reverse=reverseRowOrder)
    # matching_files = sort(matching_files)
    rows = ''
    parsed_data = []
    for file_path in matching_files:
        parsed_data.append(parse_file(file_path))
    
    df = pd.DataFrame(parsed_data)

    grouped_parsed_data = df.groupby('param_name').agg(
        std_mean_maxerror_e=('max_error_e_plane','std'),
        std_mean_maxerror_h=('max_error_h_plane','std'),
        std_mean_meanerror_e=('mean_error_e_plane','std'),
        std_mean_meanerror_h=('mean_error_h_plane','std'),
        std_max_absError=('max_absolute_error','std'),
        std_mean_absError=('mean_absolute_error','std'),
        std_mean_firstSidelobeError=('first_sidelobe_error','std'),
        std_max_firstSidelobeError=('first_sidelobe_error','std'),
    ).reset_index()

    grouped_parsed_data['mm_value'] = grouped_parsed_data['param_name'].str.extract(r'(\d+)').astype(int)
    sorted_data = grouped_parsed_data.sort_values(by='mm_value',ascending=(not reverseRowOrder)).drop(columns=['mm_value'])

    for index, row in sorted_data.iterrows():  
        print(row)  
        latex_row = generate_latex_row(row)
        rows += f'{latex_row}\n'
    
    filePathNew = filePath.replace("*","all/std")
    Path(filePathNew).mkdir(parents=True, exist_ok=True)
    write_file(rows, f'{filePathNew}/summaryTable.txt')

def generateFromTestDescriptors(rootPath, descriptors, showProgress):
    for descriptor in tqdm(descriptors, disable=(not showProgress)):
        generateSaveTable(f'{rootPath}/*/{descriptor.testName}', descriptor.reverseTableRowOrder)
