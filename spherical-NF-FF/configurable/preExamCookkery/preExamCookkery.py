from modules.loadData import *
from modules.simulate_NF_spherical import *
from modules.errors import *
from modules.transform_NF_FF import *
from modules.pre_process import *
from modules.post_process import *
from modules.output import *
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Test_Params:
    def __init__(self, value, name, deviationFactor = None):
        self.value = value
        self.name = name
        self.deviationFactor = deviationFactor
    
    def getTestParamsTxt(self):
        temp = f'{self.name} {self.value}'
        if self.deviationFactor is not None:
            temp += f'; DeviationFactor: {self.deviationFactor}'

        return temp + '\n'
    
    def getParams(self):
        if self.deviationFactor is not None:
            return [self.deviationFactor, self.value]

        return [self.value]

class Test_Descript:
    def __init__(self, testName, testParams, errorApplyMethod, reverseTableRowOrder, titleSuffix='error', legendType='Error'):
        self.testName = testName # also used for path to save data
        self.testParams = testParams # list of "Test_Params" to run for a given "errorApplyMethod"
        self.errorApplyMethod = errorApplyMethod
        self.reverseTableRowOrder = reverseTableRowOrder
        self.titleSuffix = titleSuffix 
        self.legendType = legendType


class NF_FF_Transform_tester:
    # The idea of this script is to make it easy to configure different scenarios
    # The heavy lifting will be done in multiple sub-files

    # Examlpe configurations:
    # read_NF_lab_data -> transform -> plot
    # read_NF_lab_data -> introduce_error -> transform -> plot
    # read_NF_lab_data -> transform -> smooth -> plot
    # read_NF_lab_data -> introduce_error -> transform -> smooth -> plot

    ##############################################################################################################
    # 1. Where is the NF data from? (simulated or from a file)
    ##############################################################################################################
    def loadNFData(self):
        self.frequency_Hz = 10e9 # 10GHz
        self.transform_from_dist_meters = 0.3 # The distance you want the transform from.
        self.transform_to_dist_meters = 10e6 # the distance you want the transform to!

        # data from CST simulation:
        # file_path = './simoulation CST/Parabolic/0.5sampleNearfield_2.98mParabolic.txt'
        # nfData_1, theta_deg, phi_deg, theta_step_deg, phi_step_deg = load_data_cst(file_path)

        #file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3mParabolic.txt'
        #nfData_2, _, _, _, _ = load_data_cst(file_path)

        #file_path = './simoulation CST/Parabolic/0.5sampleNearfield_3.02mParabolic.txt'
        #nfData_3, _, _, _, _ = load_data_cst(file_path)

        #nfData = combine_data_for_position_error([nfData_1, nfData_2, nfData_3])

        # rezise theta and phi axis
        # new_shape = (int(nfData_1.shape[0] / 2), int(nfData_1.shape[1] / 2))
        # nfData, theta_deg, phi_deg, theta_step_deg, phi_step_deg = get_theta_phi_error_from_fine_set(nfData_1, new_shape, sample_theta=True, sample_phi=True)

        # data from lab-measurements:
        #file_path = './NF-FF-data/SH800_CBC_006000.CSV' # use relative path! i.e. universal :)
        file_path = './NF-FF-Data-2/16240-20CBCFF_dir_30_010000.CSV'
        self.nfData, self.theta_deg, self.phi_deg, theta_step_deg, phi_step_deg = load_data_lab_measurements(file_path)

        # file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
        # file_path2 = './NF-FF-Data-2/Flann16240-20_CBC_FF_dir_010000.CSV'
        # ffData_loaded, theta_deg_loaded, phi_deg_loaded, _, _ = load_data_lab_measurements(file_path2)

        # simulate data
        #nfData = simulate_NF_dipole_array()

        # zero-pad before converting theta, phi values
        self.nfData, theta_deg2, self.num_zero_nfData = pad_theta(self.nfData, theta_step_deg)
        # ffData_loaded, theta_deg2, num_zero_ffData = pad_theta(ffData_loaded, theta_step_deg)

        # Define theta and phi ranges for far-field plotting
        # get zero in center
        self.phi_deg_center = np.floor(self.phi_deg - (np.max(self.phi_deg) / 2))
        self.theta_deg_center = np.linspace(-np.max(self.theta_deg), np.max(self.theta_deg), (len(self.theta_deg)*2)-1)


    ##############################################################################################################
    # 2. Introduction of errors in the NF, comment out if no errors should be present
    ##############################################################################################################
    def applyError(self, errorMethod, configParams):
        self.nfDataError = np.copy(self.nfData)
        appliedError = errorMethod(self.nfDataError, *configParams)
        self.appliedError = removeXFromEnd(appliedError, int(self.num_zero_nfData))

    ##############################################################################################################
    # 3. Transform data - most likely static...
    ##############################################################################################################
    def transFormData(self):
        # pre-process nfData
        nfData_sum = HansenPreProcessing(self.nfData)
        nfData_sum_error = HansenPreProcessing(self.nfDataError)

        # transform NF to FF
        ffData_no_error = spherical_far_field_transform_SNIFT(nfData_sum, self.frequency_Hz, self.transform_from_dist_meters, self.transform_to_dist_meters)
        ffData_error = spherical_far_field_transform_SNIFT(nfData_sum_error, self.frequency_Hz, self.transform_from_dist_meters, self.transform_to_dist_meters)

        # post-process FF
        ffData_no_error_2D = sum_NF_poles_sqrt(ffData_no_error)
        ffData_error_2D = sum_NF_poles_sqrt(ffData_error)
        # ffData_loaded = sum_NF_poles_sqrt(ffData_loaded)

        # Normalize plots
        # ffData_loaded = ffData_loaded / np.max(np.abs(ffData_loaded))
        norm_factor = np.abs(ffData_no_error_2D)
        ffData_no_error_2D = ffData_no_error_2D / np.max(norm_factor)
        ffData_error_2D = ffData_error_2D / np.max(norm_factor)

        # Remove original zero padding
        # ffData_loaded2 = removeXFromEnd(ffData_loaded, int(num_zero_nfData))
        self.ffData_no_error_2D = removeXFromEnd(ffData_no_error_2D, int(self.num_zero_nfData))
        self.ffData_error_2D = removeXFromEnd(ffData_error_2D, int(self.num_zero_nfData))

    ##############################################################################################################
    # 4. Select far field at angle and smooth data
    ##############################################################################################################
    def selectData(self):
        self.phi_select_angle = 0 # the angle of witch to represent h-plane plot in degrees

        self.ffData_no_error_2D = 20 * np.log10(self.ffData_no_error_2D)
        self.ffData_error_2D = 20 * np.log10(self.ffData_error_2D)

        # dataLoaded = select_data_at_angle(ffData_loaded2, phi_deg_loaded, phi_select_angle)

        self.selected_ffData_no_error = select_data_at_angle(self.ffData_no_error_2D, self.phi_deg, self.phi_select_angle)
        self.selected_ffData_error = select_data_at_angle(self.ffData_error_2D, self.phi_deg, self.phi_select_angle)
        #dataDif = select_data_at_angle(farfieldDataDiff, phi_deg, phi_select_angle)

    ##############################################################################################################
    # 5. Output FF - plot or write to file
    ##############################################################################################################
    def outputResults(self, PATH_PREFIX, testParamsTxt):
        

        diff = np.abs(self.selected_ffData_no_error.e_plane_data_original - self.selected_ffData_error.e_plane_data_original)
        ffirstSidelobeDiff = diff[19]

        ### save metrics data in txt (HPBW, mean, max)
        metricsTxt = f'TEST_PARAMS: {testParamsTxt}\n'

        #HPBW
        metricsTxt += f'NF transformed data (FF) no errors:\n{calculate_print_hpbw(self.selected_ffData_no_error, self.theta_deg_center)}\n'
        metricsTxt += f'\nNF transformed data (FF) with errors:\n{calculate_print_hpbw(self.selected_ffData_error, self.theta_deg_center)}\n'

        # mean, max (selected data)
        metricsTxt += '\nMean and Max errors between selected data with errors and selected data no errors\n'
        metricsTxt += f"Max error e-plane: {calculate_max_indexed_error(self.selected_ffData_no_error.e_plane_data_original, self.selected_ffData_error.e_plane_data_original)}\n" 
        metricsTxt += f"Max error h-plane: {calculate_max_indexed_error(self.selected_ffData_no_error.h_plane_data_original, self.selected_ffData_error.h_plane_data_original)}\n"
        metricsTxt += f"Mean error e-plane: {calculate_mean_indexed_error(self.selected_ffData_no_error.e_plane_data_original, self.selected_ffData_error.e_plane_data_original)}\n"
        metricsTxt += f"Mean error h-plane: {calculate_mean_indexed_error(self.selected_ffData_no_error.h_plane_data_original, self.selected_ffData_error.h_plane_data_original)}\n"

        # mean, max (all data)
        metricsTxt += '\nMean and Max errors between data with errors and data no errors (2D array)\n'
        metricsTxt += f"Max absolute error (all data): {calculate_max_indexed_error(self.ffData_no_error_2D, self.ffData_error_2D)}\n" 
        metricsTxt += f"Mean error (all data): {calculate_mean_indexed_error(self.ffData_no_error_2D, self.ffData_error_2D)}\n"

        # extra data
        metricsTxt += f"First sidelobe error: {ffirstSidelobeDiff}\n"

        write_file(metricsTxt, PATH_PREFIX + 'metrics.txt')

        # show all figures
        # show_figures()

    def runTesets(self, root_path, testDescriptions, showProgress=True):
        self.loadNFData()

        for testDescription in tqdm(testDescriptions, disable=(not showProgress)):            
            for params in tqdm(testDescription.testParams, disable=(not showProgress), leave=False):
                PATH_PREFIX = f'{root_path}/{testDescription.testName}/{params.name}/'
                # print(f'STARTED: {PATH_PREFIX}')
                # ensure folder exist
                Path(PATH_PREFIX).mkdir(parents=True, exist_ok=True)

                self.applyError(testDescription.errorApplyMethod, params.getParams())
                self.transFormData()
                self.selectData()
                self.outputResults(PATH_PREFIX, params.getTestParamsTxt())
