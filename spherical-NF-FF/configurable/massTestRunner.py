from NF_FF_reconfigurabel import NF_FF_Transform_tester, Test_Descript, Test_Params
from modules.errors import *
from generateTables import generateFromTestDescriptors
from combine_FF_plots import generateCompareImageFromTestDescriptors
import math

# configure all test scenarios to testDescriptions = []
testDescriptions = []

# POSITION: normal distribution, uniform
def calcPercentPosition(distErrorM):
    return (distErrorM/0.03)/2 # 10GHz wavelength

test_params_position_normalUniform = [Test_Params(calcPercentPosition(0.001), '1mm'), 
                                      Test_Params(calcPercentPosition(0.005), '5mm'), 
                                      Test_Params(calcPercentPosition(0.01), '10mm'), 
                                      Test_Params(calcPercentPosition(0.02), '20mm'), 
                                      Test_Params(calcPercentPosition(0.03), '30mm'), 
                                      Test_Params(calcPercentPosition(0.05), '50mm')]
testDescriptions.append(Test_Descript('position_both_pol_same_error_normal', test_params_position_normalUniform, phase_same_errors_normal, reverseTableRowOrder=False))
testDescriptions.append(Test_Descript('position_both_pol_same_error_uniform', test_params_position_normalUniform, phase_same_errors_uniform, reverseTableRowOrder=False))


# POSITION: correlated theta, correlated phi
deviationFactor = 0.05
test_params_position_correlated = [Test_Params(calcPercentPosition(0.001), '1mm', deviationFactor), 
                                   Test_Params(calcPercentPosition(0.005), '5mm', deviationFactor), 
                                   Test_Params(calcPercentPosition(0.01), '10mm', deviationFactor), 
                                   Test_Params(calcPercentPosition(0.02), '20mm', deviationFactor), 
                                   Test_Params(calcPercentPosition(0.03), '30mm', deviationFactor), 
                                   Test_Params(calcPercentPosition(0.05), '50mm', deviationFactor)]
testDescriptions.append(Test_Descript('position_both_pol_same_error_correlated_phi', test_params_position_correlated, phase_errors_correlated_phi_same, reverseTableRowOrder=False))
testDescriptions.append(Test_Descript('position_both_pol_same_error_correlated_theta', test_params_position_correlated, phase_errors_correlated_theta_same, reverseTableRowOrder=False))


# NOISE: normaldistribution for same for both pol. and independen/only one
def calcPercentSNR(SNR):
    firstSidelobeNF = 1.3675165240369127
    n = (firstSidelobeNF / math.pow(10, (SNR/20)))
    return np.sqrt((n)) # 10GHz wavelength

test_params_noise = [Test_Params(calcPercentSNR(100), '100dB'), 
                     Test_Params(calcPercentSNR(90), '90dB'), 
                     Test_Params(calcPercentSNR(80), '80dB'), 
                     Test_Params(calcPercentSNR(70), '70dB'), 
                     Test_Params(calcPercentSNR(60), '60dB'), 
                     Test_Params(calcPercentSNR(50), '50dB'), 
                     Test_Params(calcPercentSNR(40), '40dB'), 
                     Test_Params(calcPercentSNR(30), '30dB'), 
                     Test_Params(calcPercentSNR(20), '20dB'), 
                     Test_Params(calcPercentSNR(10), '10dB'), 
                     Test_Params(calcPercentSNR(0), '0dB')]
testDescriptions.append(Test_Descript('noise/amplitude_same_errors_normal', test_params_noise, amplitude_same_errors_normal_noise, reverseTableRowOrder=True, titleSuffix='noise', legendType='SNR'))
testDescriptions.append(Test_Descript('noise/amplitude_independent_errors_normal', test_params_noise, amplitude_independent_errors_normal_noise, reverseTableRowOrder=True, titleSuffix='noise', legendType='SNR'))
testDescriptions.append(Test_Descript('noise/amplitude_single_pol_errors_normal', test_params_noise, amplitude_single_pol_errors_normal_noise, reverseTableRowOrder=True, titleSuffix='noise', legendType='SNR'))


# GIMBAL: 
test_params_gimbal = [Test_Params(0.02, '2E-2'), 
                      Test_Params(0.1, '1E-1'), 
                      Test_Params(0.5, '5E-1'), 
                      Test_Params(1, '1'), 
                      Test_Params(5, '5'), 
                      Test_Params(10, '10')]
testDescriptions.append(Test_Descript('gimbal_errors_uniform', test_params_gimbal, gimbal_error_uniform, reverseTableRowOrder=False))


root_path = './spherical-NF-FF/testResults_2025_01_17'
comparepath = f'./spherical-NF-FF/testResults/FF_data_no_error.txt'
NF_FF_Transform_tester().runTesets(root_path, testDescriptions, showProgress=True)
generateFromTestDescriptors(root_path, testDescriptions, showProgress=True)
generateCompareImageFromTestDescriptors(root_path, testDescriptions, phi_select_angle=0, compareToPath=comparepath, showProgress=True)
