import pandas as pd
import numpy as np
import os
import utils

# this value is defined in utils
# NN_WORKING_FREQUENCY = 20000  # Hz, frequency with which neural network will be called

def parse_and_save_excel(full_path, sample_frequency):
    """
    Parsing the data from the prepared excel files
    :param full_path:
    :return:
    """
    # Read the Excel file
    excel_data = pd.ExcelFile(full_path)
    sheet_names = ['time', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration', 'forward_pid', 'reverse_pid',
                   'new_pid', 'initial_pid', 'controller_switch_spike', 'desired_velocity']
    saturated_data_names = ['forward_pid', 'reverse_pid', 'new_pid', 'initial_pid']  # values of these fields will be limited to [-1, 1] range
    # Dictionary to store numpy arrays from each sheet
    sheet_arrays = {}
    for sheet_name in sheet_names:  # Iterate through sheets in the Excel file
        sheet_df = pd.read_excel(full_path, sheet_name=sheet_name, header=None)
        tmp_array = sheet_df.iloc[:, 0].to_numpy()
        if sheet_name in saturated_data_names:
            tmp_array = np.clip(tmp_array, -1, 1)  # limiting values to [-1, 1] range
        sheet_arrays[sheet_name] = tmp_array


    sheet_df = pd.read_excel(full_path, sheet_name='pitch_type', header=None)
    pitch_type = sheet_df.iloc[0, 0]
    if pitch_type == 'force_optimal':
        tooth_pitch = 3.626908295808783e-04
    else:
        raise Exception("Pitch type from excel data is not supported yet.")

    sheet_arrays['mod_rack_position'] = np.mod(sheet_arrays['rack_position'], tooth_pitch)
    data_saving_order = ('mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
                         'desired_velocity', 'forward_pid', 'reverse_pid', 'new_pid', 'initial_pid',
                         'controller_switch_spike')
    data_len = sheet_arrays["time"].shape[0]
    features_cnt = len(data_saving_order)
    combined_mat = np.zeros((1, features_cnt))

    cur_t = sheet_arrays["time"][0]
    full_time_step = 0
    full_step_idx = 0
    for i in range(data_len):  # generating matrix with desired frequency
        if sheet_arrays["time"][i] - cur_t > 1 / sample_frequency or i == 0:
            if i != 0 and cur_t == sheet_arrays["time"][0]:
                full_time_step = sheet_arrays["time"][i] - sheet_arrays["time"][0]
                full_step_idx = i
            cur_t = sheet_arrays["time"][i]
            tmp_array = np.zeros((1, features_cnt))
            for j in range(features_cnt):
                tmp_array[0][j] = sheet_arrays[data_saving_order[j]][i]
            combined_mat = np.concatenate((combined_mat, tmp_array))
    combined_mat = combined_mat[1:, :]  # because first raw is zeros
    df = pd.DataFrame(combined_mat, columns=data_saving_order)
    folder_mame = 'processed_data_frequency_' + str(sample_frequency) + "_Hz"  # TODO if folder does not exist, create one
    csv_file = os.path.join(folder_mame, 'series_0_' + excel_file[:-5] + '.csv')
    df.to_csv(csv_file, index=False)

    # in case the data frequency is much higher that the NN frequency,
    # it is possible to offset the timestep by half a period and generate a second matrix of data
    #if full_step_idx > 10:
    split_cnt = full_step_idx//40  # number of extra datasets obtained from the same record
    print('split count = ', split_cnt)
    for k in range(1, split_cnt):
        combined_mat = np.zeros((1, features_cnt))
        cur_t = full_time_step * k / split_cnt
        for i in range(data_len):  # generating matrix with desired frequency
            if sheet_arrays["time"][i] - cur_t > 1 / sample_frequency:
                cur_t = sheet_arrays["time"][i]
                tmp_array = np.zeros((1, features_cnt))
                for j in range(features_cnt):
                    tmp_array[0][j] = sheet_arrays[data_saving_order[j]][i]
                combined_mat = np.concatenate((combined_mat, tmp_array))
        combined_mat = combined_mat[1:, :]  # because first raw is zeros
        df = pd.DataFrame(combined_mat, columns=data_saving_order)
        csv_file = os.path.join(folder_mame, 'series_' + str(k) + '_' + excel_file[:-5] + '.csv')
        df.to_csv(csv_file, index=False)

folder = 'C:\\Users\\h19343\\files\\PhD\\MSM actuators simualtion\\Simscape model\\SimResults\\pid_data\\excel_force_optimal_pitch_plates_cnt_4'
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
for i in range(len(files)):
    # excel_file = 'load_2_5513_vel_0_01934_freq_0_001_amp_0_00987.xlsx'
    excel_file = files[i]
    full_path = os.path.join(folder, excel_file)
    parse_and_save_excel(full_path, sample_frequency=utils.NN_WORKING_FREQUENCY)
    if i % 5 == 0:
        print(i, ' out of ', len(files), ' files have been parsed')



