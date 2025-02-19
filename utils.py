import math
import os
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import keras

NN_WORKING_FREQUENCY = 4000  # 4000 Hz, frequency with which control action is called. Previously tested with 20 kHz
NN_WORKING_PERIOD = 1 / NN_WORKING_FREQUENCY  # seconds, period of calling control action

# old feature columns
#feature_columns = ['mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
#                       'desired_velocity']
feature_columns = ['mod_rack_position', 'rack_velocity', 'tahn_rack_acceleration',
                       'desired_velocity']
feature_columns_extended = ['mod_rack_position', 'rack_velocity', 'tahn_rack_acceleration',
                       'desired_velocity', 'error_integral', 'error_derivative']
#features_cnt = len(feature_columns)
features_cnt = len(feature_columns_extended)
sequence_length = 25


# msm constants
tooth_alpha = math.radians(20)

msm_length = 10e-3  # meters
msm_width = 4e-3  # meters
msm_height = 1e-3  # meters
msm_A0 = msm_width * msm_height  # MSM crystal cross section area


constant_magnetic_stress = 3.27e+6  # Pa, from Saren and Ullakko 2017, ref [24] there
k_0 = 11.8  #  from Saren and Ullakko 2017
cos_a = math.cos(math.radians(43.23))
e_0 = 0.06  # max elongation, from Saren and Ullakko, approximate value
allowed_elongation = 0.05  # full extension is not allowed to keep the TB and increase cyclic life
one_side_elongation_capacity = ( e_0 - allowed_elongation ) / 2  # Used as an offset and shows the amount of possible additional elongation/contraction
one_side_absolute_offset = one_side_elongation_capacity * msm_length
e_initial = 0.05  # initial contraction
contraction_initial = -msm_length * e_initial  # meters, value wrt to the full elongation, same as initial position of the Twin Boundary


simulation_timestep = 2e-6  # value from matlab 5e-6  Tested timestep 3e-6
msm_elements_cnt = 4

# masses
ro = 8000  # kg/m^3, MSM alloy density
msm_mass = ro * msm_A0 * msm_length  # total mass of msm element
tooth_plate_mass = 0.22e-3  # kg
rack_mass = 0.002  # kg
useful_mass = 0.050  # kg
total_rack_mass = rack_mass + useful_mass


ro_k_0_cos_coeff = ro / (k_0 * cos_a)

# friction
rack_static_friction = 0.5  # Newtons
tooth_plate_static_friction = 0.1  # Newtons

CONTROL_VALUE_BLOCKING_LIMIT = 0.4  # unactuated teeth will engage the rack if control value will be lower that this one
MAX_BLOCKING_OUTPUT_VALUE = 0.5  # maximum control value applied to the blocking teeth


# metrics
reward_list = np.array([])

def get_pid_data(sequence_length, num_files, print_time=False, folder= 'processed_data'):
    """

    :param sequence_length:
    :param num_files: int, if -1 is give then returns all the available samples
    :return:
    """
    t0 = time.time()
    # available columns:
    # ['mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
    #  'desired_velocity', 'forward_pid', 'reverse_pid', 'new_pid', 'initial_pid',
    #  'controller_switch_spike']
    target_columns = ['new_pid']
    x = None
    y = None
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    file_num = min(num_files, len(files))
    if num_files < 0:
        file_num = len(files)
    for i in range(file_num):
        cur_file = os.path.join(folder, files[i])
        df = pd.read_csv(cur_file, usecols=feature_columns)
        tmp_x = df.to_numpy()
        tmp_x.reshape((1, tmp_x.shape[0], tmp_x.shape[1]))
        df = pd.read_csv(cur_file, usecols=target_columns)
        tmp_y = df.to_numpy()
        matrix_length = tmp_x.shape[0]
        num_rows_to_keep = (matrix_length // sequence_length) * sequence_length  # Calculate the number of rows to keep
        if num_rows_to_keep < 1:
            raise Exception('sequence_length is too long!')

        if matrix_length % sequence_length != 0:  # Exclude the remainder rows if it is not divisible by sequence length
            tmp_x = tmp_x[:num_rows_to_keep, :]
            tmp_y = tmp_y[:num_rows_to_keep, :]
        tmp_x = tmp_x.reshape(-1, sequence_length, features_cnt)  # Reshape the matrix to the shape (a//c, c, b)
        tmp_y = tmp_y.reshape(-1, sequence_length, 1)

        if x is None:
            x = tmp_x
            y = tmp_y
        else:
            x = np.vstack((x, tmp_x))
            y = np.vstack((y, tmp_y))
    if print_time:
        print('data loading finished in ', time.time()-t0, ' seconds')
    return x, y

def normalize_data(data_matrix, scaler=None, load_scaler=True, save_scaler=True):
    """
    rescaling data matrix of size (A, B, C) to range [0, 1] across C axis

    :param data_matrix: input 3d data to be normalized. Use None data_matrix to load an existing scaler
    :param scaler: if scaler exists, then uses the scaler to normalize the data, otherwise creates a new scale based on the input data
    :param load_scaler: if True loads existing scaler from file, even if a scaler in the argument is provided
    :param save_scaler: if True saves the newly created scaler into a file. Saving is not done if load_scaler is True
    :return: new normalized data, scaler
    """

    if load_scaler:  # Loading existing scaler from file
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        if data_matrix is None:
            return None, scaler

    A, B, C = data_matrix.shape[0], data_matrix.shape[1], data_matrix.shape[2]

    # Normalize the array along the C dimension to the range [0, 1]
    # Reshape array to (A*B, C)
    reshaped_data_matrix = data_matrix.reshape(-1, C)


    if scaler is None:  # Perform min-max normalization along the C dimension using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data_matrix = scaler.fit_transform(reshaped_data_matrix)
    # else:
    normalized_data_matrix = scaler.transform(reshaped_data_matrix)
    # Reshape the array back to (A, B, C)
    normalized_data_matrix = normalized_data_matrix.reshape(A, B, C)

    if save_scaler and not load_scaler:  # saving the newly created scaler. No saving is made if the scaler was loaded
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)


    return normalized_data_matrix, scaler

def get_nn_model(folder='pid_neural_networks', controller_name="test_controller_tf_2_10"):  # controller_name="test_controller"
    model = keras.models.load_model(os.path.join(folder, controller_name))
    return model

def get_matlab_model_path():
    os_name = os.name
    if os_name == 'posix':  # unix-like system
        model_path = os.path.join('~', 'ik_files', 'msm_matlab_model')
    elif os_name == 'nt':  # windows
        model_path = 'C:\\Users\\h19343\\files\\PhD\\MSM actuators simualtion\\Simscape model'

    return model_path

