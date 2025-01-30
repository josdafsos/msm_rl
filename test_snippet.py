# import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.model import Sequential have to call in full from code, wtf
import tensorflow as tf
from tensorflow.keras.models import Sequential # no error here actually, wtf
from numpy import loadtxt
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os



def generate_multivar_data(sequence_length, num_sequences):
    a = np.array([np.sin(np.linspace(0, 4 * np.pi, sequence_length) + i) for i in range(num_sequences)])
    b = np.array([np.cos(np.linspace(0, 4 * np.pi, sequence_length) - i) + i / sequence_length \
                  for i in range(num_sequences)])
    y = a * b
    x = np.concatenate((a.reshape((a.shape[0], a.shape[1], 1)),
                                     b.reshape((b.shape[0], b.shape[1], 1))),
                                    axis=-1)
    y = y.reshape((y.shape[0], y.shape[1], 1))
    return x, y

def get_pid_data(sequence_length, num_sequences):
    """

    :param sequence_length:
    :param num_sequences: int, if -1 is give then returns all the available samples
    :return:
    """
    folder = 'processed_data'
    # available columns:
    # ['mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
    #  'desired_velocity', 'forward_pid', 'reverse_pid', 'new_pid', 'initial_pid',
    #  'controller_switch_spike']
    feature_columns = ['mod_rack_position', 'rack_position', 'rack_velocity', 'tahn_rack_acceleration',
                       'desired_velocity']
    features_cnt = len(feature_columns)
    target_columns = ['new_pid']
    x = None
    y = None
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    file_num = min(num_sequences, len(files))
    if num_sequences < 0:
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

    return x, y


sequence_length = 50
num_sequences = -1  # -1 is to get all available samples

model = tf.keras.models.load_model('pid_neural_networks\\test_nn.keras')
model.summary()

test_x, test_y = get_pid_data(sequence_length=50, num_sequences=1)

predicted = model.predict(test_x)
plt.plot(test_y.flatten())
plt.plot(predicted.flatten())
plt.title('model cmp')
plt.ylabel('value')
plt.xlabel('index')
# plt.yscale('log')
plt.legend(['true value', 'predicted'], loc='upper left')
plt.show()


# Print some sample predictions
# for i in range(5):
#     print(f"True sequence: {y[i].flatten()}")
#     print(f"Predicted sequence: {predicted[i].flatten()}\n")
# for i in [1, 20, 45]:
#     plt.plot(y[i].flatten())
#     plt.plot(predicted[i].flatten())
#     plt.title('model cmp')
#     plt.ylabel('value')
#     plt.xlabel('index')
#     #plt.yscale('log')
#     plt.legend(['true value', 'predicted'], loc='upper left')
#     plt.show()