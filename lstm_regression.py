# import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization, Activation
import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.model import Sequential have to call in full from code, wtf
import tensorflow as tf
from tensorflow.keras.models import Sequential  # no error here actually, wtf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import os
import time
import utils

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

def get_model(create_new_model, sequence_length, features_cnt, network_name='test_controller'):
    if not create_new_model:
        print('loading existing model : ', network_name)
        model = keras.models.load_model(os.path.join('pid_neural_networks', network_name)) # + '.keras'))  # "test_controller_tf_2_10"; for new keras:  network_name + '.keras'
        return model
    # Create the LSTM model
    # options to try: Normalization (+), L1 or L2 norm. Dropout. He initialization (+); Batch normalization;
    # ELU activation (tested, does not improve NN)
    # He Initialization: Rectified Linear activation unit(ReLU) and Variants.
    # LeCun Initialization: Scaled Exponential Linear Unit(SELU)
    # options to try (continue):
    # Learning Rate Scheduler: Adjust the learning rate during training to ensure better convergence.
    # from tensorflow.keras.callbacks import ReduceLROnPlateau
    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    # options to try (continue): early stopping;
    model = keras.models.Sequential()  # model = Sequential()
    model.add(Input((sequence_length, features_cnt)))
    model.add(Dense(160, kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(LSTM(100, return_sequences=True))  # , input_shape=(sequence_length, features_cnt)
    model.add(Dense(160, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(LSTM(100, return_sequences=True))  # , input_shape=(sequence_length, features_cnt)
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1))  # sequence_length
    return model

num_files = -1  # -1 is to get all available samples
create_new_model = True  # if False, an existing network will be loaded
train_model = True
normalize_data_flag = True

data_folder = 'processed_data_frequency_' + str(utils.NN_WORKING_FREQUENCY) + '_Hz'

# Generate data
X, y = utils.get_pid_data(utils.sequence_length, num_files,
                          folder=data_folder,
                          print_time=True)  # generate_data(sequence_length, num_files)
# if num_files < 1:
print('total number of samples: ', X.shape[0])
if normalize_data_flag:
    X, scaler = utils.normalize_data(X, load_scaler=True, save_scaler=False)

features_cnt = X.shape[2]
network_name = "simple_Relu255_LSTM75_double.keras" # "no_batchnorm_controller_tf_2_10"  'test_controller' - used for training of tf newer version
model = get_model(create_new_model, utils.sequence_length, features_cnt, network_name=network_name)
# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.0000002)
# Create a TensorBoard callback
log_dir = os.path.join("logs", "fit", "latest_log")  # datetime.datetime.now().strftime("%Y%model%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

# Train the model
if train_model:
    t0 = time.time()
    history = model.fit(X, y, epochs=350, batch_size=192, validation_split=0.1, verbose=1, callbacks=[tensorboard_callback, lr_scheduler])
    print('training finished in ', time.time() - t0, ' seconds')
    model.save(os.path.join('pid_neural_networks', 'no_batchnorm_controller_py_3_12.keras'))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
model.summary()

#print(model.modelFolder)
# tf.saved_model.save(model, 'pid_neural_networks\\test_svd_mdl')

# # model.save(os.path.join('pid_neural_networks', 'test_nn.h5'))  # for keras file saving
# model.export(os.path.join('pid_neural_networks', 'export_test_nn_16'))
# #model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model)
# # Save the ONNX model
# #onnx.save_model(model_proto, os.path.join('pid_neural_networks', 'model.onnx'))
# output_path = os.path.join('pid_neural_networks', 'model.onnx')
# model.output_names=['output']
# input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='digit')]
#
# # spec = (tf.TensorSpec((sequence_length, features_cnt), tf.float32, name="input"),)
# model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13) #, input_signature=spec, opset=13)
# onnx.save_model(model_proto, output_path)





# rescal_value = 2
# new_length = int(sequence_length * rescal_value)
# test_x = np.sin(np.linspace(0, rescal_value*2 * np.pi, new_length))
# test_y = np.cos(np.linspace(0, rescal_value*2 * np.pi, new_length))
# test_x = test_x.reshape(1, new_length, 1)
# Predict using the trained model
sequence_length = 100
test_x, test_y = utils.get_pid_data(sequence_length=sequence_length, num_files=2, folder=data_folder)  # sequence_length=50

print('test_x orig shape: ', test_x.shape)
print('test_y orig shape: ', test_y.shape)






if normalize_data_flag:
    test_x, _ = utils.normalize_data(test_x, scaler=scaler)

test_x = test_x[-1]  # getting only first entry
test_x = test_x.reshape(1, sequence_length, features_cnt)
test_y = test_y[-1]  # getting only first entry
test_y = test_y.reshape(1, sequence_length, 1)

test_x = tf.keras.utils.pad_sequences(test_x, maxlen=utils.sequence_length, padding='pre', dtype='float32')  # is pre padding actually correct?
test_y = tf.keras.utils.pad_sequences(test_y, maxlen=utils.sequence_length, padding='pre', dtype='float32')  # is pre padding actually correct?

print('test_x new shape: ', test_x.shape)
print('test_y new shape: ', test_y.shape)
# print('test_x:\n', test_x)
# print('test_y:\n', test_y)
# --- data has padding now ---


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