import keras
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.model import Sequential have to call in full from code, wtf
import tensorflow
from tensorflow.keras.models import Sequential # no error here actually, wtf
from numpy import loadtxt
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os


# test data for a single variable
x = np.linspace(0, 50, 1000)
y = 2*np.sin(x) + x

# Generate some dummy sequential data
def generate_data(sequence_length, num_sequences):
    x = np.array([np.sin(np.linspace(0, 2 * np.pi, sequence_length) + i) for i in range(num_sequences)])
    y = np.array([np.cos(np.linspace(0, 2 * np.pi, sequence_length) + i) for i in range(num_sequences)])
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1))
    return x, y

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


sequence_length = 100
num_sequences = 2000

# Generate data
X, y = generate_multivar_data(sequence_length, num_sequences) # generate_data(sequence_length, num_files)
# Reshape data to fit LSTM input requirements (samples, timesteps, features)

print('x shape after reshaping: ', X.shape, ' y shape: ', y.shape)
features_cnt = X.shape[2]

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, features_cnt), return_sequences=True))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))  # sequence_length


# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Create a TensorBoard callback
log_dir = os.path.join("logs", "fit", "latest_log")  # datetime.datetime.now().strftime("%Y%model%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

# Train the model
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0, callbacks=[tensorboard_callback])
model.summary()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# rescal_value = 2
# new_length = int(sequence_length * rescal_value)
# test_x = np.sin(np.linspace(0, rescal_value*2 * np.pi, new_length))
# test_y = np.cos(np.linspace(0, rescal_value*2 * np.pi, new_length))
# test_x = test_x.reshape(1, new_length, 1)
# Predict using the trained model
test_x, test_y = generate_multivar_data(sequence_length, 1)
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