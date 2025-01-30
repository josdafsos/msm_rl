# file to test regression with NNs
#import tensorflow.python.keras.models
# first neural network with keras make predictions
import keras
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.model import Sequential have to call in full from code, wtf
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from numpy import loadtxt
import matplotlib.pyplot as plt
import os

# load the dataset
# my data: x, 1: x*x, 2: sin(x), 3: sign(x), 4: sin(x)*x, 5: -x
dataset = loadtxt('test_data\\x_square.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0]
Y = dataset[:,4]
# define the keras model
#model = Sequential()
model = tensorflow.keras.models.Sequential()

model.add(keras.Input((1,)))
model.add(Dense(60,  activation='relu', kernel_initializer='he_uniform')) # input_shape=(1,),
#model.add(Dropout(0.2))
model.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
#model.add(Dropout(0.2))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1))
#model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
log_dir = os.path.join("logs", "latest_log")  # datetime.datetime.now().strftime("%Y%model%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10, verbose=0, callbacks=[tensorboard_callback])
# make class predictions with the model
t = np.linspace(-15, 15, 100)
x_sinx = np.sin(t) * t
predictions = model.predict(t)
plt.plot(t, predictions)
plt.plot(t, x_sinx)
plt.title('predicted plot')
plt.ylabel('prediction')
plt.xlabel('t')
plt.show()


# list all data in history
# print(history.history.keys())
# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
