from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
import json
from keras.callbacks import ModelCheckpoint
import pickle
#The training data and test data are got from preprocessing.py
with open('X_train.p', mode='rb') as f:
    X_train = pickle.load(f)
with open('X_test.p', mode='rb') as f:
    X_test = pickle.load(f)
with open('y_train.p', mode='rb') as f:
    y_train = pickle.load(f)
with open('y_test.p', mode='rb') as f:
    y_test = pickle.load(f)
#Network architechture
model = Sequential()
#First conv layer
model.add(Convolution2D(32, 4, 4, input_shape=(32, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

#Second conv layer
model.add(Convolution2D(64, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

#Third conv layer
model.add(Convolution2D(96, 4, 4))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
#drop out
model.add(Dropout(0.5))

#Four fully-connected layers.
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(80))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('relu'))
model.add(Dense(1))

#train
model.compile('adam', 'mean_squared_error', ['mean_squared_error'])

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, verbose = 0, batch_size=128, validation_split = 0.25,callbacks=callbacks_list, nb_epoch=50)
#history = model.fit(X_train, y_train, verbose = 1, batch_size=128, nb_epoch=35)
#evaluate on test data
#metrics = model.evaluate(X_test, y_test)
#for metric_i in range(len(model.metrics_names)):
#    metric_name = model.metrics_names[metric_i]
#    metric_value = metrics[metric_i]
#    print('{}: {}'.format(metric_name, metric_value))

#save model.
json_model = model.to_json()
with open('model.json', 'w') as f:
    json.dump(json_model, f)
model.load_weights("weights-improvement-20-0.03.hdf5")
model.save_weights('./model.h5')