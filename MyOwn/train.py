import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, MaxPooling2D
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from DataUtils import DataPreprocessing
from ImageUtils import ImageUtils
from datetime import datetime

NUM_OF_TRAIN = 500
if __name__ == '__main__':
    print("reading image paths===================")
    train_image_paths = ImageUtils.get_image_paths('../data/train', do_shuffle=True)
    test_image_paths = ImageUtils.get_image_paths('../data/valid', do_shuffle=True)
    #  train_image_paths = train_image_paths[:NUM_OF_TRAIN]
    #  test_image_paths = test_image_paths[:NUM_OF_TRAIN//10]
    print("reading training set===================")
    x_train = ImageUtils.read_multi_image(train_image_paths)
    y_train = DataPreprocessing.get_one_vs_hot_labels(train_image_paths)
    number_of_classes = len(y_train[0])
    print("reading test set===================")
    x_test = ImageUtils.read_multi_image(test_image_paths)
    y_test = DataPreprocessing.get_one_vs_hot_labels(test_image_paths)
    # build up model
    model = Sequential()
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3),padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(256, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3),padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(512, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3),padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(512, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(25088, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='Models/vgg_model_sgd.hdf5', verbose=1, save_best_only=True)
    print("training===================")
    model.fit(x_train, y_train, batch_size=128, epochs=100, callbacks=[checkpointer])
    print("testing====================")
    score = model.evaluate(x_test, y_test, batch_size=32)
    time_stamp = datetime.now().time()  # time object
    print("now =", time_stamp)
    print("type(time_stamp) =", type(time_stamp))
    model.save('Models/' + str(time_stamp) + 'final_vgg_model_sgd.h5')
    print(score)