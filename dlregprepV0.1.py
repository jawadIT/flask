# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:59:28 2019

@author: jawad_zf1uaw5
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 10000

def get_data():
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    return dataset_path

def clean_data(dataset_path):
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,na_values = "?", comment='\t',sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0
    return dataset

def split_data(dataset):
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset,test_dataset

def plot_data(train_data):
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    plt.show()
    return plt

def get_stats(train_dataset):
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    print(train_stats)
    return train_stats

def label_seperation(train_dataset,test_dataset):
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    return train_labels,test_labels

def normalise_data(train_dataset,test_dataset):
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    return normed_train_data,normed_test_data

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

def build_model(train_dataset):
    model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

def model_summary(model):
    model.summary()
    return

def model_verify_before_fit(model,train_dataset):
    example_batch = train_dataset[:10]
    example_result = model.predict(example_batch)
    print(example_batch,example_result)

def model_training_with_progress(train_dataset,train_labels):

    history = model.fit(
            train_dataset, train_labels,
            epochs=EPOCHS, validation_split = 0.2, verbose=0,
            callbacks=[PrintDot()])

    return history

def plot_model_iterations_cost(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

def model_train_optimal_cost(train_dataset,train_labels):
    model = build_model(train_dataset)

# The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)

    history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                        validation_split = 0.2, verbose=0, 
                        callbacks=[early_stop, PrintDot()])
    
    plot_model_iterations_cost(history)
    
    return model

def model_evaluation(model):
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

def model_prediction_plot_error(model):
    test_predictions = model.predict(test_dataset).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    
    plt.figure()
    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    return 

def model_checkpoint():
    #import os
    checkpoint_path = "C:/jawad_zf1uaw5/models/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    model = build_model(train_dataset)

    #model.fit(train_images, train_labels,  epochs = 10,
    #          validation_data = (test_images,test_labels),
    #          callbacks = [cp_callback])

    model.fit(train_dataset, train_labels, epochs=70,
               validation_split = 0.2, verbose=0, callbacks=[cp_callback])

    return

def model_checkpoint_insteps():
    
    checkpoint_path = "C:/jawad_zf1uaw5/modelsteps/cp-{epoch:04d}.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True,
            period=5)

    model = build_model(train_dataset)
    model.save_weights(checkpoint_path.format(epoch=0))
#   model.fit(train_images, train_labels,
#          epochs = 50, callbacks = [cp_callback],
#          validation_data = (test_images,test_labels),
#          verbose=0)
    model.fit(train_dataset, train_labels, epochs=70,
              validation_split = 0.2, verbose=0, callbacks=[cp_callback])

#def model_load_weights():
#    checkpoint_path = "C:/jawad_zf1uaw5/models/cp.ckpt"
#    model.load_weights(checkpoint_path)
#    loss, mae, mse = model1.evaluate(normed_test_data, test_labels, verbose=0)
#    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    return 

def model_save_h5():

    model = build_model(train_dataset)

    history = model.fit(train_dataset, train_labels, epochs=8000,
              validation_split = 0.1, verbose=2)

    # Save entire model to a HDF5 file
    model.save('my_model.h5')
    
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['mean_absolute_error'])
    #plt.plot(history.history['mean_absolute_percentage_error'])
    #plt.plot(history.history['cosine_proximity'])
    plt.show()
    
    return

if __name__ == "__main__":
    print(("Entry in to model building"))
    dataset_path = get_data()
    dataset = clean_data(dataset_path)
    train_dataset,test_dataset = split_data(dataset)
    plot_data(train_dataset)
    train_stats = get_stats(train_dataset)
    train_labels,test_labels = label_seperation(train_dataset,test_dataset)
    #normed_train_data,normed_test_data = normalise_data(train_dataset,test_dataset)
    #model = build_model(train_dataset)
    #model_summary(model)
    #model_verify_before_fit(model,train_dataset)
    #history = model_training_with_progress(train_dataset,train_labels)
    #plot_model_iterations_cost(history)
    #model = model_train_optimal_cost(train_dataset,train_labels)
    #model_evaluation(model)
    #model_prediction_plot_error(model)
    #model_checkpoint()
    #model_checkpoint_insteps()
    model_save_h5()
    #model_load_weights()