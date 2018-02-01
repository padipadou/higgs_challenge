# Usage of keras models to predict class
#
# Authors : Paul-Alexis Dray,
#           Adrian Ahne
# Date : 01-02-2018
#
# Information: For testing just uncomment the piece of code you need
# We put everything in comments to avoid that everything is printed at the same time

import numpy as np
import keras.backend as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from keras.models import model_from_json
from grid_search_pipeline import load_data, make_pipeline
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return keras.sum(s, axis=0)


# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=keras.variable(value=0.5)):
    y_pred = keras.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = keras.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = keras.sum(y_pred - y_pred * y_true)
    return FP / N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=keras.variable(value=0.5)):
    y_pred = keras.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = keras.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = keras.sum(y_pred * y_true)
    return TP / P


def create_model(input_dim_=23):
    activation_func = 'relu'
    architecture_name = activation_func
    hidden_60x2 = 0  # hidden layers

    model = Sequential()

    model.add(Dense(30, input_dim=input_dim_, kernel_initializer='normal', activation=activation_func))

    if hidden_60x2 >= 1:
        model.add(Dense(60, kernel_initializer='normal', activation=activation_func))
        model.add(Dense(60, kernel_initializer='normal', activation=activation_func))
        model.add(Dense(60, kernel_initializer='normal', activation=activation_func))
        model.add(Dropout(0.5))

    if hidden_60x2 == 2:
        model.add(Dense(60, kernel_initializer='normal', activation=activation_func))
        model.add(Dense(60, kernel_initializer='normal', activation=activation_func))
        model.add(Dense(60, kernel_initializer='normal', activation=activation_func))
        model.add(Dropout(0.5))

    model.add(Dense(30, input_dim=input_dim_, kernel_initializer='normal', activation=activation_func))
    model.add(Dropout(0.5))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer="adagrad", metrics=['accuracy', auc])
    return model


def interesting_plots(history):
    fig = plt.figure(figsize=(10, 10))
    sub1 = fig.add_subplot(221)
    sub1.set_title('model accuracy')
    sub1.plot(history.history['acc'])
    sub1.plot(history.history['val_acc'])
    sub1.set_ylabel('accuracy')
    sub1.set_xlabel('epoch')
    sub1.legend(['train', 'test'], loc='upper left')

    sub1 = fig.add_subplot(222)
    sub1.set_title('model loss')
    sub1.plot(history.history['loss'])
    sub1.plot(history.history['val_loss'])
    sub1.set_ylabel('loss')
    sub1.set_xlabel('epoch')
    sub1.legend(['train', 'test'], loc='upper left')

    sub1 = fig.add_subplot(223)
    sub1.set_title('model auc')
    sub1.plot(history.history['auc'])
    sub1.plot(history.history['val_auc'])
    sub1.set_ylabel('auc')
    sub1.set_xlabel('epoch')
    sub1.legend(['train', 'test'], loc='upper left')

    plt.show()
    return


if __name__ == '__main__':
    print("Load data...")
    X, y = load_data()

    print("Building the pipeline...")
    pipeline = make_pipeline(model=False)

    # -- Train-test split
    print("Train test splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

    pipeline.fit(X_train, y_train)
    pipeline.transform(X_train)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # -- Create model
    model, architecture_name = create_model()
    # print(model.summary())  # Model architecture

    history = model.fit(X, y, validation_split=0.2, epochs=40, batch_size=50, verbose=1)

    # -- Plot
    interesting_plots(history)

    activation_func = "relu"
    model_name = activation_func + "_30_30_1"

    # -- Save the model
    with open('models/nn_{}_40_epochs.json'.format(model_name), 'w') as f:
        json.dump(model.to_json(), f, ensure_ascii=False)
    model.save_weights('models/nn_{}_40_epochs.h5'.format(model_name))

    # -- Load the model
    # Best model: 30-60-60-30-1 Relu activation
    # with open('models/nn_{}_40_epochs.json'.format(model_name)) as json_string:
    #     model = model_from_json(json_string)
    # model.load_weights('models/nn_{}_40_epochs.h5'.format(model_name))
