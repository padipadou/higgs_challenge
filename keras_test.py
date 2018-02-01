import numpy as np
import keras.backend as keras
import tensorflow as tf
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


if __name__ == '__main__':
    print("Load data...")
    X, y = load_data()

    print("Building the pipeline...")
    pipeline = make_pipeline(model=False)

    # -- Train-test split
    print("Train test splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    pipeline.fit(X_train, y_train)
    pipeline.transform(X_train)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # -- Create model
    model = create_model()
    print(model.summary())

    pass
