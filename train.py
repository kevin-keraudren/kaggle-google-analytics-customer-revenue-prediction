# run tensorflow on CPU
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from preprocess import *

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Dropout
from keras import optimizers
import numpy as np

from keras.losses import mean_squared_error

from keras import backend as K

from scipy.special import expit

"""
LSTM example:
https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

Keras batch generator example:
https://www.kaggle.com/protan/keras-resnet50-with-batch-generator

Logistic function in Python:
https://stackoverflow.com/a/25164452

Notes
-----

 * we need to use either large batches or a generator of balanced batches.
 
"""


def batch_generator(X, Y, positive_examples, negative_examples,
                    batch_size=64, rng=np.random.RandomState(0)):
    """Batch generator for keras model."""

    while True:
        # randomly sample positive and negative examples
        positive_selection = positive_examples[rng.randint(0,
                                                           positive_examples.shape[0],
                                                           batch_size // 2)]
        negative_selection = negative_examples[rng.randint(0,
                                                           negative_examples.shape[0],
                                                           batch_size // 2)]
        x_batch = np.concatenate([X[positive_selection],
                                  X[negative_selection]])
        y_batch = np.concatenate([Y[positive_selection],
                                  Y[negative_selection]])

        yield x_batch, y_batch


def my_loss(y_true, y_pred):
    # return K.log(K.mean(1 + mean_squared_error(y_true, y_pred)))
    # return (K.sum(y_true, axis=[1, 2]) - K.sum(y_pred, axis=[1, 2])) ** 2
    return (K.log(1 + K.sum(y_true, axis=[1, 2])) - K.log(1 + K.sum(y_pred, axis=[1, 2]))) ** 2


def binary_crossentropy(x, y):
    return -(y * K.log(x) + (1 - y) * K.log(1 - x))


def regression_error(y_true, y_pred, epsilon=1e-8):
    inside = K.cast(K.not_equal(y_true[:, :, 1], -1), K.floatx())
    transactions = K.cast(K.greater(y_true[:, :, 1], 0), K.floatx())
    return (
        # learn to predict a regression for non-zero scores
        K.sum(transactions * K.square(y_pred[:, :, 1] - y_true[:, :, 1]),
              axis=[1]) / (epsilon + K.sum(transactions, axis=[1])) +

        # learn to predict which scores are non-zero
        K.sum(inside * binary_crossentropy(K.sigmoid(y_pred[:, :, 0]), transactions),
              axis=[1]) / K.sum(inside, axis=[1])
    )


def predict(model, x, seq_len):
    y = np.squeeze(model.predict([x]))[:seq_len]
    return (expit(y[:, 0]) < 0.5).astype('float32') * y[:, 1]
    # print(y,y.shape,expit(y[seq_len-1, 0]))
    # if expit(y[seq_len-1, 0]) < 0.5:
    #     return 0
    # else:
    #     return y[seq_len-1, 1]


def my_metric(y_true, y_pred):
    return K.mean(regression_error(y_true, y_pred))


def convert2log(a):
    res = a.sum()
    if res > 0:
        res = np.log(res)
    return res


def get_model(max_seq_length, feature_size, filename=None):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(max_seq_length, feature_size)))
    # model.add(TimeDistributed(Dense(1024, activation='relu')))
    # model.add(TimeDistributed(Dense(512, activation='relu')))
    # model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(2, activation='linear', return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(2, activation='linear')))
    # model.add(Dense(1, activation='linear'))

    # load previously trained model
    if filename is not None:
        print("Loading weights from", filename)
        model.load_weights(filename)

    model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                  loss=regression_error,
                  metrics=[my_metric])

    return model


if __name__ == "__main__":
    user_features = {}
    user_scores = {}
    for f in list(glob("train/*.csv"))[:30]:
        user_features, user_scores = encode_all_users(f, user_features, user_scores)

    user_features, user_scores = split_users(user_features, user_scores, max_visits=10)

    n_users = len(list(user_features.keys()))
    feature_size = len(user_features[list(user_features.keys())[0]][0])
    max_seq_length = np.max([len(scores) for scores in user_scores.values()])
    print("n_users:", n_users, 'max_seq_length:', max_seq_length, 'feature_size:', feature_size)

    input_data = np.zeros((n_users, max_seq_length, feature_size), dtype='float32')

    target_data = -np.ones((n_users, max_seq_length, 2), dtype='float32')

    nonzero_count = 0
    for i, (features, scores) in enumerate(zip(user_features.values(), user_scores.values())):
        features = np.array(features)
        scores = np.array(scores)
        nonzero_count += scores.sum() > 0
        input_data[i, :features.shape[0]] = features
        target_data[i, :scores.shape[0], 1] = scores  # [..., np.newaxis] #convert2log(scores)

    print("Nonzero count:", nonzero_count)
    print("User count:", len(user_features))
    print("Nonzero proba:", nonzero_count / len(user_features))

    model = get_model(max_seq_length, feature_size)

    # model.fit(input_data, target_data,
    #           batch_size=1000,
    #           epochs=100,
    #           validation_split=0.2,
    #           )

    positive_examples = np.squeeze(np.argwhere((target_data[:, :, 1] > 0).sum(axis=-1) > 0))
    negative_examples = np.squeeze(np.argwhere((target_data[:, :, 1] <= 0).sum(axis=-1) > 0))

    print("Positive and negative examples", len(positive_examples), len(negative_examples))

    fit_generator = batch_generator(input_data, target_data, positive_examples, negative_examples,
                                    rng=np.random.RandomState(123))
    val_generator = batch_generator(input_data, target_data, positive_examples, negative_examples,
                                    rng=np.random.RandomState(456))

    # https://stackoverflow.com/questions/51150468/error-you-must-compile-your-model-before-using-it-in-case-of-lstm-and-fit-gene

    model.fit_generator(fit_generator, steps_per_epoch=100, epochs=10,
                        verbose=1, callbacks=None, validation_data=val_generator,
                        validation_steps=10, class_weight=None,
                        max_queue_size=10, workers=1, use_multiprocessing=False,
                        shuffle=True, initial_epoch=0)

    # Save model
    model.save('s2s.h5')
