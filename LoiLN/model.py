from utils import *
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np

import tensorflow as tf
from crf import CRF
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score, classification_report


class BiLSRM_CRF(object):
    def __init__(self, embedding_dim, vocab_size, n_tags, lr=1e-3):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_tags = n_tags
        self.lr = lr

    def f1(self, y_true, y_pred): #taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def build(self):
        model = tf.keras.Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True))
        model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_tags))

        crf = CRF(self.n_tags, sparse_target=True)
        model.add(crf)
        model.compile(loss=crf.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics=[self.f1])
        return model
