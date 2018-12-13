#!/usr/bin/env python

import os
import pickle
import sys

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Lambda, Embedding, Concatenate
from keras.layers import SimpleRNN, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l1, l2, l1_l2
from zipfile import ZipFile

from nn_models import OptimizableModel

class KerasModel(OptimizableModel):

    def train_model_for_data(self, train_x, train_y, epochs, config, valid=0.1, use_class_weights=True, checkpoint_prefix=None, early_stopping=False):
        vocab_size = train_x.max() + 1
        class_weights = {}
        num_outputs = 0
        callbacks = [] # [ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)]
        if early_stopping:
            callbacks.append(self.get_early_stopper())

        if train_y.ndim == 1:
            num_outputs = 1
            ## 1-dim array of 0 and 1
            one_proportion = float(train_y.sum()) / len(train_y)
            one_weight = 0.5 / one_proportion
            zero_weight = 0.5 / (1. - one_proportion)
            class_weights[0] = zero_weight
            class_weights[1] = one_weight
        elif train_y.shape[1] == 1:
            num_outputs = 1
        else:
            num_outputs = train_y.shape[1]

        if not checkpoint_prefix is None:
            callbacks.append(self.get_checkpointer(checkpoint_prefix))

        model = self.get_model(train_x.shape, vocab_size, num_outputs, config)
        history = model.fit(train_x,
            train_y,
            epochs=epochs,
            batch_size=config['batch_size'],
            validation_split=valid,
            callbacks=callbacks,
            verbose=2)

        if not checkpoint_prefix is None:
            print("Loading best checkpointed model:")
            model = load_model(checkpoint_prefix + ".h5")

        return model, history
    
    ## This might be general to all models?
    def write_model(self, working_dir, trained_model):
        trained_model.save(os.path.join(working_dir, 'model_weights.h5'), overwrite=True)
        fn = open(os.path.join(working_dir, 'model.pkl'), 'wb')
        pickle.dump(self, fn)
        fn.close()

        with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
            myzip.write(os.path.join(working_dir, 'model_weights.h5'), 'model_weights.h5')
            myzip.write(os.path.join(working_dir, 'model.pkl'), 'model.pkl')

    def get_early_stopper(self):
        return EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    
    def get_checkpointer(self, fileprefix):
        return ModelCheckpoint('%s.h5' % fileprefix, monitor='val_loss', save_best_only=True)

    def get_default_optimizer(self):
        return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    def get_default_regularizer(self):
        return l2(0.1)

    def get_model(self, dimension, vocab_size, num_outputs, params):
        raise NotImplementedError("Subclass must implement get_model()")

    @staticmethod
    def read_model(working_dir):
        with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
            myzip.extract('model_weights.h5', working_dir)
            myzip.extract('model.pkl', working_dir)

        model = pickle.load( open(os.path.join(working_dir, 'model.pkl'), 'rb' ) )
        model.framework_model = load_model(os.path.join(working_dir, "model_weights.h5"))
        model.label_lookup = {val:key for (key,val) in model.label_alphabet.items()}
        return model
