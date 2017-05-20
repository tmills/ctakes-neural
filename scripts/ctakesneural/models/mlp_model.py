#!/usr/bin/env python

from .nn_models import OptimizableModel
import random
from sklearn.datasets import load_svmlight_file

class MlpModel(OptimizableModel):
    
    def __init__(self, configs):
        if configs is None:
            ## Default is not smart -- single layer with between 50 and 1000 nodes
            self.configs = []
            self.configs['layers'] = ( (50,), (100,), (200,), (500,), (1000,))
        else:
            self.configs = configs
        
    def get_model(self, dimension, vocab_size, num_outputs, params):
        layers = params['layers']
        sgd = get_mlp_optimizer()
        
        model = Sequential()
        prev_dim = dimension
        for layer_size in layers:
            model.add(Dense(layer_size, input_dim=prev_dim, init='uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            prev_dim = layer_size
         
        if num_outputs == 1:
            model.add(Dense(1, init='uniform'))
            model.add(Activation('sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])
        else:
            model.add(Dense(num_outputs, init='uniform'))
            model.add(Activation('softmax'))                
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])
        return model
    
    def get_random_config(self):
        config = {}
        layers = random.choice(self.configs['layers'])
        config['layers'] = layers
        return config
    
    def run_one_eval(self, epochs, config,  train_x, train_y, valid_x, valid_y):
        model = self.get_model(train_x.shape[1], -1, 1 if train_y.shape[1] == 1 else train_y.shape[1], config)
        history = model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=128,
            verbose=1,
            validation_data=(valid_x, valid_y))
        return history.history['loss'][-1]
        
    def read_training_instances(self, filename):
        ## Assume our inputs are numerical vectors in liblinear format:
        X_train, y_train = load_svmlight_file(filename)
        self.num_feats = X_train.shape[1]
        return X_train, y_train
    
    def read_test_instance(self, line, num_feats=-1):
        feat_split = line.split(" ")
        str_feats = [feat.split(':') for feat in feat_split]
        
        if num_feats != -1:
            x_vec = np.zeros(num_feats)
        elif self.num_feats != -1:
            x_vec = np.zeros(self.num_feats)
        else:
            x_vec = np.zeros(str_feats[-1][0]+1)
        
        for feat in str_feats:
            x_vec[int(feat[0])] = float(feat[1])
        
        return x_vec
        
def get_mlp_optimizer():
    return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
