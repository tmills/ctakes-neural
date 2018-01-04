#!/usr/bin/env python

from ctakesneural.models import nn_models
from ctakesneural.models.nn_models import read_model
from ctakesneural.models.entity_model import EntityModel
from ctakesneural.io import cleartk_io as ctk_io
from ctakesneural.opt.random_search import RandomSearch

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers.merge import Concatenate

import numpy as np
import os.path
import pickle
import random
import sys
from zipfile import ZipFile


class LstmEntityModel(EntityModel):
    def __init__(self, configs=None):
        if configs is None:
            ## Default is not smart -- single layer with between 50 and 1000 nodes
            self.configs = {}
            self.configs['embed_dim'] = (10,25,50,100,200)
            self.configs['layers'] = ( (50,), (100,), (200,), (500,), (1000,) )
            self.configs['batch_size'] = (32, 64, 128, 256)
        else:
            self.configs = configs

    def get_model(self, dimension, vocab_size, num_outputs, config):
        layers = config['layers']
        
        optimizer = self.param_or_default(config, 'optimizer', self.get_default_optimizer())
        weights = self.param_or_default(config, 'weights', None)
        regularizer = self.param_or_default(config, 'regularizer', self.get_default_regularizer())
        
        feat_input = Input(shape=(None,), dtype='int32', name='Main_Input')
        
        if weights is None:
            x = Embedding(input_dim=vocab_size, output_dim=config['embed_dim'], name='Embedding')(feat_input)
        else:
            print("Using pre-trained embeddings in bilstm model")
            x = Embedding(input_dim=vocab_size, output_dim=config['embed_dim'], weights=[config['weights']])(feat_input)
        
        right = left = x
        for layer_width in layers:
            ## TODO - see if default LSTM activation is suitable, does it need to be config/param?
            left = LSTM(layer_width, return_sequences=False, go_backwards=False, recurrent_regularizer=regularizer, activity_regularizer=regularizer, name='forward_lstm')(left)
            right = LSTM(layer_width, return_sequences=False, go_backwards=True, recurrent_regularizer=regularizer, activity_regularizer=regularizer, name='backward_lstm')(right)
        
        x = Concatenate(name='merge_lstms')([left, right])
        
        if num_outputs == 1:
            out_activation = 'sigmoid' 
            loss = 'binary_crossentropy'
        else:
            out_activation = 'softmax'
            loss = 'categorical_crossentropy'
        
        output = Dense(num_outputs, activation=out_activation, name='dense_output')(x)
        
        model = Model(inputs=feat_input, outputs = output)
        model.compile(optimizer=optimizer,
                      loss = loss,
                      metrics=['accuracy'])

        return model
                
    def get_random_config(self):
        config = {}
        config['layers'] = random.choice(self.configs['layers'])
        config['embed_dim'] = random.choice(self.configs['embed_dim'])
        config['batch_size'] = random.choice(self.configs['batch_size'])
        return config

    def get_default_config(self):
        config = {}
        config['layers'] = (50,)
        config['embed_dim'] = 10
        config['batch_size'] = 32
        return config
               
    def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
        model, history = self.train_model_for_data(train_x, train_y, epochs, config, valid=0.1)
        loss = model.evaluate(valid_x, valid_y)
        return loss[0]


def main(args):
    if len(args) < 2:
        sys.stderr.write('Two required arguments: <train|classify|optimize> <data directory>\n')
        sys.exit(-1)

    if args[0] == 'train':
        working_dir = args[1]
        model = LstmEntityModel()
        train_x, train_y = model.read_training_instances(working_dir)
        trained_model, history = model.train_model_for_data(train_x, train_y, 80, model.get_default_config())
        model.write_model(working_dir, trained_model)
        
    elif args[0] == 'classify':
        working_dir = args[1]
        model = read_model(working_dir)
     
        while True:
            try:
                line = sys.stdin.readline().rstrip()
                if not line:
                    break
                
                label = model.classify_line(line)
                print(label)
                sys.stdout.flush()
            except Exception as e:
                print("Exception %s" % (e) )
    elif args[0] == 'optimize':
        working_dir = args[1]
        model = LstmEntityModel()
        train_x, train_y = model.read_training_instances(working_dir)
        optim = RandomSearch(model, train_x, train_y)
        best_model = optim.optimize()
        print("Best config: %s" % best_model)
    else:
        sys.stderr.write("Do not recognize args[0] command argument: %s\n" % (args[0]))
        sys.exit(-1)
        
if __name__ == "__main__":
    main(sys.argv[1:])
    
