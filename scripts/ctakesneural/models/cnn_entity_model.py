#!/usr/bin/env python

from ctakesneural.models import nn_models
from ctakesneural.models.nn_models import read_model, max_1d, get_mlp_optimizer
from ctakesneural.models.entity_model import EntityModel
from ctakesneural.io import cleartk_io as ctk_io
from ctakesneural.opt.random_search import RandomSearch

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Lambda, Embedding
from keras.layers.merge import Concatenate

import numpy as np
import os.path
import pickle
import random
import sys
from zipfile import ZipFile


class CnnEntityModel(EntityModel):
    def __init__(self, configs=None):
        if configs is None:
            ## Default is not smart -- single layer with between 50 and 1000 nodes
            self.configs = {}
            self.configs['embed_dim'] = (10,25,50,100,200)
            self.configs['layers'] = ( (50,), (100,), (200,), (500,), (1000,) )
            self.configs['batch_size'] = (32, 64, 128, 256)
            self.configs['filters'] = ((64,), (128,), (256,), (512,), (1024,), (2048,))
            self.configs['widths'] = ( (2,), (3,), (4,), (2,3), (3,4), (2,3,4))
        else:
            self.configs = configs

    def get_random_config(self):
        config = {}
        config['layers'] = random.choice(self.configs['layers'])
        config['embed_dim'] = random.choice(self.configs['embed_dim'])
        config['batch_size'] = random.choice(self.configs['batch_size'])
        config['filters'] = random.choice(self.configs['filters'])
        config['width'] = random.choice(self.configs['widths'])
        return config
    
    def get_default_config(self):
        config = {}
        config['layers'] = (100,)
        config['embed_dim'] = 100
        config['batch_size'] = 64
        config['filters'] = (128,)
        config['width'] = (2,3)
        return config
    
    def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
        model, history = self.train_model_for_data(train_x, train_y, epochs, config, valid=0.1)
        loss = model.evaluate(valid_x, valid_y)
        return loss

    def get_model(self, dimension, vocab_size, num_outputs, config):
        input = Input(shape=(dimension[1],), dtype='int32', name='Main_Input')   
        x = Embedding(input_dim=vocab_size, output_dim=config['embed_dim'], input_length=dimension[1])(input)
    
        optimizer = self.param_or_default(config, 'optimizer', self.get_default_optimizer())
        weights = self.param_or_default(config, 'weights', None)
        regularizer = self.param_or_default(config, 'regularizer', self.get_default_regularizer())
        conv_layers = config['filters']
        
        convs = []
        for width in config['width']:
            conv = Convolution1D(conv_layers[0], width, activation='relu', kernel_initializer='uniform')(x)
            pooled = Lambda(max_1d, output_shape=(conv_layers[0],))(conv)
            convs.append(pooled)
        
        if len(convs) > 1:
            x = Concatenate() (convs)
        else:
            x = convs[0]
    
        for nb_filter in conv_layers[1:]:
            convs = []
            for width in config['width']:
                conv = Convolution1D(nb_filter, width, activation='relu', kernel_initializer='uniform')(x)
                pooled = Lambda(max_1d, output_shape=(nb_filter,))(conv)
                convs.append(pooled)
            
            if len(convs) > 1:
                x = Concatenate()(convs)
            else:
                x = convs[0]
           
        for num_nodes in config['layers']:
            x = Dense(num_nodes, kernel_initializer='uniform')(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
    
        out_name = "Output"
        if num_outputs == 1:
            output = Dense(1, kernel_initializer='uniform', activation='sigmoid', name=out_name)(x)
            loss = 'binary_crossentropy'
        else:
            output = Dense(num_outputs, kernel_initializer='uniform', activation='softmax', name=out_name)(x)
            loss='categorical_crossentropy'
    
        sgd = get_mlp_optimizer()
        model = Model(inputs=input, outputs=output)
            
        model.compile(optimizer = sgd,
                      loss = loss)
        
        return model
        

def main(args):
    if args[0] == 'train':
        working_dir = args[1]
        model = CnnEntityModel()
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
        model = read_model(working_dir)
        train_x, train_y = model.read_training_instances(working_dir)
        model = CnnEntityModel()
        optim = RandomSearch(model, train_x, train_y)
        best_config = optim.optimize()
        print("Best config: %s" % best_config)
    else:
        sys.stderr.write("Do not recognize args[0] command argument: %s\n" % (args[0]))
        sys.exit(-1)
        
if __name__ == "__main__":
    main(sys.argv[1:])

