#!/usr/bin/env python

from ctakesneural.models.nn_models import OptimizableModel
from ctakesneural.io import cleartk_io as ctk_io
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Merge, LSTM

import numpy as np
import random
import sys

class LstmEntityModel(OptimizableModel):
    def __init__(self, configs=None):
        if configs is None:
            ## Default is not smart -- single layer with between 50 and 1000 nodes
            self.configs = {}
            self.configs['embed_dim'] = (10,25,50,100,200)
            self.configs['layers'] = ( (50,), (100,), (200,), (500,), (1000,) )
            self.configs['batch_size'] = (32, 64, 128, 256)
        else:
            self.configs = configs

    def get_model(self, dimension, vocab_size, num_outputs, params):
        layers = params['layers']
        
        optimizer = self.param_or_default(params, 'optimizer', self.get_default_optimizer())
        weights = self.param_or_default(params, 'weights', None)
        regularizer = self.param_or_default(params, 'regularizer', self.get_default_regularizer())
        
        feat_input = Input(shape=(None,), dtype='int32', name='Main_Input')
        
        if weights is None:
            x = Embedding(input_dim=vocab_size, output_dim=params['embed_dim'], name='Embedding')(feat_input)
        else:
            print("Using pre-trained embeddings in bilstm model")
            x = Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[params['weights']])(feat_input)
        
        right = left = x
        for layer_width in layers:
            ## TODO - see if default LSTM activation is suitable, does it need to be config/param?
            left = LSTM(layer_width, return_sequences=False, go_backwards=False, W_regularizer=regularizer, U_regularizer=regularizer, name='forward_lstm')(left)
            right = LSTM(layer_width, return_sequences=False, go_backwards=True, W_regularizer=regularizer, U_regularizer=regularizer, name='backward_lstm')(right)
        
        x = Merge(mode='concat', name='merge_lstms')([left, right])
        
        if num_outputs == 1:
            out_activation = 'sigmoid' 
            loss = 'binary_crossentropy'
        else:
            out_activation = 'softmax'
            loss = 'categorical_crossentropy'
        
        output = Dense(num_outputs, activation=out_activation, name='dense_output')(x)
        
        model = Model(input=feat_input, output = output)
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
        config['layers'] = (100,)
        config['embed_dim'] = 100
        config['batch_size'] = 64
        return config
               
    def run_one_eval(self, epochs, config, params):
        train_x = params['train_x']
        train_y = params['train_y']
        valid_x = params['valid_x']
        valid_y = params['valid_y']
        model, history = self.train_model_for_data(train_x, train_y, epochs, config, params)
        return history.history['loss'][-1]

    def read_training_instances(self, working_dir):
        ## our inputs use the ctakes/cleartk standard for sequence input: 
        ## label | token1 * <e> [entity* ]</e> token2 *
        (labels, label_alphabet, feats, feats_alphabet) = ctk_io.read_token_sequence_data(working_dir)
        train_y = np.array(labels)
        train_y, indices = ctk_io.flatten_outputs(train_y)
                   
        self.label_alphabet = label_alphabet
        self.feats_alphabet = feats_alphabet
        return feats, train_y
    
    def read_test_instance(self, line, num_feats=-1):
        feats = [ctk_io.read_bio_feats_with_alphabet(feat, self.feats_alphabet) for feat in line.split()]

    def train_model_for_data(self, train_x, train_y, epochs, config, params={}):
        vocab_size = train_x.max() + 1
        params['layers'] = config['layers']
        params['embed_dim'] = config['embed_dim']
        num_outputs = 0
        if train_y.ndim == 1:
            num_outputs = 1
        elif train_y.shape[1] == 1:
            num_outputs = 1
        else:
            num_outputs = train_y.shape[1]
            
        model = self.get_model(-1, vocab_size, num_outputs, params)
        history = model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=config['batch_size'],
            verbose=1)
        return model, history
        
def main(args):
    if args[0] == 'train':
        working_dir = args[1]
        model = LstmEntityModel()
        train_x, train_y = model.read_training_instances(working_dir)
        trained_model, history = model.train_model_for_data(train_x, train_y, 80, model.get_default_config())
        trained_model.save(os.path.join(working_dir, 'model.h5'), overwrite=True)
        
        fn = open(os.path.join(working_dir, 'alphabets.pkl'), 'w')
        pickle.dump( (feature_alphabet, label_alphabet), fn)
        fn.close()

        with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
            myzip.write(os.path.join(working_dir, 'model.h5'), 'model.h5')
            myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')
    else:
        sys.stderr.write("Do not recognize args[0] command argument: %s\n" % (args[0]))
        sys.exit(-1)
        
if __name__ == "__main__":
    main(sys.argv[1:])
    