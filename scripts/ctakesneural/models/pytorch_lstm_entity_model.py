#!/usr/bin/env python

from ctakesneural.opt.random_search import RandomSearch
from pytorch_model import PytorchModel

import torch
import torch.optim as optim
import torch.nn as nn
from entity_model import EntityModel
import torch.nn.functional as F

import random
import sys

## This code is derived (and some copy/pasted) from the pytorch websites
## sequence model tutorial:
## http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class LstmEntityModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LstmEntityModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return None
                
    def forward(self, sentence):
        # sentence is batch_Size x length
        embeds = self.word_embeddings(sentence)
        # embeds is batch_size x length x embed_dims, 
        # but lstm wants length x batch x embed_dims
        lstm_out, hidden = self.lstm(
            embeds.permute(1,0,2))
        tag_space = self.hidden2tag(hidden[0].squeeze())
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space
    
        
class PytorchLstmEntityModelTrainer(EntityModel,PytorchModel):
    def __init__(self, configs=None):
        if configs is None:
            ## Default is not smart -- single layer with between 50 and 1000 nodes
            self.configs = {}
            self.configs['embed_dim'] = (10,25,50,100,200)
            self.configs['hidden_dims'] = ( 50, 100, 200, 500, 1000 )
            self.configs['batch_size'] = (32, 64, 128, 256)
        else:
            self.configs = configs
        self.input_len = 100

    def get_model(self, dimension, vocab_size, num_outputs, config):
        hidden_dims = config['hidden_dims']
        embedding_dims = config['embed_dim']
        
        # optimizer = self.param_or_default(config, 'optimizer', self.get_default_optimizer())
        # weights = self.param_or_default(config, 'weights', None)
        # regularizer = self.param_or_default(config, 'regularizer', self.get_default_regularizer())
        
        model = LstmEntityModel(embedding_dims, hidden_dims, vocab_size, num_outputs)
                
        return model
    
    def get_standard_input_len(self):
        return self.input_len
    
    def get_default_config(self):
        config = {}
        config['embed_dim'] = 100
        config['hidden_dims'] = 100
        config['batch_size'] = 64
        config['loss_fn'] = nn.BCEWithLogitsLoss()
        config['opt_fn'] = lambda model: optim.Adam(model.parameters(), lr=0.001)
        
        return config

    def get_random_config(self):
        config = {}
        config['hidden_dims'] = random.choice(self.configs['hidden_dims'])
        config['embed_dim'] = random.choice(self.configs['embed_dim'])
        config['batch_size'] = random.choice(self.configs['batch_size'])
        config['loss_fn'] = nn.BCEWithLogitsLoss()
        return config

    def predict_one_instance(self, X):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.sigmoid(self.framework_model(torch.LongTensor(X).to(device))).cpu().detach().numpy()
    
    def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
        model,_ = self.train_model_for_data(train_x, train_y, epochs, config, valid=0.1)
        # Copy the evaluation code from the pytorch model class

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        valid_input = torch.LongTensor(valid_x).to(device)
        pred = torch.sigmoid(model(valid_input)).cpu().detach().numpy()

        bin_pred = (pred > 0.5)

        acc = (bin_pred == valid_y).sum() / len(valid_y)

        return 1 - acc


def main(args):
    if len(args) < 2:
        sys.stderr.write('Two required arguments: <train|classify|optimize> <data directory>\n')
        sys.exit(-1)

    if args[0] == 'train':
        working_dir = args[1]
        model = PytorchLstmEntityModelTrainer()
        train_x, train_y = model.read_training_instances(working_dir)
        trained_model, history = model.train_model_for_data(train_x, train_y, 80, model.get_default_config())
        model.write_model(working_dir, trained_model)
        
    elif args[0] == 'classify':
        working_dir = args[1]
        model = PytorchModel.read_model(working_dir)
     
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
        model = PytorchLstmEntityModelTrainer()
        train_x, train_y = model.read_training_instances(working_dir)
        optim = RandomSearch(model, train_x, train_y)
        best_model = optim.optimize()
        print("Best config: %s" % best_model)
    else:
        sys.stderr.write("Do not recognize args[0] command argument: %s\n" % (args[0]))
        sys.exit(-1)
        
if __name__ == "__main__":
    main(sys.argv[1:])
