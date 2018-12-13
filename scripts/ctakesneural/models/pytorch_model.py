#!/usr/bin/env python

import os
import pickle
import sys
import tempfile
from zipfile import ZipFile

from nn_models import OptimizableModel
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, RandomSampler, BatchSampler

class PytorchModel(OptimizableModel):
    def __init__(self):
        super(PytorchModel,self).__init__()
        print('here we go')

    def train_model_for_data(self, train_x, train_y, epochs, config, valid=0.1, use_class_weights=True, checkpoint_prefix=None, early_stopping=False):
        vocab_size = train_x.max() + 1
        class_weights = {}
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = self.get_model(train_x.shape, vocab_size, num_outputs, config).to(device)
        batch_size = config['batch_size']
        loss_fn = config['loss_fn'].to(device)
        opt = config['opt_fn'](model) if 'opt_fn' in config else self.get_default_optimizer(model)

        # num_batches = train_x.shape[0] // batch_size
        tensor_dataset = TensorDataset(torch.LongTensor(train_x), torch.FloatTensor(train_y))

        train_size = int((1-valid) * len(tensor_dataset))
        valid_size = len(tensor_dataset) - train_size
        train_dataset, dev_dataset = random_split(tensor_dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = -1
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for i_batch, sample_batched in enumerate(train_loader):
                model.zero_grad()

                batch_x = sample_batched[0].to(device)
                batch_y = sample_batched[1].to(device)

                pred_y = model(batch_x)
                loss = loss_fn(pred_y, batch_y)
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
            
            model.eval()
            pred_y = model(dev_dataset[:][0].to(device))
            val_loss = loss_fn(pred_y, dev_dataset[:][1].to(device))
            if val_loss < best_val_loss or best_val_loss < 0:
                best_val_loss = val_loss
                outdir = tempfile.gettempdir()
                if not checkpoint_prefix is None:
                    torch.save(model, os.path.join(outdir, '%s_best_model.pt' % (checkpoint_prefix,)))
                else:
                    torch.save(model, os.path.join(outdir, 'best_model.pt'))
            print('Epoch %d: Training loss=%f, validation loss=%f' % (epoch, epoch_loss, val_loss.item()))
        
        if best_val_loss > 0:
            if not checkpoint_prefix is None:
                best_model = torch.load(os.path.join(outdir, '%s_best_model.pt' % (checkpoint_prefix,)))
            else:
                best_model = torch.load(os.path.join(outdir, 'best_model.pt'))
        else:
            raise Exception('No good models found!')
        
        return best_model, None



    def write_model(self, working_dir, trained_model):
        torch.save(trained_model, os.path.join(working_dir, 'model.pt'))
        with open(os.path.join(working_dir, 'model.pkl'), 'wb') as fn:
            pickle.dump(self, fn)

        with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
            myzip.write(os.path.join(working_dir, 'model.pt'), 'model.pt')
            myzip.write(os.path.join(working_dir, 'model.pkl'), 'model.pkl')

    def get_default_optimizer(self, model):
        return optim.Adam(model.parameters())

    def get_default_regularizer(self):
        return None

    def get_model(self, dimension, vocab_size, num_outputs, params):
        raise NotImplementedError("Subclass must implement get_model()")

    @staticmethod
    def read_model(working_dir):
        with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
            myzip.extract('model.pt', working_dir)
            myzip.extract('model.pkl', working_dir)

        with open(os.path.join(working_dir, 'model.pkl'), 'rb') as mf:
            model = pickle.load( mf )
        
        model.framework_model = torch.load(os.path.join(working_dir, 'model.pt'))
        model.label_lookup = {val:key for (key,val) in model.label_alphabet.items()}
        return model
