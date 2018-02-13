#!/usr/bin/env python

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from ctakesneural.models.cnn_entity_model import CnnEntityModel

def main(args):
    if len(args) < 1:
        sys.stderr.write("One required argument: <h5 model file> [optional .pkl file]\n")
        sys.exit(-1)
        
    from keras.models import load_model
    from ctakesneural.models.nn_models import read_model
    model = load_model(args[0])
    conv = model.get_layer('conv1d_1')
    conv_weights = conv.get_weights()[0]
    embed_layer = model.get_layer('embedding_1')
    embed_weights = embed_layer.get_weights()[0]

    wordind_to_word = None
    if len(args) > 1:
        pkl_file = open(args[1], 'rb')
        model_wrapper = pickle.load(pkl_file)
        pkl_file.close()
        wordind_to_word = {int(val):key for (key,val) in model_wrapper.feats_alphabet.iteritems()}

    width, embed_dim, nb_filters = conv_weights.shape
  
    print("Shape of filter bank is %s"% (str(conv_weights[:,:,0].shape)))
    print("Shape of embeddings is %s" % (str(embed_weights.shape)))
    
    for filter_ind in range(nb_filters):
        filter = conv_weights[:,:,filter_ind]
        prod = np.dot(filter, embed_weights.transpose())
        # now prod is width x vocab_size
        max_inds = []
        for pos_ind in range(width):
            max_ind = np.argmax(prod[pos_ind,:])
            if wordind_to_word is None:
                max_inds.append(str(max_ind))
            else:
                max_inds.append(wordind_to_word[max_ind])
        
        print("Max word set for filter=%d is (%s)" % (filter_ind, ','.join(max_inds)))

    # x = np.arange(width+1)
    # y = np.arange(embed_dim+1)
    
    # z = conv_weights[:,:,0].transpose()
    
    # x, y = np.meshgrid(x, y)
    
    # ## Bring into the range 0<
    # z -= z.min()
    # ## scale to 0:1
    # z /= z.max()
    
    # plt.set_cmap('gray')
    # plt.pcolormesh(x, y, z)
    # plt.colorbar() #need a colorbar to show the intensity scale
    # plt.show() #boom
    
if __name__ == "__main__":
    main(sys.argv[1:])
