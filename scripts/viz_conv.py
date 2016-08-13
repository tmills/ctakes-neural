#!/usr/bin/env python

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(argv):
    if len(argv) < 1:
        sys.stderr.write("One required argument: <h5 model file>\n")
        sys.exit(-1)
        
    h5fp = h5py.File(argv[0], 'r')
    conv_layer = h5fp['convolution1d_1']
    conv_weights = conv_layer['convolution1d_1_W']
    (nb_filters, embed_dim, width, channels) = conv_weights.shape
    
    x = np.arange(width+1)
    y = np.arange(embed_dim+1)
    
    z = conv_weights[0,:,:,0]
    
    x, y = np.meshgrid(x, y)
    
    ## Bring into the range 0<
    z -= z.min()
    ## scale to 0:1
    z /= z.max()
    
    plt.set_cmap('gray')
    plt.pcolormesh(x, y, z)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show() #boom
    
if __name__ == "__main__":
    main(sys.argv[1:])
