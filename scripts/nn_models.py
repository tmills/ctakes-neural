#!/usr/bin/env python

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Lambda, Embedding, Merge
from keras.layers import SimpleRNN, LSTM, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping

def get_mlp_model(dimension, num_outputs, layers=(64, 256, 256) ):
    model = Sequential()
    sgd = get_mlp_optimizer()

    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(layers[0], input_dim=dimension, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(layers[1], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(layers[2], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
#            model.add(Dense(layers[2], init='uniform'))
#            model.add(Activation('relu'))
#            model.add(Dropout(0.5))

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

def get_multitask_mlp(dimension, vocab_size, output_size_list, fc_layers = (64,), embed_dim=200 ):
    input = Input(shape=(dimension[1],), dtype='int32', name='Main_Input')
    
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=dimension[1])(input)
    
    for num_nodes in fc_layers:
        x = Dense(num_nodes, init='uniform')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
    
    outputs = []
    losses = {}
    loss_weights = {} ## don't do anything with these yet.
    
    for ind, output_size in enumerate(output_size_list):
        out_name = "Output_%d" % ind
        if output_size == 1:
            output = Dense(1, activation='sigmoid', init='uniform', name=out_name)(x)
            losses[out_name] = 'binary_crossentropy'
            outputs.append( output )
        else:
            output = Dense(output_size, activation='softmax', init='uniform', name=out_name)(x)
            
            losses[out_name] = 'categorical_crossentropy'
            outputs.append( output )
    
    sgd = get_mlp_optimizer()
    model = Model(input=input, output = outputs)
    model.compile(optimizer=sgd,
                 loss=losses)
    
    return model

def get_cnn_model(dimension, vocab_size, num_outputs, conv_layers = (64,), fc_layers=(64, 64, 256), embed_dim=200, filter_width=3 ):
    sgd = get_mlp_optimizer()

    input = Input(shape=(dimension[1],), dtype='int32', name='Main_Input')   
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=dimension[1])(input)

    ## Convolutional layers:
    x = (Convolution1D(conv_layers[0], filter_width, activation='relu'))(x)

    for nb_filter in conv_layers[1:]:
        x = Convolution1D(nb_filter, filter_width, activation='relu')(x)

    x = Lambda(max_1d, output_shape=(conv_layers[-1],))(x)

    
    #model.add(MaxPooling1D())

    for num_nodes in fc_layers:
        x = Dense(num_nodes, init='uniform')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

    out_name = "Output"
    if num_outputs == 1:
        output = Dense(1, init='uniform', activation='sigmoid', name=out_name)(x)
        loss = 'binary_crossentropy'
    else:
        output = Dense(num_outputs, init='uniform', activation='softmax', name=out_name)(x)
        loss='categorical_crossentropy'

    sgd = get_mlp_optimizer()
    model = Model(input=input, output=output)
    model.compile(optimizer = sgd,
                  loss = loss)
    
    return model

def get_multitask_cnn(dimension, vocab_size, output_size_list, conv_layers = (64,), fc_layers = (64,), embed_dim=200, filter_widths=(3,) ):
    input = Input(shape=(dimension[1],), dtype='int32', name='Main_Input')
    
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=dimension[1])(input)
    
    convs = []
    for width in filter_widths:
        conv = Convolution1D(conv_layers[0], width, activation='relu', init='uniform')(x)
        pooled = Lambda(max_1d, output_shape=(conv_layers[0],))(conv)
        convs.append(pooled)
    
    if len(convs) > 1:
        x = Merge(mode='concat') (convs)
    else:
        x = convs[0]
    
    for nb_filter in conv_layers[1:]:
        convs = []
        for width in filter_widths:
            conv = Convolution1D(nb_filter, filter_width, activation='relu', init='uniform')(x)    
            pooled = Lambda(max_1d, output_shape=(nb_filter,))(conv)
            convs.append(pooled)
        
        if len(convs) > 1:
            x = Merge(mode='concat')(convs)
        else:
            x = convs[0]
       
    for num_nodes in fc_layers:
        x = Dense(num_nodes, init='uniform')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
    
    outputs = []
    losses = {}
    loss_weights = {} ## don't do anything with these yet.
    
    for ind, output_size in enumerate(output_size_list):
        out_name = "Output_%d" % ind
        if output_size == 1:
            output = Dense(1, activation='sigmoid', init='uniform', name=out_name)(x)
            losses[out_name] = 'binary_crossentropy'
            outputs.append( output )
        else:
            output = Dense(output_size, activation='softmax', init='uniform', name=out_name)(x)
            
            losses[out_name] = 'categorical_crossentropy'
            outputs.append( output )
    
    sgd = get_mlp_optimizer()
    model = Model(input=input, output = outputs)
    model.compile(optimizer=sgd,
                 loss=losses)
    
    return model

def get_rnn_model(dimension, vocab_size, num_outputs, layers=(128,), embed_dim=100):
    feat_input = Input(shape=(None,), dtype='int32', name='Main_Input')
    #label_input = Input(shape=(None,3), dtype='int32', name='Label_Input')
    
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(feat_input)
     
    for layer_width in layers:
        x = LSTM(layer_width, return_sequences=True)(x)    
    
    
    if num_outputs == 1:
        activation = 'sigmoid' 
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    
    output = SimpleRNN(num_outputs, return_sequences=True, activation=activation)(x)
    
#    layer_merge = Merge(mode='concat', concat_axis=-1)([x,label_input])
    
#    output = TimeDistributed(Dense(num_outputs, activation=activation))(layer_merge)
    
    sgd = get_mlp_optimizer()
    optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08)
    
    model = Model(input=feat_input, output = output)
    model.compile(optimizer=optimizer,
                  loss = loss,
                  metrics=['accuracy'])
    
    ## This is known to be working but not as flexible: can delete when functional
    ## version is fully running -- still working on integrating previous time step
    ## classification as a feature
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size,
#                       output_dim=embed_dim,
#                       #input_length=dimension[1],
#                       dropout=0.5))
#     model.add(LSTM(layers[0],
#                  return_sequences=True,
#                  #input_length=dimension[1],
#                  input_dim=embed_dim))
#     model.add(TimeDistributed(Dense(num_outputs)))
#     model.add(Activation('softmax'))
# 
#     optimizer = RMSprop(lr=0.01,
#                       rho=0.9, epsilon=1e-08)
#     model.compile(loss='categorical_crossentropy',
#                 optimizer=optimizer,
#                 metrics=['accuracy'])
            

    return model

def get_early_stopper():
    return EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
    
def get_mlp_optimizer():
    return SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

def max_1d(X):
    return K.max(X, axis=1)
