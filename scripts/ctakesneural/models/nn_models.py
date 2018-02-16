#!/usr/bin/env python

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Lambda, Embedding, Merge
from keras.layers import SimpleRNN, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l1, l2, l1_l2
from zipfile import ZipFile
import os.path
import pickle

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

def get_cnn_model(dimension, vocab_size, num_outputs, conv_layers = (64,), fc_layers=(64, 64, 256), embed_dim=200, filter_widths=(3,)):
    sgd = get_mlp_optimizer()

    input = Input(shape=(dimension[1],), dtype='int32', name='Main_Input')   
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=dimension[1])(input)

    ## FIXME: allow for filter_width to be a tuple like in the multitask version.
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

def get_multitask_cnn(dimension, vocab_size, output_size_list, conv_layers = (64,), fc_layers = (64,), embed_dim=(200,), filter_widths=(3,)):

    if type(embed_dim) is int:
        num_views = 1
        embed_dim = [embed_dim]
    else:
        num_views = len(embed_dim)
    
    if type(vocab_size) is int:
        vocab_size = [vocab_size]
    
    input_views = []
    embeddings = []
    
    for view in range(num_views):
        input_views.append(Input(shape=(dimension[1],), dtype='int32', name='Main_Input_%d' % view))
        embeddings.append(Embedding(input_dim=vocab_size[view], output_dim=embed_dim[view], input_length=dimension[1])(input_views[-1]))
    
    if len(embeddings) > 1:
        x = Merge(mode='concat')(embeddings)
    else:
        x = embeddings[0]
    
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
    model = Model(input=input_views, output = outputs)
    model.compile(optimizer=sgd,
                 loss=losses)
    
    return model

def get_bio_lstm_model(dimension, vocab_size, num_outputs, layers=(128,), embed_dim=100, go_backwards=False, activation='tanh', weights=None, lr=0.01):
    feat_input = Input(shape=(None,), dtype='int32', name='Main_Input')
    #label_input = Input(shape=(None,3), dtype='int32', name='Label_Input')
    
    if weights is None:
        x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(feat_input)
    else:
        #print("Using pre-trained embeddings in uni-lstm model")
        x = Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[weights])(feat_input)
     
    for layer_width in layers:
        x = LSTM(layer_width, return_sequences=True, go_backwards=go_backwards, activation=activation, W_regularizer=get_regularizer(), U_regularizer=get_regularizer())(x)
    
    
    if num_outputs == 1:
        out_activation = 'sigmoid' 
        loss = 'binary_crossentropy'
    else:
        out_activation = 'softmax'
        loss = 'categorical_crossentropy'
    
    output = SimpleRNN(num_outputs, return_sequences=True, activation=out_activation, go_backwards=go_backwards)(x)
    #output = TimeDistributed(Dense(num_outputs))(x)
    
    #sgd = get_mlp_optimizer()
    optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08)
    
    model = Model(input=feat_input, output = output)
    model.compile(optimizer=optimizer,
                  loss = loss,
                  metrics=['accuracy'])

    return model

def get_bio_bilstm_model(dimension, vocab_size, num_outputs, layers=(128,), embed_dim=100, go_backwards=False, activation='tanh', weights=None, lr=0.01):
    feat_input = Input(shape=(None,), dtype='int32', name='Main_Input')
    
    if weights is None:
        x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(feat_input)
    else:
        print("Using pre-trained embeddings in bilstm model")
        x = Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[weights])(feat_input)
    
    right = left = x
    for layer_width in layers:
        left = LSTM(layer_width, return_sequences=True, go_backwards=False, activation=activation, W_regularizer=get_regularizer(), U_regularizer=get_regularizer())(left)
        right = LSTM(layer_width, return_sequences=True, go_backwards=True, activation=activation, W_regularizer=get_regularizer(), U_regularizer=get_regularizer())(right)
        
    x = Merge(mode='concat')([left, right])
    
    if num_outputs == 1:
        out_activation = 'sigmoid' 
        loss = 'binary_crossentropy'
    else:
        out_activation = 'softmax'
        loss = 'categorical_crossentropy'
    
    output = SimpleRNN(num_outputs, return_sequences=True, activation=out_activation, go_backwards=go_backwards)(x)
    #output = TimeDistributed(Dense(num_outputs))(x)
    
    #sgd = get_mlp_optimizer()
    optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08)
    
    model = Model(input=feat_input, output = output)
    model.compile(optimizer=optimizer,
                  loss = loss,
                  metrics=['accuracy'])

    return model

def get_early_stopper():
    return EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    
def get_mlp_optimizer():
    return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

def max_1d(X):
    return K.max(X, axis=1)

def get_regularizer(l1=0.01, l2=0.01):
    return l1_l2(l1=l1, l2=l2)

def get_checkpointer(fileprefix):
    return ModelCheckpoint('%s.h5' % fileprefix, monitor='val_loss', save_best_only=True)

class OptimizableModel:
    def get_random_config(self):
        raise NotImplementedError("Subclass must implement this method!")
    
    def run_one_eval(self, train_x, train_y, valid_x, valid_y, epochs, config):
        raise NotImplementedError("Subclass must implement run_one_eval()")

    def get_model(self, dimension, vocab_size, num_outputs, params):
        raise NotImplementedError("Subclass must implement get_model()")

    def read_training_instances(self, filename):
        raise NotImplementedError("Subclass should implement this to turn a line of training data into a vector, label tuple for training.")
    
    def read_test_instance(self, line):
        raise NotImplementedError("Subclass should implement this to turn a line of test data into a vector for classification.")

    ## This might be general to all models?
    def train_model_for_data(self, train_x, train_y, epochs, config, valid=0.1, use_class_weights=True, checkpoint_prefix=None, early_stopping=False):
        vocab_size = train_x.max() + 1
        class_weights = {}
        num_outputs = 0
        callbacks = [] # [ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)]
        if early_stopping:
            callbacks.append(get_early_stopper())

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
            callbacks.append(get_checkpointer(checkpoint_prefix))

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
        fn = open(os.path.join(working_dir, 'model.pkl'), 'w')
        pickle.dump(self, fn)
        fn.close()

        with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
            myzip.write(os.path.join(working_dir, 'model_weights.h5'), 'model_weights.h5')
            myzip.write(os.path.join(working_dir, 'model.pkl'), 'model.pkl')
    
    def get_default_optimizer(self):
        return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    def get_default_regularizer(self):
        return l2(0.1)

    def param_or_default(self, dict, param, default):
        if param in dict:
            print("Found param item %s in config" % (param))
            return dict[param]
        else:
            return default
        
def read_model(working_dir):
    with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
        myzip.extract('model_weights.h5', working_dir)
        myzip.extract('model.pkl', working_dir)

    model = pickle.load( open(os.path.join(working_dir, 'model.pkl'), 'rb' ) )
    model.keras_model = load_model(os.path.join(working_dir, "model_weights.h5"))
    model.label_lookup = {val:key for (key,val) in model.label_alphabet.iteritems()}
    return model
