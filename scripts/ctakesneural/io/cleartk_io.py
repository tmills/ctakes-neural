#!/usr/bin/env python3

import numpy as np
import os, os.path, sys
import subprocess


_UNK_STRING = "_UNK_OR_PADDING_"

def string_label_to_label_vector(label_string, outcome_maps):    
    label_vec = []
    
    all_labels = label_string.split(' ')
    
    if len(all_labels) == 1:
        return [label_string]
    
    for label_val in all_labels:
        (label, val) = label_val.split('=')
        cur_map = outcome_maps[label]
        label_ind = cur_map[val]
        label_vec.append(label_ind)
        
    return label_vec
    
def get_data_dimensions(data_file):
    wc_out = subprocess.check_output(['wc',  data_file])
    wc_fields = wc_out.decode().strip().split(' ')
    file_len = int(wc_fields[0])

    num_feats = 0
    for line in open(data_file):
        max_dim = int( line.rstrip().split(' ')[-1].split(':')[0] )
        if max_dim > num_feats:
            num_feats = max_dim

    return (file_len, num_feats)

def flatten_outputs(Y):

    if Y.ndim == 1:
        Y = np.expand_dims(Y, 1)
        
    maxes = Y.max(0)

    #print("Maxes = %s" % (maxes) )
    reqd_dims = 0
    indices = [0]
    
    ## Create an indices array that maps from "true" label indices to neural network 
    ## output layer indices -- binary labels map to single output nodes (2->1) while n-ary
    ## labels map to n nodes.
    for val in maxes:
        if val == 1:
            reqd_dims += 1
        elif val > 1:
            reqd_dims += (int(val) + 1)
        else:
            raise Exception("There is a column with all zeros!")
            
        indices.append(reqd_dims)

    Y_adj = np.zeros( (Y.shape[0], reqd_dims) )
    for row_ind in range(0, Y.shape[0]):
        for col_ind in range(0, Y.shape[1]):
            if maxes[col_ind] == 1:
                ## For binary variables just need the offset and copy the value
                Y_adj[row_ind][ int(indices[col_ind]) ] = Y[row_ind][col_ind]
            else:
                ## for n-ary variables we use the value to find the offset that will 
                ## be set to 1.
                Y_adj[row_ind][ int(indices[col_ind]) + int(Y[row_ind][col_ind]) ] = 1
    
    return Y_adj, indices

def read_outcome_maps(dirname):
    raw_outcomes = []
    raw_outcomes.append(None)
    
    derived_maps = {}
    lookup_map = {}
    ## First read outcome file
    for line in open(os.path.join(dirname, 'outcome-lookup.txt') ):
        (index, label) = line.rstrip().split(' ')
        raw_outcomes.append(label)
        
        for task_label in label.split(' '):
            #print(task_label)
            (task, val) = task_label.rstrip().split("=")
            if not task in derived_maps:
                derived_maps[task] = {}
                lookup_map[task] = []
                
            cur_map = derived_maps[task]
            lookup = lookup_map[task]
            if not val in cur_map:
                cur_map[val] = len(cur_map)
                lookup.append(val)
    
    return raw_outcomes, derived_maps, lookup_map

def outcome_list(raw_outcomes):
    outcomes = []
    for outcome_val in raw_outcomes[1].split("#"):
        outcomes.append(outcome_val.split("=")[0])
    
    return outcomes
    
def read_multitask_liblinear(dirname):
    
    raw_outcomes, derived_maps, outcome_lookups = read_outcome_maps(dirname)
        
    data_file = os.path.join(dirname, 'training-data.libsvm')
    
    (data_points, feat_dims) = get_data_dimensions(data_file)
    
    label_dims = len(derived_maps)
    
    label_matrix = np.zeros( (data_points, label_dims) )
    feat_matrix = np.zeros( (data_points, feat_dims) )
    
    line_ind = 0
    for line in open( data_file ):
        label_and_feats = line.rstrip().split(' ')
        label = label_and_feats[0]
        string_label = raw_outcomes[int(label)]
        label_vec = string_label_to_label_vector(string_label, derived_maps)
        
        for ind, val in enumerate(label_vec):
            label_matrix[line_ind, ind] = val
    
        ## Go from 1 on -- skip the label
        ## the bias term from the liblinear data writer.
        feat_matrix[line_ind, :] = feature_array_to_list( label_and_feats[1:], feat_dims )            
                
        line_ind += 1

    return label_matrix, feat_matrix

def read_liblinear(dirname):
    data_file = os.path.join(dirname, 'training-data.libsvm')
    
    (data_points, feat_dims) = get_data_dimensions(data_file)
    
    label_array = np.zeros( (data_points, 1) )
    feat_matrix = np.zeros( (data_points, feat_dims) )

    line_ind = 0
    for line in open( data_file ):
        label_and_feats = line.rstrip().split(' ')
        label = label_and_feats[0]

        label_array[line_ind] = float(label) - 1

        ## Go from 1 on -- skip the label
        ## the bias term from the liblinear data writer.
        feat_matrix[line_ind, :] = feature_array_to_list( label_and_feats[1:], feat_dims )            
                
        line_ind += 1

    label_matrix = np.zeros( (data_points, label_array.max()+1) )
    
    for ind,val in np.ndenumerate(label_array):
        label_matrix[ind,val] = 1

    return label_matrix, feat_matrix
    
def convert_multi_output_to_string(outcomes, outcome_list, lookup_map):
    """Return the int value corresponding to the class implied by the
    set of outputs in the outcomes array."""
    str = ''
    for ind, label in enumerate(outcome_list):
        str += label
        str += "="
        str += lookup_map[label][outcomes[ind]]
        str += " "
        
    str = str[:-1]
    return str

def get_outcome_array(working_dir):
    labels = []
    
    for line in open(working_dir, "outcome-lookup.txt"):
       (ind, val) = line.rstrip().split(" ")
       labels.append(val)
    
    return labels     

def feature_string_to_list( feat_string, length=-1 ):
    return feature_array_to_list( feat_string.split(' '), length )

def feature_array_to_list( feats, length=-1 ):
    if length == -1:
        length = len(feats)
        
    #f = np.zeros(length)
    f = [0] * length
    
    for feat in feats:
        (ind, val) = feat.split(':')
        ind = int(ind) - 1
        if int(ind) >= len(f):
            raise Exception("Feature index %d is larger than feature vector length %d -- you may need to specify the expected length of the vector." % (int(ind), len(f) ) )
        f[int(ind)] = val
    
    return f
    
def read_token_sequence_data(working_dir):
    feature_alphabet = {}
    label_alphabet = {}
    label_seq = []
    instance_seq = []
    
    for line in open(os.path.join(working_dir, 'training-data.libsvm')):
        (label, token_str) = split_sequence_line_singleLabel(line.strip())
        
        if not label in label_alphabet:
            label_alphabet[label] = len(label_alphabet)
        
        label_ind = label_alphabet[label]
        ## So that numpy turns it into a (n,1) 2d array instead of a (n,) 1d array
        label_seq.append(label_ind)
        
        token_ind_seq = string_to_feature_sequence2(token_str, feature_alphabet)
        instance_seq.append(token_ind_seq)
    
    ## If we don't pad the shorter instances numpy conversion won't be able to turn it into a 2d array 
    pad_instances(instance_seq)
    
    return label_seq, label_alphabet, np.array(instance_seq), feature_alphabet

def read_token_and_pos_sequence_data(working_dir):
    feature_alphabet = {}
    label_alphabet = {}
    pos_alphabet = {}
    label_seq = []
    instance_seq = []
    pos_seq = []
    
    for line in open(os.path.join(working_dir, 'training-data.libsvm')):
        (label, token_str) = split_sequence_line_singleLabel(line.strip())
        
        if not label in label_alphabet:
            label_alphabet[label] = len(label_alphabet)
        
        label_ind = label_alphabet[label]
        ## So that numpy turns it into a (n,1) 2d array instead of a (n,) 1d array
        label_seq.append(label_ind)
        
        (token_ind_seq, pos_ind_seq) = string_to_feature_sequence(token_str, feature_alphabet, pos_alphabet)
        instance_seq.append(token_ind_seq)
        pos_seq.append(pos_ind_seq)
    
    ## If we don't pad the shorter instances numpy conversion won't be able to turn it into a 2d array 
    pad_instances(instance_seq)
    pad_instances(pos_seq)
    
    
    return label_seq, label_alphabet, np.array(instance_seq), feature_alphabet, np.array(pos_seq), pos_alphabet
        
def read_multitask_token_sequence_data(working_dir):
    feature_alphabet = {_UNK_STRING:0}
    instance_seq = []
    label_seq = []
    outcome_maps = {}
    outcome_list = []
    
    for line in open(os.path.join(working_dir, 'training-data.libsvm')):
        (label_set, token_seq) = split_sequence_line(line.strip())
        instance_labels = []
        
        for outcome in label_set:
            (outcome_type, value) = outcome.split("=")
            if not outcome_type in outcome_maps:
                outcome_maps[outcome_type] = {}
                outcome_list.append(outcome_type)
            
            outcome_map = outcome_maps[outcome_type]
            if value not in outcome_map:
                outcome_map[value] = len(outcome_map)
            
            instance_labels.append( outcome_map[value] )
        
        label_seq.append(instance_labels)
        instance_seq.append(list_to_feature_sequence(token_seq, feature_alphabet))
    
    pad_instances(instance_seq)
    
    return np.array(label_seq), outcome_maps, outcome_list, np.array(instance_seq), feature_alphabet

def pad_instances(instance_seq):    
    max_len = max(map(len, instance_seq))
    
    for inst in instance_seq:
        fix_instance_len(inst, max_len)

def fix_instance_len(inst, req_len):
    if len(inst) < req_len:
        while len(inst) < req_len:
            inst.append(0)
    elif len(inst) > req_len:
        ## Instance is too long -- can happen at test time -- truncate to the end of the sequence
        while len(inst) > req_len:
            inst.pop(0)
    
def split_sequence_line(line):
    sections = line.split(' | ')
    if len(sections) == 2:
        (label_str, feats) = sections
    else:
        label_str = sections[0]
        feats = ' | '.join(sections[1:])

    return label_str.strip().split(' '), feats.strip().split(' ')

def split_sequence_line_singleLabel(line):
    sections = line.split(' | ')
    if len(sections) == 2:
        (label_str, feats) = sections
    else:
        label_str = sections[0]
        feats = ' | '.join(sections[1:])

    return label_str.strip(), feats.strip().split(' ')

def string_to_feature_sequence2(token_str, alphabet, read_only=False):
    token_seq = token_str
    return list_to_feature_sequence(token_seq, alphabet, read_only)
    
def string_to_feature_sequence(token_str, token_alphabet, pos_alphabet=None, read_only=False):
    token_ind_seq = []
    pos_ind_seq = []
    
    for ind,str in enumerate(token_str):
        if ("|" in str and len(str)>1 ):
            token,pos = str.split("|")
        else:
            token = str
            pos = None
            
        if not token in token_alphabet:
            if not read_only:
                token_alphabet[token]=len(token_alphabet)
                token_ind_seq.append(token_alphabet[token])
            else:
                token_ind_seq.append(0)
        else:
            token_ind_seq.append(token_alphabet[token])
            
        if pos is not None and pos_alphabet is not None:
            if not pos in pos_alphabet:
                if not read_only:
                    pos_alphabet[pos]=len(pos_alphabet)
                    pos_ind_seq.append(pos_alphabet[pos])
                else:
                    pos_ind_seq.append(0)
            else:
                pos_ind_seq.append(pos_alphabet[pos])
                
    return token_ind_seq, pos_ind_seq

def list_to_feature_sequence(token_seq, alphabet, read_only=False):
    token_ind_seq = [] # np.zeros( len(token_seq), dtype=np.int )
    
    for ind,token in enumerate(token_seq):
            if not token in alphabet:
                if not read_only:
                    alphabet[token] = len(alphabet)
                    token_ind_seq.append(alphabet[token])
                else:
                    token_ind_seq.append(0)
            else:
                token_ind_seq.append(alphabet[token])
                
    return token_ind_seq

def reverse_outcome_maps(outcome_maps):
    rev = {}
    for label in outcome_maps.keys():
        rev[label] = {}
        ## Why is enumerate returning val, key instead of key, val? 
        for key,val in outcome_maps[label].iteritems():
            rev[label][val] = key
    
    return rev

## Read a line of the sequence file and map labels and features to ints
def read_bio_sequence_data(working_dir):
    feat_alphabet = {"UNK":0}
    label_alphabet = {"O":0}  ## May have other tags but wil always have O and want that to be = to padded label
    labels = []
    feats = []
    
    for line in open(os.path.join(working_dir, 'training-data.libsvm')):
        (str_labels, str_feats) = split_sequence_line(line.strip())

        cur_labels = []
        for label in str_labels:
            if not label in label_alphabet:
                label_alphabet[label] = len(label_alphabet)
        cur_labels = [label_alphabet[label] for label in str_labels]

        cur_feats = []
        for feat in str_feats:
            cur_feats.append( read_bio_feats_with_alphabet(feat, feat_alphabet, read_only=False) )
            if not feat in feat_alphabet:
                feat_alphabet[feat] = len(feat_alphabet)      
        
        labels.append(cur_labels)
        feats.append(np.array(cur_feats))
        
    return labels, label_alphabet, feats, feat_alphabet

def expand_labels(label_matrix, label_alphabet):
    ''' If we're given a matrix of labels for sequence instances, with instances in the rows and
        time steps in the columns, convert into the representation where each time step has
        boolean variables: from N x M to N x D x M where D is the number of possible labels
        '''
    padded_labels = np.zeros((label_matrix.shape[0], label_matrix.shape[1], len(label_alphabet)))
    for instance_ind in range(label_matrix.shape[0]):
        instance = label_matrix[instance_ind]
        padded_instance = []
        for token_ind,output in enumerate(instance):
            padded_labels[instance_ind, token_ind, output] = 1
#            label_nodes = [0] * len(label_alphabet)
#            label_nodes[output] = 1
#            padded_instance.append(label_nodes)
#        padded_labels.append(np.array(padded_instance))
    return padded_labels
    
def read_bio_feats_with_alphabet(feat_string, feat_alphabet, read_only=True):
    feats = []
    
    feat = feat_string.split()[0]
    if not feat in feat_alphabet:
        if not read_only:
            feat_alphabet[feat] = len(feat_alphabet)
            val = feat_alphabet[feat]
        else:
            val = 0
    else:
        val = feat_alphabet[feat]
            
    feats.append( val )

    if len(feat_string.split()) > 1:
        label = feat_string.split()[1]
        if label == 'O':
            feats.append( 0 )
        elif label == 'B':
            feats.append( 1 )
        else:
            feats.append( 2 )
            
    return feats[0] if len(feats) == 1 else feats

def read_embeddings(filename, alphabet):
    vocab_size = len(alphabet)
    dims = -1
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if dims < 0:
                vals = line.split()
                #vocab_size = int(vals[0])
                dims = int(vals[1])
                embed_matrix = np.zeros( (vocab_size, dims) )
            else:
                vec_list = line.split()
                if vec_list[0] in alphabet:
                    vec = np.array( [float(element) for element in vec_list[1:]] )
                    embed_matrix[ alphabet[vec_list[0]]] = vec
                    
    return embed_matrix
                
def print_label(label):
    print(label)
    sys.stdout.flush()

def debug(msg):
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()

if __name__ == "__main__":
    (labels, feats) = read_multitask_liblinear('data_testing/multitask_assertion/train_and_test/')
    print("train[0][100] = %f" % feats[0][100])
