#!/usr/bin/env python

from ctakesneural.models.nn_models import OptimizableModel
from ctakesneural.io import cleartk_io as ctk_io

import numpy as np

class EntityModel(OptimizableModel):

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

    def classify_line(self, line):
        feat_seq = ctk_io.string_to_feature_sequence2(line.split(), self.feats_alphabet, read_only=True)
        ctk_io.fix_instance_len( feat_seq , self.get_standard_input_len())
        feats = [feat_seq]
        outcomes = []
        out = self.keras_model.predict( np.array(feats), batch_size=1, verbose=0)
        if len(out[0]) == 1:
            pred_class = 1 if out[0][0] > 0.5 else 0
        else:
            pred_class = out[0].argmax()
         
        return self.label_lookup[pred_class]

    def get_standard_input_len(self):
        return self.keras_model.input_shape[1]
