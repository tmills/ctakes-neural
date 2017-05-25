#!/usr/bin/env python
#
# This class wraps Code from: http://people.eecs.berkeley.edu/~kjamieson/hyperband.html
#

import numpy as np
import time
from sklearn.cross_validation import train_test_split

class RandomSearch:
    def __init__(self, model, train_x, train_y):
#        self.config_fn = config_fn
#        self.eval_fn = eval_fn
        self.model = model
        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=18)
        
    def optimize(self, max_iter=81):
        start_time = time.time()
        
        eta = 3 # defines downsampling rate (default=3)
        logeta = lambda x: np.log(x)/np.log(eta)
        s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
        B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
        
        
        #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
        for s in reversed(range(s_max+1)):
            n = int(np.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
            r = max_iter*eta**(-s) # initial number of iterations to run configurations for

            #print("Running s=%d, num configs=%d, num iters=%d" % (s, n, r) )
            
            #### Begin Finite Horizon Successive Halving with (n,r)
            T = [ self.model.get_random_config() for i in range(n) ]
            #print("Starting this halving iteration with %d configs" % ( len(T) ) )
            for i in range(s+1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = int( r*eta**(i) )
                val_losses = [ self.model.run_one_eval(self.train_x, self.train_y, self.valid_x, self.valid_y, r_i, t) for t in T ]
                T = [ T[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
                #print("After iteration %d T has %d configurations" % (s, len(T)))
                
            #### End Finite Horizon Successive Halving with (n,r)
        return T[0]
