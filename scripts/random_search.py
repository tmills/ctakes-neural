#!/usr/bin/env python
#
# This class wraps Code from: http://people.eecs.berkeley.edu/~kjamieson/hyperband.html
#


class RandomSearch:
    def __init__(self, config_fn, eval_fn):
        self.config_fn = config_fn
        self.eval_fn = eval_fn
    
    def optimize(self, max_iter=81):
        eta = 3 # defines downsampling rate (default=3)
        logeta = lambda x: log(x)/log(eta)
        s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
        B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
        
        
        #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
        for s in reversed(range(s_max+1)):
            n = int(ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
            r = max_iter*eta**(-s) # initial number of iterations to run configurations for

            #### Begin Finite Horizon Successive Halving with (n,r)
            T = [ self.config_fn() for i in range(n) ] 
            for i in range(s+1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = r*eta**(i)
                val_losses = [ self.eval_fn(num_iters=r_i,hyperparameters=t) for t in T ]
                T = [ T[i] for i in argsort(val_losses)[0:int( n_i/eta )] ]

            #### End Finite Horizon Successive Halving with (n,r)