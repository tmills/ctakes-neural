#!/usr/bin/env python

class PytorchLstmEntityModel(EntityModel):
    def __init__(self, configs=None):
        if configs is None:
            ## Default is not smart -- single layer with between 50 and 1000 nodes
            self.configs = {}
            self.configs['embed_dim'] = (10,25,50,100,200)
            self.configs['layers'] = ( (50,), (100,), (200,), (500,), (1000,) )
            self.configs['batch_size'] = (32, 64, 128, 256)
        else:
            self.configs = configs

    def get_model(self, dimension, vocab_size, num_outputs, config):
        layers = config['layers']
        
        optimizer = self.param_or_default(config, 'optimizer', self.get_default_optimizer())
        weights = self.param_or_default(config, 'weights', None)
        regularizer = self.param_or_default(config, 'regularizer', self.get_default_regularizer())

    def get_standard_input_len(self):

def main(args):
    if len(args) < 2:
        sys.stderr.write('Two required arguments: <train|classify|optimize> <data directory>\n')
        sys.exit(-1)

    if args[0] == 'train':
        working_dir = args[1]
        model = PytorchLstmEntityModel()
        train_x, train_y = model.read_training_instances(working_dir)
        trained_model, history = model.train_model_for_data(train_x, train_y, 80, model.get_default_config())
        model.write_model(working_dir, trained_model)
        
    elif args[0] == 'classify':
        working_dir = args[1]
        model = read_pytorch_model(working_dir)
     
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
        model =PytorchLstmEntityModel()
        train_x, train_y = model.read_training_instances(working_dir)
        optim = RandomSearch(model, train_x, train_y)
        best_model = optim.optimize()
        print("Best config: %s" % best_model)
    else:
        sys.stderr.write("Do not recognize args[0] command argument: %s\n" % (args[0]))
        sys.exit(-1)
        
if __name__ == "__main__":
    main(sys.argv[1:])