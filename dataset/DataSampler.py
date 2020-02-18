import torch
import random
import numpy as np
import os 

NUM_GRAPHEME_CLASS = 1295
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices


        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        df = dataset.df.reset_index()
        # print(len(dataset), len(df))
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = df.iloc[idx]['grapheme']
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[df.loc[idx, 'grapheme']]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    
############################################ Define DataSampler
# see trorch/utils/data/sampler.py
class BalanceSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        
        self.length = len(dataset)

        df = dataset.df.reset_index()

        group = []
        
        # we encode to num already
        grapheme_gb = df.groupby(['grapheme'])
        
        for i in range(NUM_GRAPHEME_CLASS):
            
            g = grapheme_gb.get_group(i).index
            group.append(list(g))
            
            assert(len(g)>0)

        self.group=group

    def __iter__(self):
        
        index = []
        n = 0

        is_loop = True
        while is_loop:
            num_class = NUM_GRAPHEME_CLASS #1295
            c = np.arange(num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n+=1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)