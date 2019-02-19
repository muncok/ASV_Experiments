import math
import random
import itertools
import torch
import numpy as np

from torch.utils.data.dataloader import DataLoader

def index_dataset(dataset):
    return {c : [example_idx for example_idx, (_, class_label_ind) in \
                 enumerate(zip(dataset.embeds, dataset.labels)) if class_label_ind == c] for c in set(dataset.labels)}

def get_sample_dict(df):
    samples_by_class = df.groupby('label').apply(lambda x: x.index).to_dict()
    return samples_by_class

def sample_from_class(samples_by_class, class_label_ind):
    return samples_by_class[class_label_ind][random.randrange(len(samples_by_class[class_label_ind]))]

def simple(batch_size, dataset, prob_other = 0.5):
    '''lazy sampling, not like in lifted_struct. they add to the pool all postiive combinations, then compute the average number of positive pairs per image, then sample for every image the same number of negative pairs'''
    samples_by_class = index_dataset(dataset)
    for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
        example_indices = []
        for i in range(0, batch_size, 2):
            perm = random.sample(samples_by_class.keys(), 2)
            example_indices += [sample_from_class(samples_by_class, perm[0]), sample_from_class(samples_by_class, perm[0 if i == 0 or random.random() > prob_other else 1])]
        yield example_indices[:batch_size]

def triplet(batch_size, dataset, samples_by_class):
    for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
        example_indices = []
        for i in range(0, batch_size, 3):
            perm = random.sample(samples_by_class.keys(), 2)
            example_indices += [sample_from_class(samples_by_class, perm[0]), sample_from_class(samples_by_class, perm[0]), sample_from_class(samples_by_class, perm[1])]
        yield example_indices[:batch_size]

def npairs(batch_size, dataset, K = 4):
    samples_by_class = index_dataset(dataset)
    for batch_idx in range(int(math.ceil(len(dataset) * 1.0 / batch_size))):
        example_indices = [sample_from_class(samples_by_class, class_label_ind) for k in range(int(math.ceil(batch_size * 1.0 / K))) for class_label_ind in [random.choice(samples_by_class.keys())] for i in range(K)]
        yield example_indices[:batch_size]
        


def get_metric_dataloader(dataset, sampler, batch=64):
    gen_sampler = lambda batch, dataset, sampler, **kwargs: \
    type('', (torch.utils.data.sampler.Sampler,), 
         dict(__len__ = dataset.__len__, __iter__ = \
              lambda _: itertools.chain.from_iterable(sampler(batch, dataset, **kwargs))))(dataset)
    si_loader = torch.utils.data.DataLoader(
        si_dataset, 
        sampler = adapt_sampler(batch, 
                               si_dataset, 
                               triplet, 
                               class2img=class2idx), 
        num_workers = 8, batch = batch, 
        drop_last = True, pin_memory = True
    )

    sv_loader = DataLoader(sv_dataset, batch_size=128, num_workers=4, shuffle=False)
    
def hard_mining(anchor, pos_egs, neg_egs, margin=1.0):
    pos_dist = (anchor - pos_egs).pow(2).sum(1)
    pos_dist = torch.clamp(pos_dist, min=1e-16)
    pos_dist = pos_dist.sqrt()
    
    neg_dist = (anchor - neg_egs).pow(2).sum(1)
    neg_dist = torch.clamp(neg_dist, min=1e-16)
    neg_dist = neg_dist.sqrt()
    
    
    hard_pos_dist = pos_dist.max()
    hard_neg_dist = neg_dist.min()
    
#     print(f"hard_pos:{hard_pos_dist}, hard_neg:{hard_neg_dist}")
    
    triplet_loss = torch.clamp(hard_pos_dist - hard_neg_dist + margin, min=0)
    triplet_loss = torch.sum(triplet_loss)
    
    return triplet_loss

def class_weight(config, train_df):
    class_ratios = train_df.label.value_counts().sort_index()
    class_inv = 1 / class_ratios
    max_inv = np.max(class_inv)
    min_inv = np.min(class_inv)
    class_weights = ((class_inv - min_inv) / max_inv).values
    class_weights = torch.from_numpy(class_weights).float()
    if not config['no_cuda']:
        class_weights = class_weights.cuda()
    
    return class_weights