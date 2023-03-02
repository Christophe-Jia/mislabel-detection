from collections import namedtuple
import tqdm
import os
import torch
import tqdm
import json
import numpy as np
import pandas as pd

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Welford(object):
    """
    Computes and stores a running average and variance
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._count = 0
        self._mean = None
        self._sum_sq = None

    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, new_value, batch=True):
        if isinstance(new_value, torch.autograd.Variable):
            new_value = new_value.data
        if not batch:
            new_value = new_value.unsqueeze(0)

        self._mean = new_value.new(
            *list(new_value.size())[1:]).zero_() if self._mean is None else self._mean
        self._sum_sq = new_value.new(
            *list(new_value.size())[1:]).zero_() if self._sum_sq is None else self._sum_sq

        for item in new_value:
            self._count += 1
            delta = item - self._mean
            self._mean += (item - self._mean) / float(self._count)
            self._sum_sq += delta * (item - self._mean)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._sum_sq / (self._count - 1)

    @property
    def std(self):
        return self.var.sqrt()


def result_class(fields):
    class Result(namedtuple('Result', fields)):
        def items(self):
            for field in self._fields:
                yield (field, getattr(self, field))

        def to_str(self):
            return ",".join(str(item) for item in self)

        def __repr__(self):
            res = 'Results:\n'
            fieldstrs = []
            for key in self._fields:
                fieldstrs.append('  - %s: %s' % (key, repr(getattr(self, key))))
            res = res + '\n'.join(fieldstrs)
            return res

    return Result


def output_class(fields):
    class Output(namedtuple('Output', fields)):
        def __repr__(self):
            res = 'Outputs:\n'
            fieldstrs = []
            for key in self._fields:
                fieldstrs.append('  - %s: %s' % (key, repr(getattr(self, key).size())))
            res = res + '\n'.join(fieldstrs)
            return res

    return Output


def softmax_with_temperature(x,T):
    x_exp = torch.exp(x/T)
    return x_exp / torch.sum(x_exp)
            
    
# drop last 遗留问题
def transfor_and_save(savedir, probas, assigned_targets):
    """Transform training dynamics with linear interpolation, then save in 'npz' format.
    Args:
        probas (dict): A dictionary recording training dynamics.
        assigned_targets (list): The assigned targets of dataset.
    """
    probas = [(k, probas[k]) for k in sorted(probas.keys())]
    probas = np.asarray([probas[i][1] for i in range(len(probas))])

    # Linear interpolation of given probas, to fix the probas broken by drop_last
    # for logit in probas:
    #     bad_indexes = np.isnan(logit)
    #     good_indexes = np.logical_not(bad_indexes)
    #     interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], logit[good_indexes])
    #     logit[bad_indexes] = interpolated
    probas = probas.astype(np.float16)

    targets_list = np.argsort(-probas.mean(axis=1), axis=1)
    training_dynamics = np.ones_like(probas,dtype=np.float16)
    labels = np.ones_like(probas[:,0,:],dtype=np.int16)

    for index,targets in enumerate((targets_list)):
        # save ground turth td
        labels[index,0] = assigned_targets[index]
        training_dynamics[index,:,0] = probas[index,:,assigned_targets[index]].tolist()
        # save topk td
        top_i=1
        for target in targets:
            if target != assigned_targets[index]:
                labels[index,top_i] = target
                training_dynamics[index,:,top_i] = probas[index,:,target].tolist()
                top_i+=1

    np.savez_compressed(os.path.join(savedir, 'training_dynamics.npz'), **{'labels': labels, 'td': training_dynamics})