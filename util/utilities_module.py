#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/utilities3.py
"""

import torch
import operator
from functools import reduce
import numpy as np
import scipy.io
import hdf5storage

#################################################
#
# utilities
#
#################################################

def to_torch(x, to_float=True):
    """
    send input numpy array to single precision torch tensor
    """
    if to_float:
        if np.iscomplexobj(x):
            x = x.astype(np.complex64)
        else:
            x = x.astype(np.float32)
    return torch.from_numpy(x)


def validate(f, fhat):
    '''
    Helper function to compute relative L^2 error of approximations.
    Takes care of different array shape interpretations in numpy.

    INPUTS:
            f : array of high-fidelity function values
         fhat : array of approximation values

    OUTPUTS:
        error : float, relative error
    '''
    f, fhat = np.asarray(f).flatten(), np.asarray(fhat).flatten()
    return np.linalg.norm(f-fhat) / np.linalg.norm(f)


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    # Reference: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__,})


class MatReader(object):
    """
    reading data
    """
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True,
                 variable_names=None):
        super(MatReader, self).__init__()

        self.file_path = file_path
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.variable_names = variable_names    # a list of strings (key values in mat file)

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path, variable_names=self.variable_names)
            self.old_mat = True
        except:
            self.data = hdf5storage.loadmat(self.file_path, variable_names=self.variable_names)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if self.to_float:
            if np.iscomplexobj(x):
                x = x.astype(np.complex64)
            else:
                x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class UnitGaussianNormalizer(object):
    """
    normalization, pointwise gaussian
    """
    def __init__(self, x, eps=1e-6):
        super(UnitGaussianNormalizer, self).__init__()

        # x has sample/batch size as first dimension (could be ntrain*n or ntrain*T*n or ntrain*n*T)
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class LpLoss(object):
    """
    loss function with rel/abs Lp norm loss
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True, eps=1e-6):
        super(LpLoss, self).__init__()

        if not (d > 0 and p > 0):
            raise ValueError("Dimension d and Lp-norm type p must be postive.")

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps =eps

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[-1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        y_norms += self.eps     # prevent divide by zero

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class LppLoss(object):
    """
    loss function with rel/abs Lp norm to the p-th power loss
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True, eps=1e-6):
        super(LppLoss, self).__init__()

        if not (d > 0 and p > 0):
            raise ValueError("Dimension d and Lp-norm type p must be postive.")

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = eps

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[-1] - 1.0)

        all_norms = (h**(self.d))*(torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)**(self.p))

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)**(self.p)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)**(self.p)
        y_norms += self.eps     # prevent divide by zero

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def count_params(model):
    """
    print the number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
