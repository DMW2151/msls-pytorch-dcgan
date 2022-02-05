import torch

try:
    from habana_frameworks.torch.core import mark_step
except ImportError:

    def mark_step():
        pass


class LimitDataset(torch.utils.data.Dataset):
    """
    Simple wrapper around torch.utils.data.Dataset to limit # of data-points
    passed to a DataLoader
    """

    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        """Clobber the old Length"""
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


class MarkHTStep(object):
    """Context manager for marking optimizer steps for SynapseAI"""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        if self.state:
            mark_step()

    def __exit__(self, type, value, traceback):
        if self.state:
            mark_step()


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)