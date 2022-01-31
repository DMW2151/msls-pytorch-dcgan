import torch

try:
    from habana_frameworks.torch.core import mark_step
except ImportError:

    def mark_step():
        pass


class LimitDataset(torch.utils.data.Dataset):
    """
    Simple wrapper around torch.utils.data.Dataset to limit # of data-points passed
    to a DataLoader
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
