import torch


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
