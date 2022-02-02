import io

import h5py
import numpy as np
from PIL import Image
import io

import torch
from torch.utils import data
import glob

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


class MSLSImageDataset(torch.utils.data.Dataset):
    """HDF5 Image dataset"""

    def __init__(self, root, recursive=True, transforms=None):
        super().__init__()
        self.dataset_metadata = []
        self.transforms = transforms or (lambda x: x)
        self.local_cache = []  # Lazy Cache Management for Now...

        # Seed the Dataset - recursive get all h5 files in the dir
        for fp in glob.iglob(f"{root}/**/*.h5", recursive=recursive):
            self._cache_file_w_metadata(fp)

    def __getitem__(self, index):
        im = Image.open(io.BytesIO(self._get_data(index)))
        return self.transforms(im), torch.tensor(0.0)

    def __len__(self):
        return self.dataset_metadata.__len__()

    def _cache_file_w_metadata(self, fp):
        """
        Iterate over all datasets, group...
        """
        with h5py.File(fp) as h5_file:
            for gname, group in h5_file.items():
                for ds in group:

                    # Add File Data
                    self.local_cache.append(ds)

                    # Add Metadata
                    self.dataset_metadata.append(
                        {
                            "file_path": fp,
                            "shape": ds.shape,
                        }
                    )

    def _get_data(self, index):
        """Fetch from Cache or Disk"""
        return self.local_cache.__getitem__(index)
