import bisect
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    """A ConcatDataset that forwards extra *args and **kwargs to sub-dataset __getitem__."""

    def __init__(self, datasets):
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        s = 0
        for d in self.datasets:
            s += len(d)
            self.cumulative_sizes.append(s)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx, *args, **kwargs):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].__getitem__(sample_idx, *args, **kwargs)
