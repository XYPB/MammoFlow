"""
Balanced Batch Sampler for Imbalanced Binary Classification

This module provides a custom sampler that ensures each mini-batch contains
a specified minimum number of samples from the minority class. This helps
prevent gradient spikes that can occur when batches contain only negative
(majority class) samples in highly imbalanced datasets.

Key Features:
- Controls the ratio of positive/negative samples per batch
- Prevents batches with only majority class samples
- Supports binary classification scenarios
- Compatible with PyTorch DataLoader
"""

import torch
import numpy as np
from torch.utils.data import Sampler, Dataset
from typing import Iterator, List, Optional
from tqdm import tqdm


class BalancedBatchSampler(Sampler[List[int]]):
    """
    A batch sampler that ensures each batch contains a minimum number of 
    minority class samples for binary classification tasks.
    
    This is particularly useful for highly imbalanced datasets (e.g., cancer detection)
    where the minority class (positive/cancer) is extremely rare and random sampling
    often produces batches with only majority class samples.
    
    Args:
        dataset: PyTorch dataset to sample from
        batch_size: Number of samples per batch
        min_positive_per_batch: Minimum number of minority class samples per batch.
                               If there aren't enough minority samples remaining,
                               the sampler will use as many as available.
        drop_last: If True, drop the last incomplete batch
        num_classes: Number of classes (default: 2 for binary classification)
        
    Example:
        >>> sampler = BalancedBatchSampler(
        ...     dataset=train_dataset,
        ...     batch_size=32,
        ...     min_positive_per_batch=4,
        ...     drop_last=True
        ... )
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        min_positive_per_batch: int = 1,
        drop_last: bool = False,
        num_classes: int = 2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_positive_per_batch = min_positive_per_batch
        self.drop_last = drop_last
        self.num_classes = num_classes
        
        # Pre-compute indices for each class
        self.positive_indices = []
        self.negative_indices = []
        
        print(f"Building balanced sampler index (batch_size={batch_size}, "
              f"min_positive_per_batch={min_positive_per_batch})...")
        
        self._build_class_indices()
        
        # Validate configuration
        if self.min_positive_per_batch >= self.batch_size:
            raise ValueError(
                f"min_positive_per_batch ({self.min_positive_per_batch}) must be "
                f"less than batch_size ({self.batch_size})"
            )
        
        if len(self.positive_indices) == 0:
            raise ValueError("No positive samples found in dataset!")
        
        print(f"Balanced sampler initialized:")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Positive samples: {len(self.positive_indices)} "
              f"({100*len(self.positive_indices)/len(dataset):.2f}%)")
        print(f"  - Negative samples: {len(self.negative_indices)} "
              f"({100*len(self.negative_indices)/len(dataset):.2f}%)")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Min positive per batch: {self.min_positive_per_batch}")
    
    def _build_class_indices(self):
        """
        Pre-compute indices for positive and negative classes.
        This allows efficient sampling during training.
        """
        # Sample a subset first to determine label format
        sample = self.dataset[0]
        
        # Determine how to extract labels based on dataset format
        if isinstance(sample, dict):
            label_key = 'label'
        else:
            label_key = None  # Assume (image, label) tuple format
        
        for idx in tqdm(range(len(self.dataset)), desc="Building class indices"):
            try:
                try:
                    sample = self.dataset.__getitem__(idx, no_image=True)
                except TypeError:
                    print(f"Dataset does not support no_image=True. Falling back to standard __getitem__ for class counting.")
                    sample = self.dataset.__getitem__(idx)
                
                # Extract label based on format
                if label_key is not None:
                    label = sample[label_key]
                else:
                    label = sample[1]
                
                # Convert label to class index
                if torch.is_tensor(label):
                    # Handle one-hot encoded labels
                    if len(label.shape) > 0 and label.shape[0] == self.num_classes:
                        class_idx = torch.argmax(label).item()
                    else:
                        class_idx = int(label.item())
                elif isinstance(label, (int, np.integer)):
                    class_idx = int(label)
                elif isinstance(label, float):
                    class_idx = int(label)
                else:
                    # Try to convert to int
                    class_idx = int(label)
                
                # Binary classification: class 1 is positive (minority), class 0 is negative
                if class_idx == 1:
                    self.positive_indices.append(idx)
                else:
                    self.negative_indices.append(idx)
                    
            except Exception as e:
                # Skip problematic samples
                print(f"Warning: Could not process sample {idx}: {e}")
                continue
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches with guaranteed minimum positive samples.
        
        Each batch is constructed by:
        1. Sampling min_positive_per_batch indices from positive class
        2. Filling the rest with negative class samples
        3. Shuffling the batch to randomize order
        """
        # Create shuffled copies of indices for this epoch
        positive_indices = np.array(self.positive_indices.copy())
        negative_indices = np.array(self.negative_indices.copy())
        
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        
        # Convert to lists for popping
        positive_pool = list(positive_indices)
        negative_pool = list(negative_indices)
        
        # Calculate how many complete batches we can make
        # Each batch needs min_positive_per_batch positives
        # and (batch_size - min_positive_per_batch) negatives
        num_negatives_per_batch = self.batch_size - self.min_positive_per_batch
        
        batches = []
        
        while True:
            # Check if we have enough samples for a complete batch
            has_enough_positives = len(positive_pool) >= self.min_positive_per_batch
            has_enough_negatives = len(negative_pool) >= num_negatives_per_batch
            
            if not has_enough_positives or not has_enough_negatives:
                # Handle last incomplete batch
                if not self.drop_last:
                    # Create a partial batch with remaining samples
                    remaining = []
                    
                    # Add remaining positives (up to min_positive_per_batch)
                    num_pos = min(len(positive_pool), self.min_positive_per_batch)
                    for _ in range(num_pos):
                        remaining.append(positive_pool.pop())
                    
                    # Fill with negatives
                    while len(remaining) < self.batch_size and len(negative_pool) > 0:
                        remaining.append(negative_pool.pop())
                    
                    # Add any remaining positives if we still have space
                    while len(remaining) < self.batch_size and len(positive_pool) > 0:
                        remaining.append(positive_pool.pop())
                    
                    if len(remaining) > 0:
                        np.random.shuffle(remaining)
                        batches.append(remaining)
                
                break
            
            # Create a balanced batch
            batch = []
            
            # Sample positive samples
            for _ in range(self.min_positive_per_batch):
                batch.append(positive_pool.pop())
            
            # Fill rest with negative samples
            for _ in range(num_negatives_per_batch):
                batch.append(negative_pool.pop())
            
            # Shuffle batch to randomize order within batch
            np.random.shuffle(batch)
            batches.append(batch)
        
        # Shuffle batch order
        np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        num_negatives_per_batch = self.batch_size - self.min_positive_per_batch
        
        # Number of batches is limited by the class with fewer "batch portions"
        max_batches_from_positives = len(self.positive_indices) // self.min_positive_per_batch
        max_batches_from_negatives = len(self.negative_indices) // num_negatives_per_batch
        
        num_complete_batches = min(max_batches_from_positives, max_batches_from_negatives)
        
        if not self.drop_last:
            # Account for potential partial batch
            remaining_positives = len(self.positive_indices) - (num_complete_batches * self.min_positive_per_batch)
            remaining_negatives = len(self.negative_indices) - (num_complete_batches * num_negatives_per_batch)
            
            if remaining_positives > 0 or remaining_negatives > 0:
                num_complete_batches += 1
        
        return num_complete_batches


class BalancedRandomSampler(Sampler[int]):
    """
    A random sampler that oversamples the minority class to achieve better balance.
    
    Unlike BalancedBatchSampler which controls batch composition directly,
    this sampler works at the sample level and can be used with standard
    DataLoader batching.
    
    This sampler oversamples positive (minority) samples so they appear
    more frequently during training, helping the model learn from rare cases.
    
    Args:
        dataset: PyTorch dataset to sample from
        oversample_ratio: How many times to oversample minority class.
                         1.0 means no oversampling, 2.0 means 2x oversampling, etc.
                         If None, automatically balances classes.
        num_classes: Number of classes (default: 2 for binary classification)
        
    Example:
        >>> sampler = BalancedRandomSampler(
        ...     dataset=train_dataset,
        ...     oversample_ratio=None  # Auto-balance
        ... )
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        oversample_ratio: Optional[float] = None,
        num_classes: int = 2,
    ):
        self.dataset = dataset
        self.oversample_ratio = oversample_ratio
        self.num_classes = num_classes
        
        # Build class indices
        self.positive_indices = []
        self.negative_indices = []
        
        print("Building balanced random sampler index...")
        self._build_class_indices()
        
        # Calculate oversampling ratio if not specified
        if self.oversample_ratio is None:
            # Balance classes by oversampling minority to match majority
            if len(self.positive_indices) > 0:
                self.oversample_ratio = len(self.negative_indices) / len(self.positive_indices)
            else:
                self.oversample_ratio = 1.0
        
        # Calculate effective number of samples per epoch
        self.num_positive_samples = int(len(self.positive_indices) * self.oversample_ratio)
        self.num_samples = self.num_positive_samples + len(self.negative_indices)
        
        print(f"Balanced random sampler initialized:")
        print(f"  - Original positive samples: {len(self.positive_indices)}")
        print(f"  - Original negative samples: {len(self.negative_indices)}")
        print(f"  - Oversample ratio: {self.oversample_ratio:.2f}x")
        print(f"  - Effective positive samples per epoch: {self.num_positive_samples}")
        print(f"  - Total samples per epoch: {self.num_samples}")
    
    def _build_class_indices(self):
        """Pre-compute indices for positive and negative classes."""
        sample = self.dataset[0]
        
        if isinstance(sample, dict):
            label_key = 'label'
        else:
            label_key = None
        
        for idx in tqdm(range(len(self.dataset)), desc="Building class indices"):
            try:
                try:
                    sample = self.dataset.__getitem__(idx, no_image=True)
                except TypeError:
                    sample = self.dataset.__getitem__(idx)
                
                if label_key is not None:
                    label = sample[label_key]
                else:
                    label = sample[1]
                
                if torch.is_tensor(label):
                    if len(label.shape) > 0 and label.shape[0] == self.num_classes:
                        class_idx = torch.argmax(label).item()
                    else:
                        class_idx = int(label.item())
                elif isinstance(label, (int, np.integer)):
                    class_idx = int(label)
                elif isinstance(label, float):
                    class_idx = int(label)
                else:
                    class_idx = int(label)
                
                if class_idx == 1:
                    self.positive_indices.append(idx)
                else:
                    self.negative_indices.append(idx)
                    
            except Exception as e:
                continue
    
    def __iter__(self) -> Iterator[int]:
        """Generate sample indices with oversampled minority class."""
        # Sample positive indices with replacement (oversampling)
        positive_samples = np.random.choice(
            self.positive_indices,
            size=self.num_positive_samples,
            replace=True
        ).tolist()
        
        # Include all negative samples
        negative_samples = self.negative_indices.copy()
        
        # Combine and shuffle
        all_indices = positive_samples + negative_samples
        np.random.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        return self.num_samples


def create_balanced_sampler(
    dataset: Dataset,
    batch_size: int,
    min_positive_per_batch: int,
    drop_last: bool = False,
    num_classes: int = 2,
    sampler_type: str = "batch"
) -> Sampler:
    """
    Factory function to create a balanced sampler.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for training
        min_positive_per_batch: Minimum positive samples per batch
                               (for batch sampler) or oversample ratio indicator
        drop_last: Whether to drop last incomplete batch
        num_classes: Number of classes
        sampler_type: "batch" for BalancedBatchSampler, "random" for BalancedRandomSampler
        
    Returns:
        Configured sampler object
    """
    if sampler_type == "batch":
        return BalancedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            min_positive_per_batch=min_positive_per_batch,
            drop_last=drop_last,
            num_classes=num_classes,
        )
    elif sampler_type == "random":
        return BalancedRandomSampler(
            dataset=dataset,
            oversample_ratio=None,  # Auto-balance
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
