import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader
import tiktoken

# Global tokenizer for all models
enc = tiktoken.get_encoding("gpt2")

def load_tokens(filename):
    """Load tokenized data from a file"""
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class ShardedTextDataset(Dataset):
    def __init__(self, shards, batch_size, seq_length, split, process_rank=0, num_processes=1):
        self.shards = shards
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.rng = np.random.default_rng(1337)
        
        # Load first shard
        self.current_shard_idx = 0
        if split == 'train':
            self.rng.shuffle(self.shards)
        self.tokens = self._load_shard(self.shards[self.current_shard_idx])
        self.current_position = batch_size * seq_length * process_rank
        
        # Calculate total samples (approximate)
        self.approx_total_samples = len(self.tokens) // (seq_length + 1) * len(shards)
        
    def _load_shard(self, filename):
        shard = load_tokens(filename)
        if self.split == 'train':
            # Split tokens into documents using the <|endoftext|> special token and shuffle
            eot_positions = (torch.where(shard == enc.eot_token)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents)  # concatenate the documents back together
        return shard
        
    def __len__(self):
        return self.approx_total_samples
    
    def __getitem__(self, idx):
        B, T = self.batch_size, self.seq_length
        
        # Check if we need to load next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.tokens = self._load_shard(self.shards[self.current_shard_idx])
            self.current_position = B * T * self.process_rank
        
        # Get batch
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Update position
        self.current_position += B * T * self.num_processes
        
        return x, y
    
    def get_state(self):
        return {
            'current_shard_idx': self.current_shard_idx,
            'current_position': self.current_position
        }
    
    def set_state(self, state):
        self.current_shard_idx = state['current_shard_idx']
        self.current_position = state['current_position']
        self.tokens = self._load_shard(self.shards[self.current_shard_idx])

class DataLoader:
    """Data Loader for loading tokenized text data"""
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B", master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.master_process = master_process
        
        # Get shards
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.shards = shards
        
        # Create dataset
        self.dataset = ShardedTextDataset(
            shards=shards,
            batch_size=B,
            seq_length=T,
            split=split,
            process_rank=process_rank,
            num_processes=num_processes
        )
        
        # Create PyTorch DataLoader
        self.dataloader = PyTorchDataLoader(
            self.dataset,
            batch_size=None,  # We're already creating batches in the dataset
            num_workers=2,    # Use multiple workers for loading
            pin_memory=True,  # Speeds up host to GPU transfers
            prefetch_factor=2 # How many samples to prefetch per worker
        )
        
        # Create iterator
        self.dataloader_iter = iter(self.dataloader)
        
    def next_batch(self):
        """Get the next batch of data"""
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            # Reset iterator and try again
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)
    
    def reset(self):
        """Reset the loader state"""
        self.dataset.current_shard_idx = 0
        if self.split == 'train':
            self.dataset.rng.shuffle(self.dataset.shards)
        self.dataset.tokens = self.dataset._load_shard(self.dataset.shards[0])
        self.dataset.current_position = self.B * self.T * self.process_rank
        self.dataloader_iter = iter(self.dataloader)
    
    def set(self, loader_checkpoint):
        """Set the loader state from a checkpoint"""
        state = {
            'current_shard_idx': loader_checkpoint['current_shard'],
            'current_position': loader_checkpoint['current_position']
        }
        self.dataset.set_state(state)
        self.dataloader_iter = iter(self.dataloader)
    
    def get_loader_checkpoint(self):
        """Get a checkpoint of the loader state"""
        state = self.dataset.get_state()
        return {
            'current_shard': state['current_shard_idx'],
            'current_position': state['current_position']
        }
    
    def load_shard(self, filename):
        """Load a shard of tokenized data, potentially shuffling documents"""
        return self.dataset._load_shard(filename)