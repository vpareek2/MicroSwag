import os
import numpy as np
import torch
import tiktoken

# Global tokenizer for all models
enc = tiktoken.get_encoding("gpt2")

def load_tokens(filename):
    """Load tokenized data from a file"""
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    """Data Loader for loading tokenized text data"""
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B", master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_root = data_root
        self.master_process = master_process
        assert split in {'train', 'val'}
        self.rng = np.random.default_rng(1337)  # Fixed seed for reproducibility

        # get shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def load_shard(self, filename):
        """Load a shard of tokenized data, potentially shuffling documents"""
        shard = load_tokens(filename)
        if self.split == 'train':
            # split tokens into documents using the <|endoftext|> special token and shuffle
            eot_positions = (torch.where(shard == enc.eot_token)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents)  # concatenate the documents back together
        return shard

    def set(self, loader_checkpoint):
        """Set the loader state from a checkpoint"""
        self.current_position = loader_checkpoint['current_position'] + self.B * self.T * self.process_rank
        self.current_shard = loader_checkpoint['current_shard']
        self.tokens = load_tokens(self.shards[self.current_shard])
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard += 1
            # reshuffle after each epoch
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        """Reset the loader state"""
        # state, init at shard zero
        self.current_shard = 0
        if self.split == 'train':
            self.rng.shuffle(self.shards)
        self.tokens = self.load_shard(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Get the next batch of data"""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)  # inputs
        y = (buf[1:]).view(B,T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard += 1
            # reshuffle after each epoch
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
        return x, y

    def get_loader_checkpoint(self):
        """Get a checkpoint of the loader state"""
        return {
            'current_shard': self.current_shard,
            'current_position': self.current_position
        }
