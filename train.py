import torch
import os
import numpy as np
from model.model import MalyLLM
from config import SmallTransformerConfig as config
from transformers import AutoTokenizer

data_path = "data/numpy_data"

#Load Data
def load_tokens(File):
    tokens = np.load(File)
    tokens = tokens.astype(np.int32)
    t_token = torch.tensor(tokens , dtype=torch.long)
    return t_token
class DataLoader:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        
        
        shards = os.listdir(data_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) 
        y = (buf[1:]).view(B, T) 
        
        self.current_position += B * T * self.num_processes
        
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get("RANK" , -1)) != -1



if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="ncll")
    ddp_rank = int(os.environ("RANK"))
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True 

    device = "cuda" if torch.cuda.is_available() else "cpu"

#for cuda not cuda:0 , cuda:1 ...
device_type = "cuda" if device.startswith("cuda") else "cpu"

enc = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

model = MalyLLM(config=config())

total_batch_size =524_288 #2**19  ~~0.5 mil

micro_batch = 16

