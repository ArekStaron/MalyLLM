import torch
import os
import numpy as np
from model.model import MalyLLM
from config import SmallTransformerConfig as config
from transformers import AutoTokenizer
import math
import time

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
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
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
raw_model = model 
model = model.to(device)
model = torch.compile(model)
torch.set_float32_matmul_precision('high')

if ddp:
    model = DDP(model , device_ids=[ddp_local_rank])

total_batch_size =524_288 #2**19  ~~0.5 mil

B = 16
T = 2048 

assert  total_batch_size % ( B * T * ddp_world_size)  ==0
grand_accumulate_steps = total_batch_size //(B*T * ddp_world_size) 

if master_process:
    print(f"total desaird batch size: {total_batch_size}")
    print(f"gradient accumulation steps: { grand_accumulate_steps}")

max_lr = 6e-4
min_lr = max_lr * 0.1   
warmup_steps = 50#715
max_steps = 500 #19073 # 10B // 0.5 mil 


def get_lr(it):

    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    
    if it >max_steps:
        return min_lr
    
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (max_lr - min_lr)
     
optimizer = raw_model.configure_optimizers(weight_decay=0.1 , learning_rate= 6e-4 , device= device_type)

train_loader = DataLoader(B= B , T=T , process_rank=ddp_rank , num_processes= ddp_world_size , split="train")
valid_loader = DataLoader(B= B , T=T , process_rank=ddp_rank , num_processes= ddp_world_size , split="val")

#making log file for loss and gradient checkpoint 
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

#open to reset folder 
with open(log_file, "w") as f: 
    pass


def evaluate():
    model.eval()
    valid_loader.reset()

    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = valid_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
        
    if ddp:
        dist.all_reduce(val_loss_accum , op = dist.ReduceOp.AVG)
    
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")

        with open(log_file , "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        if step > 0 and step % 5000 ==0 or last_step == step:

            checkpoint_path = os.path.join(log_dir , f"model_{step:05d}.pt")
            checkpoints = {
                "model" : raw_model.state_dict(),
                "config" : raw_model.config,
                "step" : step ,
                "val_loss" : val_loss_accum.item(), 
                "optimizer" : optimizer.state_dict()
            }

            torch.save(checkpoints , checkpoint_path)


def generate_text():
    model.eval()
    num_seq = 4
    max_len = 32
    tokens = enc.encode("Hello I'am language model,")
    tokens = torch.tensor(tokens , dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_seq , 1)

    genx   = tokens.to(device)

     
    while genx.size(1) < max_len:

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits , loss = model(genx)
            logits = logits[:,-1,:]

            probs = torch.softmax(logits , dim=-1)

            topk_probs , topk_indecs = torch.topk(probs , 50 , dim=-1)

            ix = torch.multinomial(topk_probs , 1)

            x_col = torch.gather(topk_indecs , -1 , ix)

            genx = torch.cat((genx , x_col) , dim=1)

    for i in range(num_seq):
        tokens = genx[i, :max_len].tolist()            
        decode = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i} text: {decode}")

last_step = max_steps -1

for step in range(max_steps):
    t0 = time.time()


    if step % 250 ==0 or step == last_step:
        evaluate()
        
    if (step % 250==0 and step > 0) or step == last_step:
        generate_text()

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for microstep in range(grand_accumulate_steps):
        x , y = train_loader.next_batch()
        x , y = x.to(device) , y.to(device)

        if ddp:
            model.require_backward_grad_sync = (microstep == grand_accumulate_steps -1)

        with torch.autocast(device_type= device_type , dtype=torch.bfloat16):
            logits , loss = model(x ,y)

        loss = loss/grand_accumulate_steps
        loss_accum += loss.detach()

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum , op = dist.ReduceOp.AVG)
    
    # normalize gradinet so it doesn't explode 
    norm =  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


    lr = get_lr(step)

    for param_groups in optimizer.param_groups:
        param_groups["lr"] = lr
    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    t1  = time.time()

    t = t1 - t0

    token_process = train_loader.B * train_loader.T * ddp_world_size * grand_accumulate_steps
    tokens_per_sec = token_process / t 

    if master_process:
        print(f"step: {step:5d}|| loss: {loss_accum:6f} || lr: {lr:4e} || norm: {norm:4f} || time: {t}ms || token_per_sec {tokens_per_sec}") 
if ddp:
    destroy_process_group()
