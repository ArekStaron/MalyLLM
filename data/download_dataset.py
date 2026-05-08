import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_dataset
from transformers import AutoTokenizer



OUT_DIR = "data/numpy_data"
os.makedirs(OUT_DIR, exist_ok=True)

nprocs = max(1, os.cpu_count()//2)
SHARD_SIZE = 100_000


def all_thead_tokenizer():
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

def tokenize(example):
        tokens = tokenizer.encode(example["text"])
        tokens_np = np.array(tokens , dtype=np.uint16)
        return tokens_np

if __name__ == "__main__":
    
    data = load_dataset("HuggingFaceFW/fineweb-edu",
                        name="sample-10BT",
                        split="train[:500]")


    current_tokens = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    shard_idx = 0
    progress = tqdm(total=SHARD_SIZE, desc=f"Shard {shard_idx}", unit="tok")
    prefix = "train" if shard_idx > 0 else "valid"

    with mp.Pool(nprocs , initializer=all_thead_tokenizer) as pool:
        for tokens in pool.imap(tokenize,data , chunksize = 16):
            needed = len(tokens)
        
            while needed > 0:
                space = SHARD_SIZE - token_count
                to_write = min(space , needed)
                offset = len(tokens) - needed

                current_tokens[token_count:token_count + to_write] = tokens[offset:offset + to_write]
                token_count += to_write
                needed -= to_write
                progress.update(to_write)

                if token_count == SHARD_SIZE:
                    np.save(os.path.join(OUT_DIR, f"{prefix}_{shard_idx:04d}.npy"), current_tokens)
                    print(f"\nSaved shard {shard_idx}")
                    shard_idx += 1
                    token_count = 0
                    progress = tqdm(total=SHARD_SIZE, desc=f"Shard {shard_idx}", unit="tok")

        
        if token_count > 0:
            np.save(os.path.join(OUT_DIR, f"shard_{shard_idx:04d}.npy"), current_tokens[:token_count])
            print(f"Saved final shard {shard_idx} ({token_count:,} tokens)")


    