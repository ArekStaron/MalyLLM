from dataclasses import dataclass


@dataclass
class SmallTransformerConfig:
    #Model size 
    vocab_size : int = 49152
    hidden_dim = 576
    intermediate_dim = 1536
    num_layer = 30 
    num_key_value_heads  = 3
    num_attention_heads = 9
    head_dim = 64

    max_position_embeddings = 2048 
