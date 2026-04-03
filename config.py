from dataclasses import dataclass


@dataclass
class SmallTransformerConfig:
    #Model size 
    vocab_size : int = 49152
    hidden_dim :int = 576
    intermediate_dim :int= 1536
    num_layer :int = 30 
    num_key_value_heads :int  = 3
    num_attention_heads :int = 9
    head_dim :int = 64

    max_position_embeddings :int = 2048 

    Rope_theta : int = 10000.0
