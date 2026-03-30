import torch
import torch.nn as nn 
import torch.nn.functional as F

class Attencion(nn.Module):
    def __init__(self , config):
        super().__init__()
        assert config.hidden_dim % config.num_attention_heads ==0
        assert config.hidden_dim % config.head_dim == 0 
        assert config.num_attention_heads % config.num_key_value_heads == 0


        self.kv_dim = config.head_dim * config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_dim ,  config.hidden_dim  , bias=False)
        
        self.kv_proj = nn.Linear(config.hidden_dim , self.kv_dim * 2 , bias= False)

        self.c_proj = nn.Linear(config.hidden_dim , config.hidden_dim)


        self.n_emdb = config.hidden_dim
        self.n_head = config.num_attention_heads
        self.n_kv_heads= config.num_key_value_heads
        self.head_dim = config.head_dim


    def forward(self , x):

        B , T , C = x.shape

        kv = self.kv_proj(x)
        

        q = self.q_proj(x)


        k , v = kv.split(self.kv_dim , dim=-1)
    

        q = q.view(B, T ,self.n_head , self.head_dim ).transpose(1,2) 
        k = k.view(B,T , self.n_kv_heads , self.head_dim).transpose(1,2)
        v = v.view(B,T , self.n_kv_heads , self.head_dim).transpose(1,2)


        # from (B,n_kv_heads , T , self.head_dim ) --> (B,self.n_head, T , self.head_dim)
        # For dim to match in k , v and q 
        n_repats = self.n_head //self.n_kv_heads

        k = k.repeat_interleave(n_repats , dim=1)
        v = v.repeat_interleave(n_repats , dim=1)

        y = F.scaled_dot_product_attention(q,k,v , is_causal=True) # 

        y = y.transpose(1,2).reshape(B,T,C)

        return y 

        