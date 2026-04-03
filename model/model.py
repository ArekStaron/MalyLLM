import torch
import torch.nn as nn 
import torch.nn.functional as F

def precompute_rope(head_dim , max_seq_len , rope_theta):
    freq = 1 / (rope_theta ** torch.arange(0 , head_dim ,2).float() /head_dim)
    possition = torch.arange(max_seq_len)

    angles = torch.outer(possition , freq)
    angles = torch.cat([angles , angles] , dim=-1)

    return angles.cos() , angles.sin()




def rotate_half(x):
        x1 = x[... ,: x.shape[-1] //2] # (B,n_h , T, C //2) 
        x2 = x[... , x.shape[-1] //2:]
        
        return torch.cat((x1,x2), dim=-1)

def apply_rope(q , k , sin , cos):
    q = q * cos + (rotate_half(q) * sin)
    k = k * cos + (rotate_half(k) * sin)

    return q , k  


class Attencion(nn.Module):
    def __init__(self , config):
        super().__init__()
        assert config.hidden_dim % config.num_attention_heads ==0
        assert config.hidden_dim % config.head_dim == 0 
        assert config.num_attention_heads % config.num_key_value_heads == 0

        cos , sin = precompute_rope(config.head_dim , config.max_position_embeddings , config.Rope_theta)

        self.register_buffer("cos", cos)
        self.register_buffer("sin" , sin )

        self.kv_dim = config.head_dim * config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_dim ,  config.hidden_dim  , bias=False)
        
        self.kv_proj = nn.Linear(config.hidden_dim , self.kv_dim * 2 , bias= False)

        self.o_proj = nn.Linear(config.hidden_dim , config.hidden_dim)


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

        
        apply_rope(q , k ,self.sin[:T] , self.cos[:T] )

        # from (B,n_kv_heads , T , self.head_dim ) --> (B,self.n_head, T , self.head_dim)
        # For dim to match in k , v and q 
        n_repats = self.n_head //self.n_kv_heads

        k = k.repeat_interleave(n_repats , dim=1)
        v = v.repeat_interleave(n_repats , dim=1)

        y = F.scaled_dot_product_attention(q,k,v , is_causal=True) # 

        y = y.transpose(1,2).reshape(B,T,C)

        y = self.o_proj(y)

        return y 



class MLP(nn.Module):
     
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim , config.intermediate_dim , bias=False)
        self.up_proj = nn.Linear(config.hidden_dim , config.intermediate_dim , bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim , config.hidden_dim , bias=False)

    
    def forward(self , x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    





