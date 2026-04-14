import torch
import torch.nn as nn 
import torch.nn.functional as F

def precompute_rope(head_dim , max_seq_len , rope_theta):
    freq = 1 / (rope_theta ** torch.arange(0 , head_dim ,2).float() /head_dim)
    position = torch.arange(max_seq_len)

    angles = torch.outer(position , freq)
    angles = torch.cat([angles , angles] , dim=-1)

    return angles.cos() , angles.sin()




def rotate_half(x):
    x1 = x[... ,: x.shape[-1] //2] 
    x2 = x[... , x.shape[-1] //2:]
        
    return torch.cat((-x2,x1), dim=-1)

def apply_rope(q , k , sin , cos):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    
    q = q * cos + (rotate_half(q) * sin)
    k = k * cos + (rotate_half(k) * sin)

    return q , k  


class Attention(nn.Module):
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

        self.o_proj = nn.Linear(config.hidden_dim , config.hidden_dim , bias=False)
        
        self.o_proj.SCALE_INIT = True

        self.n_embd = config.hidden_dim
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

        
        q , k = apply_rope(q , k ,self.sin[:T] , self.cos[:T] )

        # from (B,n_kv_heads , T , self.head_dim ) --> (B,self.n_head, T , self.head_dim)
        # For dim to match in k , v and q 
        n_repeats = self.n_head //self.n_kv_heads

        k = k.repeat_interleave(n_repeats , dim=1)
        v = v.repeat_interleave(n_repeats , dim=1)

        y = F.scaled_dot_product_attention(q,k,v , is_causal=True) 

        y = y.transpose(1,2).reshape(B,T,C)

        y = self.o_proj(y)

        return y 



class MLP(nn.Module):
     
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim , config.intermediate_dim , bias=False)
        self.up_proj = nn.Linear(config.hidden_dim , config.intermediate_dim , bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim , config.hidden_dim , bias=False)

        self.down_proj.SCALE_INIT = True
    
    def forward(self , x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    


class RMSNorm(nn.Module):

    def __init__(self, hidden_size ,eps : float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x): 
        rms = torch.sqrt(torch.mean(x**2 , dim=-1 , keepdim=True) + self.eps)

        return (x / rms) * self.weight


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rms1 = RMSNorm(config.hidden_dim)
        self.rms2 = RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        x = x + self.mlp(self.rms2(x))

        return x 
    

class MalyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(config.vocab_size , config.hidden_dim),
            blocks = nn.ModuleList(Block(config) for _ in range(config.num_layer)),     
            f_rms = RMSNorm(config.hidden_dim)
        
        ))
        
        self.lm_head  = nn.Linear(config.hidden_dim , config.vocab_size , bias=False)

        self.lm_head.weight = self.transformer.embedding.weight     

        self.apply(self.__initweight)

    def __initweight(self, module):
        std = 0.02 

        if isinstance(module , nn.Linear):
            if hasattr(module , "SCALE_INIT"):
                std *= (2 * self.config. num_layer)**-0.5

            nn.init.normal_(module.weight , mean=0.0 , std = std)

            if module.bias is not None:
                nn.init.zeros_(module.bias,)
        
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0 , std=std)
    
        
    def forward(self , idx , target = None):
        B , T = idx.size()

        # Make decorator to RoPE to remove this line
        assert T<= self.config.max_position_embeddings

        x = self.transformer.embedding(idx)

        for block in self.transformer.blocks:
            x = block(x)
        
        x = self.transformer.f_rms(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        return logits , loss
    
    