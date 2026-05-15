import huggingface_hub
import torch
from model.model import MalyLLM
from config import SmallTransformerConfig
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
config = SmallTransformerConfig()
model = MalyLLM(config)
model=model.to(device)

path = huggingface_hub.hf_hub_download(
    repo_id="ArekStaron/MalyLLM",
    filename="model_19072.pt"
)

checkpoint = torch.load(path , map_location=device , weights_only=False)
params = checkpoint["model"]


enc = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")



model.load_state_dict(params)
model.eval()

input_to_model = "Hello I'am language model,"
seq_len = 40


tokens = enc.encode(input_to_model)
tokens = torch.tensor(tokens)
tokens = tokens.unsqueeze(0)

gen = tokens.to(device)

while gen.size(1) < seq_len:
    with torch.no_grad():
        logits , loss = model(gen)
        logits = logits[:,-1,:]

        probs = torch.softmax(logits , dim=-1)

        topk_probs , topk_idx = torch.topk(probs , 50 , dim=-1)

        ix = torch.multinomial(topk_probs , 1)

        x_col = torch.gather(topk_idx, -1 , ix)

        gen = torch.cat((gen, x_col) , 1)

    tokens = gen.squeeze(0).tolist()
    

decode = enc.decode(tokens)
print(decode)
