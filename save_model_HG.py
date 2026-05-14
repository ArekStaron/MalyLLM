import huggingface_hub
import glob
import os

huggingface_hub.login()


repo_id = "ArekStaron/MalyLLM"
log_dir = "log"
api = huggingface_hub.HfApi()

for file in sorted(glob.glob(os.path.join(log_dir , "model_*.pt"))):
    filename = os.path.basename(file)
    
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo= filename,
        repo_id= repo_id
    ) 
    print(f"Upload: {filename}")
