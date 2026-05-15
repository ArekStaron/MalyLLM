import huggingface_hub

huggingface_hub.login()

repo_id = "ArekStaron/MalyLLM"
log_dir = "log"
api = huggingface_hub.HfApi()

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_folder(
    folder_path=log_dir,
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=["model_*.pt", "log.txt"],
)
