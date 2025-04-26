from huggingface_hub import snapshot_download

repo_id = "L0uu/final_model_isi"  # Replace with the actual LoRA repo (e.g., "peft/lora-bert")
local_dir = "./lora-adapter2"     # Directory to save the adapter

snapshot_download(repo_id=repo_id, local_dir=local_dir)