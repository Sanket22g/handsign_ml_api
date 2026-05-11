import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
api = HfApi(token=token)

try:
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/handsign_ml_api"

    print("Creating space:", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    
    print("Uploading folder...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.pyc", ".env", ".git/*", "deploy_hf.py"]
    )
    print("Deployment successful! URL: https://huggingface.co/spaces/" + repo_id)
except Exception as e:
    # Use standard print to avoid cp1252 charmap errors on windows
    print("Error:", str(e).encode('ascii', 'ignore').decode('ascii'))
