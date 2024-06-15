from huggingface_hub import HfApi, HfFolder,login

# Initialize the HfApi
api = HfApi()

# Login using your API token
login(token="")

# Upload folder to Hugging Face
api.upload_folder(
    folder_path="./results",            # Path to your local model folder
    path_in_repo="",        # Path in your Hugging Face repository
    repo_id="ERPFTW/odoo1",  # Your Hugging Face username and repository name
    repo_type="model",                  # Type of the repository (usually "model")
)
