import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Define paths first
SOURCE_PATH = "/mnt/volume2/imagenet-1k"
EXTRACT_PATH = "/mnt/volume2/imagenet-extracted"
CACHE_DIR = "/mnt/volume2/huggingface_cache"

def setup_environment():
    """Setup directories and environment variables"""
    # Create our directories
    directories = [
        SOURCE_PATH,
        EXTRACT_PATH,
        CACHE_DIR,
        f'{CACHE_DIR}/datasets',
        f'{CACHE_DIR}/modules',
        f'{CACHE_DIR}/hub',
        f'{CACHE_DIR}/tmp'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Clear and set all environment variables
    os.environ.update({
        'HF_HOME': CACHE_DIR,
        'HF_DATASETS_CACHE': f'{CACHE_DIR}/datasets',
        'TRANSFORMERS_CACHE': CACHE_DIR,
        'HF_MODULES_CACHE': f'{CACHE_DIR}/modules',
        'HF_HUB_CACHE': f'{CACHE_DIR}/hub',
        'HF_DATASETS_DIR': f'{CACHE_DIR}/datasets',
        'HUGGINGFACE_HUB_CACHE': f'{CACHE_DIR}/hub',
        'TMPDIR': f'{CACHE_DIR}/tmp',
        'TEMP': f'{CACHE_DIR}/tmp',
        'TMP': f'{CACHE_DIR}/tmp'
    })

    # Clear default cache locations
    default_cache_dirs = [
        "~/.cache/huggingface",
        "~/.cache/huggingface/hub",
        "~/.cache/huggingface/datasets",
        "/home/ubuntu/.cache/huggingface",
        tempfile.gettempdir() + "/huggingface"
    ]
    
    for cache_dir in default_cache_dirs:
        cache_path = os.path.expanduser(cache_dir)
        if os.path.exists(cache_path):
            print(f"Removing default cache: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)

# Set up environment before importing HF libraries
setup_environment()

# Now import HF libraries
from datasets import load_dataset, disable_caching
from huggingface_hub import HfFolder

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# Disable caching
disable_caching()

def download_imagenet():
    """Download ImageNet-1k dataset"""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    # Set HF token
    HfFolder.save_token(HF_TOKEN)
    print("Hugging Face token configured")
    
    print(f"Downloading ImageNet-1k to {CACHE_DIR}...")
    
    # Download train split
    print("Downloading training split...")
    train_dataset = load_dataset(
        "imagenet-1k",
        split="train",
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    # Download validation split
    print("Downloading validation split...")
    val_dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    print(f"Download complete!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return CACHE_DIR

if __name__ == "__main__":
    print(f"""
    ImageNet-1k Download Configuration:
    - Cache directory: {CACHE_DIR}
    - Using HF_TOKEN: {'Yes' if HF_TOKEN else 'No'}
    """)
    
    cache_dir = download_imagenet()
    print(f"\nDataset downloaded to: {cache_dir}") 