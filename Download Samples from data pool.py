# Clean startup Hugging Face downloader (F: drive enforced)
import os
import random
import pandas as pd
import multiprocessing
import torch
from tqdm import tqdm
from datasets import load_dataset, DownloadConfig
import time
import tempfile
import shutil
import ctypes

# ========== Ensure variables before imports ==========
f_drive = "F:/hf_cache"
tempdir = "F:/hf_tmp"
os.makedirs(f_drive, exist_ok=True)
os.makedirs(tempdir, exist_ok=True)

# Setup envs (must happen before import load_dataset)
os.environ["HF_HOME"] = f_drive
os.environ["HF_DATASETS_CACHE"] = f_drive
os.environ["HF_MODULES_CACHE"] = f_drive
os.environ["TRANSFORMERS_CACHE"] = f_drive
os.environ["HF_METRICS_CACHE"] = f_drive
os.environ["HF_HUB_CACHE"] = f_drive
os.environ["XDG_CACHE_HOME"] = f_drive
os.environ["TMP"] = tempdir
os.environ["TEMP"] = tempdir

os.environ["HF_DATASETS_DOWNLOAD_NUM_PROC"] = str(multiprocessing.cpu_count())
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ========== Confirm configuration ==========
print("tempfile.gettempdir() =>", tempfile.gettempdir())
print("os.getenv('HF_HOME') =>", os.getenv("HF_HOME"))
print("~/.cache/huggingface resolves to:", os.path.expanduser("~/.cache/huggingface"))
print("Checking if cache is symlink:", os.path.islink(os.path.expanduser("~/.cache/huggingface")))
print("C: drive available:", shutil.disk_usage("C:/").free // (1024 * 1024), "MB")
print("F: drive available:", shutil.disk_usage("F:/").free // (1024 * 1024), "MB")

# ========== Device detection ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ========== Robust download config ==========
download_config = DownloadConfig(
    resume_download=True,
    use_etag=True,
    max_retries=10,
    local_files_only=False,
    cache_dir=f_drive
)

# ========== Safe loading ==========
def safe_load_dataset(name, split):
    for attempt in range(1, 6):
        try:
            return load_dataset(name, split=split, download_config=download_config)
        except Exception as e:
            print(f"\u26a0\ufe0f Attempt {attempt} failed: {e}")
            time.sleep(10)
    raise RuntimeError(f"\u274c Failed to load {name} after 5 attempts")

# ========== Load datasets ==========
print("Loading level dataset...")
level_ds = safe_load_dataset("TheGreatRambler/mm2_level", "train")
print("Loading comments dataset...")
comments_ds = safe_load_dataset("TheGreatRambler/mm2_level_comments", "train")

# ========== Sample and merge ==========
sample_size = 5000
random.seed(42)
total = min(len(level_ds), len(comments_ds))
indices = random.sample(range(total), sample_size)

print("Sampling and merging...")
level_data = [level_ds[i] for i in tqdm(indices, desc="Level")]
comments_data = [comments_ds[i] for i in tqdm(indices, desc="Comments")]

level_df = pd.DataFrame(level_data)
comments_df = pd.DataFrame(comments_data)
merged_df = pd.concat([level_df.reset_index(drop=True), comments_df.reset_index(drop=True)], axis=1)

output_path = "F:/mm2_merged_sampled_5000.csv"
merged_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\u2705 Saved merged dataset to: {output_path}")