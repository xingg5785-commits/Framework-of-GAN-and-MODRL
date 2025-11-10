import os, random, pandas as pd, torch
from tqdm import tqdm
from datasets import load_dataset

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load datasets in streaming mode
    level_stream = load_dataset(
        "TheGreatRambler/mm2_level",
        split="train",
        streaming=True,
        cache_dir="F:/hf_cache"
    )
    comments_stream = load_dataset(
        "TheGreatRambler/mm2_level_comments",
        split="train",
        streaming=True,
        cache_dir="F:/hf_cache"
    )

    # Step 1: Collect 5000 levels and store in a dict keyed by 'data_id'
    level_dict = {}
    for i, lvl in enumerate(level_stream):
        if i >= 5000:
            break
        level_dict[lvl['data_id']] = lvl

    # Step 2: Collect comments that match levels in level_dict (same 'data_id')
    merged_rows = []
    for cmt in tqdm(comments_stream, desc="Merging comments"):
        data_id = cmt.get('data_id')
        if data_id in level_dict:
            row = {**level_dict[data_id], **cmt}  # merge dicts
            merged_rows.append(row)
        if len(merged_rows) >= 2000:
            break

    # Step 3: Export
    df_merged = pd.DataFrame(merged_rows)
    out_path = "F:/mm2_clean_merged_2000.csv"
    df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved cleaned merged dataset to: {out_path}")

if __name__ == "__main__":
    main()
