# scripts/split_index.py
import numpy as np
from pathlib import Path

INDEX_PATH  = Path("assets/list_sorted_emnist.npy")
OUT_DIGITS  = Path("assets/index_digits.npy")
OUT_LETTERS = Path("assets/index_letters.npy")

full_index = np.load(INDEX_PATH)
print(f"Full index shape: {full_index.shape}")   # should be (52000,)

half = len(full_index) // 2
index_digits  = full_index[:half]
index_letters = full_index[half:]

np.save(OUT_DIGITS,  index_digits)
np.save(OUT_LETTERS, index_letters)

print(f"Digits  index shape: {index_digits.shape}")    # (26000,)
print(f"Letters index shape: {index_letters.shape}")   # (26000,)
print(f"Digits  range: {index_digits.min()} - {index_digits.max()}")
print(f"Letters range: {index_letters.min()} - {index_letters.max()}")
print("Done.")