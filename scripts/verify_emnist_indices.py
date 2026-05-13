import numpy as np
import matplotlib.pyplot as plt
import zipfile
import gzip
from pathlib import Path

ASSETS_DIR = Path("assets")
ZIP_PATH   = Path(r"C:\Users\cesar\.cache\emnist\emnist.zip")

def read_idx(data: bytes) -> np.ndarray:
    """Parse IDX binary format used by EMNIST."""
    import struct
    magic    = struct.unpack('>I', data[:4])[0]
    n_dims   = magic & 0xFF
    dims     = struct.unpack('>' + 'I' * n_dims, data[4:4 + 4 * n_dims])
    offset   = 4 + 4 * n_dims
    dtype    = {0x08: np.uint8, 0x09: np.int8,
                0x0B: np.int16, 0x0C: np.int32,
                0x0D: np.float32, 0x0E: np.float64}[(magic >> 8) & 0xFF]
    return np.frombuffer(data[offset:], dtype=dtype).reshape(dims)

def load_split(zip_path: Path, split: str) -> tuple:
    with zipfile.ZipFile(zip_path) as zf:
        img_name = f"gzip/emnist-{split}-train-images-idx3-ubyte.gz"
        lbl_name = f"gzip/emnist-{split}-train-labels-idx1-ubyte.gz"
        with zf.open(img_name) as f:
            images = read_idx(gzip.decompress(f.read()))
        with zf.open(lbl_name) as f:
            labels = read_idx(gzip.decompress(f.read()))
    return images, labels

# ── load digits ───────────────────────────────────────────────────────────────
print("Loading digits...")
images_d, labels_d = load_split(ZIP_PATH, "digits")
idx_d          = np.argsort(labels_d)
sort_targets_d = images_d[idx_d]
sort_labels_d  = labels_d[idx_d]
print(f"Digits  shape : {sort_targets_d.shape}")
print(f"Labels  range : {sort_labels_d.min()} - {sort_labels_d.max()}")

# ── load letters ──────────────────────────────────────────────────────────────
print("Loading letters...")
images_l, labels_l = load_split(ZIP_PATH, "letters")
idx_l          = np.argsort(labels_l)
sort_targets_l = images_l[idx_l]
sort_labels_l  = labels_l[idx_l]
print(f"Letters shape : {sort_targets_l.shape}")
print(f"Labels  range : {sort_labels_l.min()} - {sort_labels_l.max()}")

# ── load index files ──────────────────────────────────────────────────────────
index_digits  = np.load(ASSETS_DIR / "index_digits.npy")
index_letters = np.load(ASSETS_DIR / "index_letters.npy")
print(f"\nDigits  index — shape: {index_digits.shape}  "
      f"range: [{index_digits.min()}, {index_digits.max()}]")
print(f"Letters index — shape: {index_letters.shape}  "
      f"range: [{index_letters.min()}, {index_letters.max()}]")

# ── verify labels ─────────────────────────────────────────────────────────────
print("\nVerifying digit labels for first 5 indices:")
for i in range(5):
    emnist_idx = int(index_digits[i])
    label      = sort_labels_d[emnist_idx]
    print(f"  index_digits[{i}] = {emnist_idx} → label: {label}  "
          f"(should be 0)")

print("\nVerifying digit labels at position 2600 (class 1):")
for i in range(2600, 2605):
    emnist_idx = int(index_digits[i])
    label      = sort_labels_d[emnist_idx]
    print(f"  index_digits[{i}] = {emnist_idx} → label: {label}  "
          f"(should be 1)")

print("\nVerifying letter labels for first 5 indices:")
for i in range(5):
    emnist_idx = int(index_letters[i])
    label      = sort_labels_l[emnist_idx]
    print(f"  index_letters[{i}] = {emnist_idx} → label: {label}  "
          f"(should be 1 = a)")

print("\nVerifying letter labels at position 1000 (class 2 = b):")
for i in range(1000, 1005):
    emnist_idx = int(index_letters[i])
    label      = sort_labels_l[emnist_idx]
    print(f"  index_letters[{i}] = {emnist_idx} → label: {label}  "
          f"(should be 2 = b)")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i in range(5):
    emnist_idx = int(index_digits[i])
    axes[0, i].imshow(sort_targets_d[emnist_idx], cmap="gray")
    axes[0, i].set_title(f"idx {emnist_idx}\nlabel={sort_labels_d[emnist_idx]}")
    axes[0, i].axis("off")

for i in range(5):
    emnist_idx = int(index_letters[i])
    axes[1, i].imshow(sort_targets_l[emnist_idx], cmap="gray")
    axes[1, i].set_title(f"idx {emnist_idx}\nlabel={sort_labels_l[emnist_idx]}")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Digits\n(should be 0s)",
                       fontsize=10, rotation=0,
                       labelpad=60, va="center")
axes[1, 0].set_ylabel("Letters\n(should be a's)",
                       fontsize=10, rotation=0,
                       labelpad=60, va="center")

plt.suptitle("EMNIST verification — direct zip read (local)", fontsize=13)
plt.tight_layout()
plt.savefig("assets/emnist_verification_local.png", dpi=150)
plt.show()
print("\nSaved to assets/emnist_verification_local.png")