# src/dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
# from emnist import extract_training_samples
import cv2
import hashlib


# ── EMNIST loader ─────────────────────────────────────────────────────────────
def load_emnist():
    import torchvision
    import tempfile

    tmp = tempfile.mkdtemp()

    digits = torchvision.datasets.EMNIST(
        root=tmp, split="digits",
        train=True, download=True)
    idx_d          = np.argsort(digits.targets.numpy())
    sort_targets_d = digits.data.numpy()[idx_d]   # (N, 28, 28) uint8
    sort_targets_d = np.rot90(sort_targets_d, k=3, axes=(1,2))

    letters = torchvision.datasets.EMNIST(
        root=tmp, split="letters",
        train=True, download=True)
    idx_l          = np.argsort(letters.targets.numpy())
    sort_targets_l = letters.data.numpy()[idx_l]  # (N, 28, 28) uint8
    sort_targets_l = np.rot90(sort_targets_l, k=3, axes=(1,2))

    return sort_targets_d, sort_targets_l


# ── helpers ───────────────────────────────────────────────────────────────────
def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_fft2(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(image))


# ── transforms ────────────────────────────────────────────────────────────────
class SpeckleTransform:
    """
    All input transforms. Input is a uint8 numpy array of shape (256, 256).
    Output is a float32 numpy array of shape (128, 128).
    """

    def __init__(self, input_type: str, image_size: int = 128):
        self.input_type = input_type
        self.image_size = image_size
        assert input_type in ("Speckle", "AC", "FM", "CroppedFM"), \
            f"Unknown input type: {input_type}"

    def __call__(self, img: np.ndarray) -> np.ndarray:
        sz  = self.image_size
        img = img.astype(np.float32)

        if self.input_type == "Speckle":
            out = self._center_crop(img, sz)
            out = self._minmax(out)

        elif self.input_type == "AC":
            cropped = self._center_crop(img, sz)
            fft     = get_fft2(cropped)
            mag     = np.log(np.abs(fft) + 1e-8)
            mag     = np.nan_to_num(mag)
            out     = self._minmax(mag)

        elif self.input_type == "FM":
            cropped = self._center_crop(img, sz)
            fft1    = get_fft2(cropped)
            fft2    = get_fft2(np.abs(fft1))
            mag     = np.abs(fft2)
            out     = self._minmax(mag)

        elif self.input_type == "CroppedFM":
            fft1    = get_fft2(img)
            fft2    = get_fft2(np.abs(fft1))
            mag     = np.abs(fft2)
            mag     = self._minmax(mag)
            out     = self._center_crop(mag, sz)

        return out.astype(np.float32)

    @staticmethod
    def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape
        r0   = h // 2 - size // 2
        c0   = w // 2 - size // 2
        return img[r0:r0+size, c0:c0+size]

    @staticmethod
    def _minmax(img: np.ndarray) -> np.ndarray:
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return np.zeros_like(img)
        return (img - mn) / (mx - mn)


class EMNISTTransform:
    """
    Output transform. Takes a raw EMNIST uint8 image (28x28),
    resizes to image_size and binarizes.
    """

    def __init__(self, image_size: int = 128, threshold: int = 153):
        self.image_size = image_size
        self.threshold  = threshold

    def __call__(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
        binary  = (resized > self.threshold).astype(np.float32)
        return binary


# ── dataset ───────────────────────────────────────────────────────────────────
class SpeckleDataset(Dataset):
    """
    PyTorch Dataset for speckle reconstruction.

    Parameters
    ----------
    h5_path            : path to HDF5 file for one diffuser
    index_digits_path  : path to index_digits.npy  (26000 entries)
    index_letters_path : path to index_letters.npy (26000 entries)
    input_type         : one of Speckle | AC | FM | CroppedFM
    dataset_size       : how many samples to use (None = all 52000)
    image_size         : spatial size of input/output (default 128)
    threshold          : binarization threshold for EMNIST output
    expected_md5       : if provided, checks HDF5 file integrity on init
    """

    def __init__(
        self,
        h5_path:            Path,
        index_digits_path:  Path,
        index_letters_path: Path,
        input_type:         str   = "FM",
        dataset_size:       int   = None,
        image_size:         int   = 128,
        threshold:          int   = 153,
        expected_md5:       str   = None,
    ):
        self.h5_path    = Path(h5_path)
        self.input_type = input_type
        self.image_size = image_size

        # integrity check
        if expected_md5 is not None:
            actual = md5_of_file(self.h5_path)
            assert actual == expected_md5, \
                f"MD5 mismatch for {self.h5_path.name}\n" \
                f"  expected : {expected_md5}\n" \
                f"  got      : {actual}"

        # load split index files
        index_digits  = np.load(index_digits_path)    # (26000,)
        index_letters = np.load(index_letters_path)   # (26000,)

        n_digits_total  = len(index_digits)
        n_letters_total = len(index_letters)
        n_total         = n_digits_total + n_letters_total   # 52000

        # subset proportionally if dataset_size is specified
        if dataset_size is None or dataset_size >= n_total:
            self.index_digits  = index_digits
            self.index_letters = index_letters
        else:
            n_dig = dataset_size // 2
            n_let = dataset_size - n_dig
            dig_positions = np.linspace(0, n_digits_total  - 1, n_dig, dtype=int)
            let_positions = np.linspace(0, n_letters_total - 1, n_let, dtype=int)
            self.index_digits  = index_digits[dig_positions]
            self.index_letters = index_letters[let_positions]

        self.n_digits  = len(self.index_digits)
        self.n_letters = len(self.index_letters)
        self.n         = self.n_digits + self.n_letters

        # load EMNIST arrays once
        self.sort_targets_d, self.sort_targets_l = load_emnist()
        self.n_emnist_digits = len(self.sort_targets_d)   # boundary for index lookup

        # transforms
        self.input_transform  = SpeckleTransform(input_type, image_size)
        self.output_transform = EMNISTTransform(image_size, threshold)

        # open h5 file handle for fast random access
        self.h5 = h5py.File(self.h5_path, "r")

        print(f"SpeckleDataset ready — {self.n} samples "
              f"({self.n_digits} digits, {self.n_letters} letters) "
              f"| input: {input_type}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # ── input ─────────────────────────────────────────────────────────────
        speckle = self.h5["speckles"][idx]           # uint8 (256, 256)
        x = self.input_transform(speckle)            # float32 (128, 128)
        x = torch.from_numpy(x).unsqueeze(0)         # (1, 128, 128)

        # ── output ────────────────────────────────────────────────────────────
        if idx < self.n_digits:
            emnist_idx = int(self.index_digits[idx])
            pattern    = self.sort_targets_d[emnist_idx]
        else:
            emnist_idx = int(self.index_letters[idx - self.n_digits])
            pattern    = self.sort_targets_l[emnist_idx]

        y = self.output_transform(pattern)           # float32 (128, 128)
        y = torch.from_numpy(y).unsqueeze(0)         # (1, 128, 128)

        return x, y

    def __del__(self):
        if hasattr(self, "h5") and self.h5.id.valid:
            self.h5.close()


# ── dataloader factory ────────────────────────────────────────────────────────
def get_dataloaders(
    h5_path:            Path,
    index_digits_path:  Path,
    index_letters_path: Path,
    input_type:         str,
    dataset_size:       int,
    batch_size:         int   = 32,
    val_split:          float = 0.1,
    num_workers:        int   = 2,
    seed:               int   = 42,
    image_size:         int   = 128,
    threshold:          int   = 153,
    expected_md5:       str   = None,
) -> tuple:

    dataset = SpeckleDataset(
        h5_path            = h5_path,
        index_digits_path  = index_digits_path,
        index_letters_path = index_letters_path,
        input_type         = input_type,
        dataset_size       = dataset_size,
        image_size         = image_size,
        threshold          = threshold,
        expected_md5       = expected_md5,
    )

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
    )

    return train_loader, val_loader