"""
Dataset and augmentation pipeline for CSIRO Biomass.

Per-sample pipeline (see ``BiomassDataset.__getitem__``):

    1. Load original image (~1000x2000, landscape).
    2. Trim ``config.DEFAULT_EDGE_TRIM_PX`` pixels from each edge.
    3. Apply per-image manual crop if listed in ``manual_crops.csv``
       (see ``clean.detect_cardboard_edge`` for how these are identified).
    4. Remove orange date-stamp text if
       ``config.APPLY_TIMESTAMP_REMOVAL`` is on.
    5. Apply the augmentation transform (train or valid). Augmentation
       operates on the FULL image before the L/R split so both halves
       receive identical geometric transforms.
    6. Split into a left square and a right square, each 3 x IMG_SIZE
       x IMG_SIZE.

The model receives ``left`` and ``right`` separately and sums the feature
representations after the backbone (see ``model.BiomassModel``).
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from clean import remove_timestamp_orange


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MANUAL_CROPS_PATH = config.OUTPUT_DIR / "manual_crops.csv"


def load_manual_crops(path=MANUAL_CROPS_PATH) -> dict:
    """
    Load per-image manual crop rectangles from ``manual_crops.csv``.

    The CSV must have columns ``image_path, top, bottom, left, right`` where
    each numeric value is a pixel count trimmed from that side (applied
    AFTER the universal edge trim).

    Args:
        path: path to the manual_crops CSV.

    Returns:
        dict mapping ``image_path`` (str) to ``(top, bottom, left, right)``
        tuple of ints. Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"image_path", "top", "bottom", "left", "right"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    crops = {}
    for _, row in df.iterrows():
        crops[row["image_path"]] = (
            int(row["top"]), int(row["bottom"]),
            int(row["left"]), int(row["right"]),
        )
    return crops


def get_train_transforms(img_size: int) -> A.Compose:
    """
    Build the training augmentation pipeline.

    Output shape is ``(img_size, 2*img_size)`` so the L/R split downstream
    produces two square inputs. Spatial augmentations are applied to the
    full 2:1 image so both halves receive identical geometric transforms.

    Args:
        img_size: side length of each half after L/R split (e.g. 1024).

    Returns:
        ``albumentations.Compose`` producing a ``(3, H, W)`` tensor via
        ``ToTensorV2``.
    """
    H, W = img_size, 2 * img_size
    augs = []

    # Spatial: probabilistic RandomResizedCrop wrapped inside OneOf with a
    # plain Resize. A.OneOf samples one child with weights proportional to
    # inner ``p`` values, so equal p=1.0 gives a 50/50 mix between RRC and
    # Resize. Both produce the same output shape.
    if config.USE_RANDOM_RESIZED_CROP:
        augs.append(A.OneOf([
            A.RandomResizedCrop(
                size=(H, W), scale=config.RRC_SCALE,
                ratio=(1.95, 2.05),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.Resize(H, W, interpolation=cv2.INTER_CUBIC, p=1.0),
        ], p=1.0))
    else:
        augs.append(A.Resize(H, W, interpolation=cv2.INTER_CUBIC))

    augs.extend([
        A.HorizontalFlip(p=config.P_HFLIP),
        A.VerticalFlip(p=config.P_VFLIP),
    ])

    if config.USE_SMALL_ROTATION:
        augs.append(A.Rotate(
            limit=(-config.ROTATION_LIMIT_DEG, config.ROTATION_LIMIT_DEG),
            p=config.ROTATION_PROB,
            border_mode=cv2.BORDER_REFLECT,
        ))

    augs.extend([
        A.RandomGamma(gamma_limit=(80, 120), p=config.P_GAMMA),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2,
            p=config.P_BRIGHTNESS_CONTRAST,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20,
            p=config.P_HSV,
        ),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.Equalize(p=0.5),
        ], p=config.P_CLAHE_OR_EQ),
        A.OneOf([
            A.Sharpen(p=0.5),
            A.Emboss(p=0.5),
        ], p=config.P_SHARPEN_OR_EMBOSS),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return A.Compose(augs)


def get_valid_transforms(img_size: int) -> A.Compose:
    """
    Build the validation/inference transform: deterministic resize +
    normalize + tensor conversion. No augmentation.

    Args:
        img_size: side length of each half after L/R split.

    Returns:
        ``albumentations.Compose`` producing a ``(3, H, W)`` tensor.
    """
    return A.Compose([
        A.Resize(img_size, 2 * img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def make_image_label_table(train_csv_path) -> pd.DataFrame:
    """
    Load the competition train.csv and pivot to one-row-per-image with all
    5 target columns joined on.

    The raw CSV has one row per (image, target) pair. This function pivots
    the targets into columns and preserves image-level metadata (state,
    date, species, NDVI, height) as sibling columns.

    Args:
        train_csv_path: path to the competition train.csv.

    Returns:
        DataFrame with one row per image containing:
            ``image_path``, ``Sampling_Date``, ``State``, ``Species``,
            ``Pre_GSHH_NDVI``, ``Height_Ave_cm``, plus the 5 target columns
            ``Dry_Green_g``, ``Dry_Dead_g``, ``Dry_Clover_g``, ``GDM_g``,
            ``Dry_Total_g``.
    """
    df = pd.read_csv(train_csv_path)
    meta_cols = ["image_path", "Sampling_Date", "State", "Species",
                 "Pre_GSHH_NDVI", "Height_Ave_cm"]
    targets = df.pivot_table(
        index="image_path", columns="target_name", values="target", aggfunc="first"
    )
    meta = df[meta_cols].drop_duplicates(subset="image_path").set_index("image_path")
    return meta.join(targets).reset_index()


class BiomassDataset(Dataset):
    """
    Image dataset yielding L/R halves and the 3 atomic biomass targets.

    Each ``__getitem__`` returns a dict:

    * ``left``:  ``(3, img_size, img_size)`` tensor — left half
    * ``right``: ``(3, img_size, img_size)`` tensor — right half
    * ``targets``: ``(3,)`` tensor — [Dry_Green_g, Dry_Dead_g, Dry_Clover_g]
    * ``image_id``: str — original ``image_path``

    See the module docstring for the full per-sample pipeline.
    """

    def __init__(self, df: pd.DataFrame, image_dir, transform,
                 apply_timestamp_removal: bool = True,
                 manual_crops: dict = None,
                 img_size: int = None,
                 strict_assert: bool = True):
        """
        Args:
            df: DataFrame with one row per image, must contain
                ``image_path``, ``Dry_Green_g``, ``Dry_Dead_g``,
                ``Dry_Clover_g``. Typically from
                ``make_image_label_table``.
            image_dir: root directory that ``image_path`` values are
                relative to.
            transform: an ``albumentations`` transform (typically from
                ``get_train_transforms`` or ``get_valid_transforms``).
            apply_timestamp_removal: run ``remove_timestamp_orange`` on
                each loaded image.
            manual_crops: dict from ``load_manual_crops`` mapping
                ``image_path`` -> (top, bottom, left, right) pixel counts
                to trim.
            img_size: side length of each half after L/R split. Defaults
                to ``config.IMG_SIZE``.
            strict_assert: enable defensive assertions on image geometry
                (aspect ratio, output shape). Recommended on for training,
                may be turned off for inference on unusual inputs.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.apply_timestamp_removal = apply_timestamp_removal
        self.manual_crops = manual_crops or {}
        self.img_size = img_size or config.IMG_SIZE
        self.strict_assert = strict_assert

    def __len__(self):
        return len(self.df)

    def _load_clean(self, rel_path):
        """Load an image from disk and apply pre-transform preprocessing."""
        full_path = self.image_dir / rel_path
        img = cv2.imread(str(full_path))
        if img is None:
            raise FileNotFoundError(f"Could not read {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        d = config.DEFAULT_EDGE_TRIM_PX
        if d > 0:
            assert img.shape[0] > 2 * d and img.shape[1] > 2 * d, (
                f"{rel_path}: image too small for {d}px default trim"
            )
            img = img[d:-d, d:-d]

        if rel_path in self.manual_crops:
            t, b, l, r = self.manual_crops[rel_path]
            h, w = img.shape[:2]
            assert h > t + b and w > l + r, (
                f"{rel_path}: manual crop ({t},{b},{l},{r}) larger than "
                f"image (already-trimmed shape {h}x{w})"
            )
            img = img[t : h - b if b > 0 else h, l : w - r if r > 0 else w]

        if self.strict_assert:
            h, w = img.shape[:2]
            assert w >= h, f"{rel_path}: expected landscape, got {w}x{h}"
            ratio = w / h
            assert 1.8 <= ratio <= 2.2, (
                f"{rel_path}: aspect {ratio:.2f} outside [1.8, 2.2]"
            )

        if self.apply_timestamp_removal:
            img = remove_timestamp_orange(img)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_clean(row["image_path"])

        out = self.transform(image=img)["image"]

        if self.strict_assert:
            assert out.shape == (3, self.img_size, 2 * self.img_size), (
                f"After transform expected (3, {self.img_size}, {2*self.img_size}), "
                f"got {tuple(out.shape)}"
            )

        left = out[:, :, : self.img_size].contiguous()
        right = out[:, :, self.img_size:].contiguous()

        targets = torch.tensor([
            row["Dry_Green_g"],
            row["Dry_Dead_g"],
            row["Dry_Clover_g"],
        ], dtype=torch.float32)

        return {
            "left":    left,
            "right":   right,
            "targets": targets,
            "image_id": row["image_path"],
        }
