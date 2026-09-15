"""
Image cleaning utilities.

The training set has two known visual artifacts that hurt model performance
if left in place:

1. **Orange date-stamp text** burned into the corner of many photos.
   Detected in HSV space and removed via ``cv2.inpaint``.

2. **Cardboard backing** placed along the edges of some photos to isolate
   the measurement area. These images need manual per-image cropping;
   ``detect_cardboard_edge`` here is a heuristic used offline to flag
   candidates for review, not an automatic cropper (false positives would
   silently corrupt training data).
"""

import cv2
import numpy as np
from pathlib import Path


def remove_timestamp_orange(img: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Remove orange date-stamp text using HSV colour detection + inpainting.

    Args:
        img: HxWx3 uint8 RGB image.
        verbose: if True, prints when a timestamp is detected and removed.

    Returns:
        Cleaned image (same shape and dtype). Returns the input unchanged if
        no timestamp is found or if the orange coverage looks like it comes
        from vegetation or soil rather than text.
    """
    assert img.dtype == np.uint8, f"Expected uint8 RGB, got {img.dtype}"
    assert img.ndim == 3 and img.shape[2] == 3, f"Expected HxWx3, got {img.shape}"

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Orange range: narrow on hue, high saturation and value
    lower = np.array([5, 150, 150], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Dilate to cover anti-aliased glyph edges before inpainting
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    if mask.sum() == 0:
        return img

    # Real timestamps are small, concentrated regions of orange. Large orange
    # areas usually mean orange flowers or dry soil — don't inpaint those.
    h, w = mask.shape
    coverage = mask.sum() / 255 / (h * w)
    if coverage > 0.05:
        if verbose:
            print(f"  [clean] orange coverage {coverage:.1%} too high — skipping inpaint")
        return img

    if verbose:
        print(f"  [clean] removed timestamp (coverage {coverage:.3%})")
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def detect_cardboard_edge(img: np.ndarray, edge_pct: float = 0.10) -> dict:
    """
    Heuristic detector for cardboard backing along image edges.

    Inspects the outer ``edge_pct`` of each edge and flags it if the dominant
    colour is light or desaturated (typical of cardboard or paper backing)
    rather than green/brown vegetation and soil.

    This is intended for offline triage: run over the training set to
    produce a candidate list of images that need manual cropping. The
    actual crop rectangles for the flagged images are stored in
    ``outputs/manual_crops.csv``.

    Args:
        img: HxWx3 uint8 RGB image.
        edge_pct: fraction of image dimension considered "edge" (default 0.10).

    Returns:
        dict mapping edge name ('top', 'bottom', 'left', 'right') to bool.
    """
    h, w = img.shape[:2]
    eh = max(1, int(h * edge_pct))
    ew = max(1, int(w * edge_pct))

    edges = {
        "top":    img[:eh, :, :],
        "bottom": img[-eh:, :, :],
        "left":   img[:, :ew, :],
        "right":  img[:, -ew:, :],
    }

    out = {}
    for name, strip in edges.items():
        hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1].mean()
        val = hsv[..., 2].mean()
        is_cardboard = (sat < 40) or (val > 220 and sat < 80)
        out[name] = bool(is_cardboard)
    return out


def quick_clean(img_path: Path, apply_timestamp: bool = True,
                verbose: bool = False) -> np.ndarray:
    """
    Convenience loader: read an image from disk as RGB and apply the enabled
    cleanups. Useful for quick inspection or standalone processing outside
    the main training loop (which handles cleaning inline in the Dataset).

    Args:
        img_path: path to an image file readable by OpenCV.
        apply_timestamp: run ``remove_timestamp_orange`` on the loaded image.
        verbose: forwarded to ``remove_timestamp_orange``.

    Returns:
        HxWx3 uint8 RGB image with the enabled cleanups applied.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if apply_timestamp:
        img = remove_timestamp_orange(img, verbose=verbose)
    return img
