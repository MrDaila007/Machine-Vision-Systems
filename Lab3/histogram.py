"""Подбор яркостного окна для рентгенограммы сварного шва."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
DICOM_NAME = "0.063-1_2.5x_View2.dcm"


def load_xray(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float64)
    if pixels.ndim == 3:
        pixels = pixels[0]
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    return pixels


def window_to_uint8(pixels: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    low, high = np.percentile(pixels, (low_pct, high_pct))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8), float(low), float(high)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pixels = load_xray(DATA / DICOM_NAME)
    wide, wide_low, wide_high = window_to_uint8(pixels, 0.5, 99.5)
    tight, tight_low, tight_high = window_to_uint8(pixels, 5.0, 95.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(tight)

    cv2.imwrite(str(OUT / "weld_wide.jpg"), wide, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUT / "weld.jpg"), clahe, [cv2.IMWRITE_JPEG_QUALITY, 92])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pixels.ravel(), bins=256, color="0.3")
    axes[0].axvline(tight_low, color="C1", label=f"окно {tight_low:.0f}")
    axes[0].axvline(tight_high, color="C1", label=f"{tight_high:.0f}")
    axes[0].set_title("исходные яркости")
    axes[0].legend(fontsize=8)
    axes[1].hist(clahe.ravel(), bins=256, color="0.2")
    axes[1].set_title("после окна 5–95% и CLAHE")
    for axis in axes:
        axis.set_xlabel("яркость")
        axis.set_ylabel("пиксели")
    figure.tight_layout()
    figure.savefig(OUT / "histograms.png", dpi=120)
    plt.close(figure)

    print(f"размер: {pixels.shape}, min={pixels.min():.1f}, max={pixels.max():.1f}")
    print(f"широкое окно: {wide_low:.1f} … {wide_high:.1f}")
    print(f"рабочее окно: {tight_low:.1f} … {tight_high:.1f}")
    print(f"шов: {OUT / 'weld.jpg'}")


if __name__ == "__main__":
    main()
