"""Корреляционный фильтр на NumPy.

Сама свёртка с ядром написана только через numpy: без cv2.filter2D и scipy.signal.
Корреляция не отражает ядро. Граница кадра дополняется краевыми значениями.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"

KERNELS = {
    "box": np.ones((5, 5), dtype=np.float64),
    "edge": np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    ),
    "edge_center": np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    ),
}

TITLES = {
    "box": "ядро 1, усреднение",
    "edge": "ядро 2, перепад",
    "edge_center": "ядро 3, центр = 1",
}


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    total = float(kernel.sum())
    if abs(total) > 1e-12:
        return kernel / total
    return kernel / float(np.abs(kernel).sum())


def correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Корреляция изображения с ядром. Только операции NumPy."""
    kernel = normalize_kernel(np.asarray(kernel, dtype=np.float64))
    source = np.asarray(image, dtype=np.float64)
    kernel_h, kernel_w = kernel.shape
    pad_y, pad_x = kernel_h // 2, kernel_w // 2
    padded = np.pad(source, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    result = np.zeros_like(source)
    height, width = source.shape
    for row in range(kernel_h):
        for col in range(kernel_w):
            weight = kernel[row, col]
            if weight == 0.0:
                continue
            result += weight * padded[row : row + height, col : col + width]
    return result


def to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1.0, 99.0))
    if high - low < 1e-8:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image), mode="L").save(path)


def montage(panels: list[tuple[str, np.ndarray]], path: Path) -> None:
    thumb_w = 420
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    tiles = []
    for title, image in panels:
        picture = Image.fromarray(to_uint8(image), mode="L").convert("RGB")
        scale = thumb_w / picture.width
        picture = picture.resize((thumb_w, max(1, int(picture.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (picture.width, picture.height + 26), (255, 255, 255))
        canvas.paste(picture, (0, 26))
        ImageDraw.Draw(canvas).text((6, 4), title, fill=(0, 0, 0), font=font)
        tiles.append(canvas)
    sheet = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), (255, 255, 255))
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (offset, 0))
        offset += tile.width
    sheet.save(path, quality=90)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("boat1.jpg", "boat2.png"):
        image = load_gray(DATA / name)
        stem = Path(name).stem
        panels = [("исходное", image)]
        save_gray(OUT / f"{stem}_original.png", image)
        for key, kernel in KERNELS.items():
            filtered = correlate(image, kernel)
            save_gray(OUT / f"{stem}_{key}.png", filtered)
            panels.append((TITLES[key], filtered))
            print(f"{stem} / {key}: min={filtered.min():.2f} max={filtered.max():.2f}")
        montage(panels, OUT / f"{stem}_montage.jpg")
    print(f"результаты записаны в {OUT}")


if __name__ == "__main__":
    main()
