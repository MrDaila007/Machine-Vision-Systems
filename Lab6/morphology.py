"""Математическая морфология: зёрна и волокно."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def colorize(labels: np.ndarray) -> np.ndarray:
    count = int(labels.max())
    rng = np.random.default_rng(1)
    palette = rng.integers(40, 255, size=(count + 1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]


def caption(image: np.ndarray, text: str) -> np.ndarray:
    picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    draw.text((8, 8), text, fill=(255, 255, 255), font=ImageFont.truetype(FONT, 22))
    return cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2BGR)


def separate_grains(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    peaks = peak_local_max(distance, min_distance=14, labels=opened, exclude_border=False)
    markers = np.zeros(distance.shape, dtype=np.int32)
    for index, (y, x) in enumerate(peaks, start=1):
        markers[int(y), int(x)] = index
    if markers.max() == 0:
        count, markers = cv2.connectedComponents(opened)
        return markers
    return watershed(-distance, markers, mask=opened.astype(bool))


def grains() -> None:
    color = cv2.imread(str(DATA / "sample10.jpg"), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    labels = separate_grains(gray)
    count = int(labels.max())
    painted = colorize(labels)
    painted = caption(painted, f"зёрен: {count}")
    binary = ((labels > 0).astype(np.uint8)) * 255
    cv2.imwrite(str(OUT / "grains_binary.png"), binary)
    cv2.imwrite(str(OUT / "grains_colored.png"), painted)
    print(f"зёрен: {count}")


def skeleton_length(skeleton: np.ndarray) -> float:
    points = set(zip(*np.where(skeleton)))
    length = 0.0
    for y, x in points:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (y + dy, x + dx) in points:
                length += float(np.hypot(dy, dx))
    return length


def measure_component(mask: np.ndarray) -> tuple[float, float, np.ndarray]:
    skeleton = skeletonize(mask.astype(bool))
    length = skeleton_length(skeleton)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if skeleton.any() and length > 0:
        width = float(2.0 * distance[skeleton].mean())
    else:
        width = 0.0
    return length, width, skeleton


def fiber() -> None:
    gray = cv2.imread(str(DATA / "sample29_cut.jpg"), cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    report = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        component = labels == index
        length, width, skeleton = measure_component(component)
        keep[component] = 255
        report.append((area, length, width))
        print(f"волокно area={area}  длина={length:.1f} px  ширина={width:.2f} px")
    if keep.any():
        length, width, skeleton = measure_component(keep > 0)
    else:
        length, width, skeleton = 0.0, 0.0, np.zeros_like(binary, dtype=bool)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[keep > 0] = (80, 80, 255)
    overlay[skeleton] = (0, 220, 0)
    overlay = caption(overlay, f"длина {length:.0f} px, ширина {width:.1f} px")
    cv2.imwrite(str(OUT / "fiber_mask.png"), keep)
    cv2.imwrite(str(OUT / "fiber_overlay.png"), overlay)
    lines = [f"компонент {i}: площадь={area}, длина={length:.2f}, ширина={width:.2f}" for i, (area, length, width) in enumerate(report, start=1)]
    lines.append(f"всего: длина={length:.2f} px, средняя ширина={width:.2f} px")
    (OUT / "fiber_measures.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grains()
    fiber()
    print(f"результаты: {OUT}")


if __name__ == "__main__":
    main()
