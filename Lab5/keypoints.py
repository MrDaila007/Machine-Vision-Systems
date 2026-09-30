"""Особые точки: Харрис, поворот, сопоставление отпечатков."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"нет файла {path}")
    return image


def harris_points(gray: np.ndarray, block_size: int, ksize: int, rel: float) -> np.ndarray:
    response = cv2.cornerHarris(np.float32(gray), block_size, ksize, 0.04)
    local_max = cv2.dilate(response, np.ones((3, 3), np.float32))
    threshold = rel * float(response.max())
    ys, xs = np.where((response == local_max) & (response > threshold))
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def reference_points(path: Path) -> np.ndarray:
    color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    red = (color[:, :, 2] > 160) & (color[:, :, 1] < 90) & (color[:, :, 0] < 90)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    points = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 1:
            continue
        cx, cy = centroids[index]
        points.append((int(round(cx)), int(round(cy))))
    if not points:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(points, dtype=np.int32)


def f1_score(predicted: np.ndarray, reference: np.ndarray, tolerance: float = 6.0) -> float:
    if len(predicted) == 0 or len(reference) == 0:
        return 0.0
    if len(predicted) > 8000:
        return 0.0
    distances = np.linalg.norm(
        predicted.astype(np.float32)[:, None, :] - reference.astype(np.float32)[None, :, :],
        axis=2,
    )
    precision = float(np.mean(distances.min(axis=1) <= tolerance))
    recall = float(np.mean(distances.min(axis=0) <= tolerance))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def search_harris(gray: np.ndarray, reference: np.ndarray) -> tuple[dict, float]:
    best = {"block_size": 2, "ksize": 3, "rel": 0.01}
    best_score = -1.0
    ref_count = max(len(reference), 1)
    for block_size in (2, 3, 4, 5, 7):
        for ksize in (3, 5, 7):
            for rel in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
                points = harris_points(gray, block_size, ksize, rel)
                if not (0.25 * ref_count <= len(points) <= 3.0 * ref_count):
                    continue
                score = f1_score(points, reference)
                if score > best_score:
                    best_score = score
                    best = {"block_size": block_size, "ksize": ksize, "rel": rel}
    return best, best_score


def draw_points(gray: np.ndarray, points: np.ndarray, radius: int = 2) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 255), -1)
    return canvas


def rotate_image(gray: np.ndarray, angle: float) -> np.ndarray:
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)


def rotation_study(gray: np.ndarray, params: dict, stem: str) -> None:
    counts = []
    tiles = []
    for angle in range(360):
        rotated = rotate_image(gray, angle)
        points = harris_points(rotated, **params)
        counts.append(len(points))
        if angle % 30 == 0:
            tiles.append(draw_points(rotated, points))
    contact = make_sheet(tiles, columns=4)
    cv2.imwrite(str(OUT / f"{stem}_sheet.jpg"), contact, [cv2.IMWRITE_JPEG_QUALITY, 90])
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(range(360), counts, color="C0", linewidth=1)
    axis.set_xlabel("угол, градусы")
    axis.set_ylabel("число точек Харриса")
    axis.set_title(stem)
    figure.tight_layout()
    figure.savefig(OUT / f"{stem}_counts.png", dpi=120)
    plt.close(figure)
    np.savetxt(OUT / f"{stem}_counts.csv", np.array(counts, dtype=int), fmt="%d")
    print(f"{stem}: точки {min(counts)} … {max(counts)}, среднее {np.mean(counts):.1f}")


def make_sheet(images: list[np.ndarray], columns: int) -> np.ndarray:
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    height, width = images[0].shape[:2]
    rows = int(np.ceil(len(images) / columns))
    sheet = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return sheet


def ink_bands(profile: np.ndarray, parts: int = 3) -> list[tuple[int, int]]:
    smooth = np.convolve(profile.astype(np.float64), np.ones(21) / 21.0, mode="same")
    active = smooth > (smooth.max() * 0.08)
    spans: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(active)))
    spans = [span for span in spans if span[1] - span[0] > 40]
    if len(spans) > parts:
        largest = sorted(spans, key=lambda span: span[1] - span[0], reverse=True)[:parts]
        spans = sorted(largest)
    return spans


def template_crops(gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
    ink = (gray < 180).astype(np.uint8)
    row_spans = ink_bands(ink.sum(axis=1))
    col_spans = ink_bands(ink.sum(axis=0))
    crops = []
    index = 1
    for top, bottom in row_spans:
        for left, right in col_spans:
            crops.append((index, gray[top:bottom, left:right]))
            index += 1
    return crops


def tight_crop(gray: np.ndarray) -> np.ndarray:
    ink = np.where(gray < 180)
    if len(ink[0]) == 0:
        return gray
    top, bottom = int(ink[0].min()), int(ink[0].max()) + 1
    left, right = int(ink[1].min()), int(ink[1].max()) + 1
    return gray[top:bottom, left:right]


def sift_inliers(query: np.ndarray, template: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=1000)
    query_kp, query_desc = sift.detectAndCompute(query, None)
    template_kp, template_desc = sift.detectAndCompute(template, None)
    if (
        query_desc is None
        or template_desc is None
        or len(query_desc) < 4
        or len(template_desc) < 4
    ):
        return 0, query_kp, template_kp, []
    pairs = cv2.BFMatcher().knnMatch(query_desc, template_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.8 * second.distance:
            good.append(first)
    if len(good) < 4:
        return len(good), query_kp, template_kp, good
    source = np.float32([query_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    destination = np.float32([template_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    _homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, query_kp, template_kp, []
    inlier_matches = [item for item, flag in zip(good, mask.ravel()) if flag]
    return len(inlier_matches), query_kp, template_kp, inlier_matches


def match_queries(templates: list[tuple[int, np.ndarray]]) -> None:
    lines = []
    for name in ("fp2.jpg", "fp3.jpg"):
        query = load_gray(DATA / name)
        scores = []
        for index, crop in templates:
            template = tight_crop(crop)
            resized = cv2.resize(query, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_LINEAR)
            count, query_kp, template_kp, matches = sift_inliers(resized, template)
            scores.append((count, index, template, resized, query_kp, template_kp, matches))
        scores.sort(key=lambda item: item[0], reverse=True)
        best = scores[0]
        lines.append(f"{name}: ближайший шаблон {best[1]}, инлайеров {best[0]}")
        for count, index, *_rest in scores:
            lines.append(f"  шаблон {index}: {count}")
        match_image = cv2.drawMatches(
            best[3],
            best[4],
            best[2],
            best[5],
            best[6][:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(OUT / f"match_{Path(name).stem}.jpg"), match_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(lines[-len(scores) - 1])
    (OUT / "matches.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprints = load_gray(DATA / "fingerprints.jpg")
    reference = reference_points(DATA / "resultfp.jpg")
    params, score = search_harris(fingerprints, reference)
    points = harris_points(fingerprints, **params)
    overlay = draw_points(fingerprints, points, radius=3)
    cv2.imwrite(str(OUT / "harris_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    compared = overlay.copy()
    for x, y in reference:
        cv2.circle(compared, (int(x), int(y)), 2, (0, 255, 0), 1)
    cv2.imwrite(str(OUT / "harris_vs_reference.jpg"), compared, [cv2.IMWRITE_JPEG_QUALITY, 90])
    report = (
        f"blockSize={params['block_size']}, ksize={params['ksize']}, "
        f"порог={params['rel']:.4f} * max, точек={len(points)}, "
        f"эталон={len(reference)}, F1={score:.3f}\n"
    )
    (OUT / "harris_params.txt").write_text(report, encoding="utf-8")
    print(report.strip())

    star = load_gray(DATA / "star.jpg")
    star_params = {"block_size": 2, "ksize": 3, "rel": 0.02}
    rotation_study(star, star_params, "star")
    rotation_study(load_gray(DATA / "fp2.jpg"), params, "fp2")

    templates = template_crops(fingerprints)
    print(f"шаблонов на сетке: {len(templates)}")
    match_queries(templates)


if __name__ == "__main__":
    main()
